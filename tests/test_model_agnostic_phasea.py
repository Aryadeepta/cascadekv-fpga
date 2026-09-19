import math
from dataclasses import replace
import pytest
import torch

from cascadekv.model_agnostic import *
from cascadekv.adaptive_lifting import StreamingLiftingForest
from cascadekv.evaluation_core import (candidate_budget, exact_attention_reference,
    exact_rerank_and_output, hierarchy_route, initialize_route, advance_until_budget,
    query_to_kv_head, reserve_ids, select_flat_k4, traffic)


def cells(g): return [(l, h) for l in g.sampled_layers for h in range(g.num_kv_heads)]

def test_geometry_mapping_and_8k_shape():
    assert [QWEN_GEOMETRY.kv_head_for_query(x) for x in range(16)] == [x // 2 for x in range(16)]
    assert [PHI35_GEOMETRY.kv_head_for_query(x) for x in range(32)] == list(range(32))
    assert PHI35_GEOMETRY.schedule_cell_count == 160
    assert len(PHI35_GEOMETRY.sampled_positions) * PHI35_GEOMETRY.num_q_heads == 96
    with pytest.raises(ValueError): ModelGeometry("phi3", 96, 1, 3, 2, 32, 8, (0,), (1,))

@pytest.mark.parametrize(("q_heads","kv_heads","expected"), [(16,8,[x//2 for x in range(16)]),(32,32,list(range(32))),(32,8,[x//4 for x in range(32)])])
def test_generic_query_to_kv_mapping(q_heads,kv_heads,expected):
    g=ModelGeometry("phi3",q_heads*16,2,q_heads,kv_heads,16,32,(0,),(31,))
    assert [query_to_kv_head(g,h) for h in range(q_heads)] == expected

def test_accounting_k4_and_g4_partitions():
    a, b = traffic_accounting(128), traffic_accounting(96)
    assert (a.fp16_key_bytes, a.fp16_value_bytes, a.k4_groups, a.k4_key_bytes) == (256, 256, 8, 80)
    assert (b.fp16_key_bytes, b.fp16_value_bytes, b.k4_groups, b.k4_key_bytes) == (192, 192, 6, 60)
    for dim, groups, width in ((128, k4_group_partitions(128), 32), (96, k4_group_partitions(96), 24)):
        assert sorted(x for g in groups for x in g) == list(range(dim))
        assert all(len(g) == 16 for g in groups)
        vg = g4_partitions(dim); assert sorted(x for g in vg for x in g) == list(range(dim)); assert all(len(g) == width for g in vg)

def test_schedule_and_profile_fail_closed():
    validate_static_schedule(QWEN_GEOMETRY, cells(QWEN_GEOMETRY)); validate_static_schedule(PHI35_GEOMETRY, cells(PHI35_GEOMETRY))
    for bad in (cells(QWEN_GEOMETRY)[:-1], cells(QWEN_GEOMETRY) + [cells(QWEN_GEOMETRY)[0]], cells(QWEN_GEOMETRY)[:-1] + [(99, 0)]):
        with pytest.raises(ValueError): validate_static_schedule(QWEN_GEOMETRY, bad)
    with pytest.raises(ValueError): require_routing_profile(PHI35_GEOMETRY, None)
    assert PHI35_ROUTING_PROFILE_UNFROZEN == "PHI35_ROUTING_PROFILE_UNFROZEN"
    valid=RoutingProfile("qwen3", "synthetic-frozen", {x:.1 for x in QWEN_GEOMETRY.sampled_layers}, {"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","sink":16,"local":64})
    assert require_routing_profile(QWEN_GEOMETRY, valid).profile_id == "synthetic-frozen"
    for broken in (RoutingProfile("phi3","x",{},{}), RoutingProfile("qwen3","x",{0:.1},{"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","sink":-1,"local":1}), RoutingProfile("qwen3","x",{x:float('nan') for x in QWEN_GEOMETRY.sampled_layers},{"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","sink":1,"local":1})):
        with pytest.raises(ValueError): require_routing_profile(QWEN_GEOMETRY,broken)

@pytest.mark.parametrize("policy", [
    {"context_behavior":"absolute_tokens","sink":1,"local":1},
    {"schema":"unknown","context_behavior":"absolute_tokens","sink":1,"local":1},
    {"schema":"cascadekv-reserve-v1","sink":1,"local":1},
    {"schema":"cascadekv-reserve-v1","context_behavior":"unknown","sink":1,"local":1},
    {"schema":"cascadekv-reserve-v1","context_behavior":"fraction_of_context","sink":.1,"local":.1},
    {"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","local":1},
    {"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","sink":1},
    {"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","sink":-1,"local":1},
    {"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","sink":float("nan"),"local":1},
    {"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","sink":float("inf"),"local":1},
])
def test_routing_profile_policy_schema_rejects_undefined_inputs(policy):
    good = {x: .1 for x in QWEN_GEOMETRY.sampled_layers}
    with pytest.raises(ValueError):
        require_routing_profile(QWEN_GEOMETRY, RoutingProfile("qwen3", "synthetic", good, policy))

@pytest.mark.parametrize("lambdas", [
    {0:.1, 7:.1, 14:.1, 21:.1},
    {**{x:.1 for x in QWEN_GEOMETRY.sampled_layers}, 26:.1},
    {x:float("inf") for x in QWEN_GEOMETRY.sampled_layers},
    {x:-.1 for x in QWEN_GEOMETRY.sampled_layers},
])
def test_routing_profile_lambda_coverage_and_finiteness(lambdas):
    policy={"schema":"cascadekv-reserve-v1","context_behavior":"absolute_tokens","sink":1,"local":1}
    with pytest.raises(ValueError):
        require_routing_profile(QWEN_GEOMETRY, RoutingProfile("qwen3", "synthetic", lambdas, policy))

def test_provenance_schema():
    p = CacheProvenance("synthetic", "r", "phi3", "0"*64, "r", PHI35_GEOMETRY, 8192, {"kind":"synthetic","id":"fixture"}, {"method":"synthetic-only"}, 0, (1,32,8192,96), (1,32,8192,96), (1,32,8192,96), "float16", 1/math.sqrt(96), "phi3-post-rope-qkv-v1")
    p.validate(); assert len(p.canonical_digest()) == 64
    for value in (float("nan"),float("inf"),-float("inf"),0.):
        with pytest.raises(ValueError): CacheProvenance("x","r","phi3","x","r",PHI35_GEOMETRY,8192,{}, {},0,(1,32,8192,96),(1,32,8192,96),(1,32,8192,96),"f",value,"x").validate()

def _synthetic_phi_provenance():
    return CacheProvenance("synthetic", "revision", "phi3", "0" * 64, "tokenizer-revision", PHI35_GEOMETRY, 8192,
        {"kind":"synthetic","id":"fixture"}, {"method":"synthetic-only"}, 0,
        (1,32,8192,96), (1,32,8192,96), (1,32,8192,96), "float16", 1/math.sqrt(96), "phi3-post-rope-qkv-v1")

@pytest.mark.parametrize("changes", [
    {"config_sha256":"bad"}, {"family":"qwen3"}, {"revision":""}, {"tokenizer_revision":""},
    {"context":1}, {"layer":32}, {"q_shape":(1,31,8192,96)}, {"k_shape":(1,31,8192,96)},
    {"v_shape":(1,31,8192,96)}, {"dtype":"int8"}, {"attention_scaling":float("nan")},
    {"attention_scaling":float("inf")}, {"attention_scaling":float("-inf")}, {"attention_scaling":0.},
    {"attention_scaling":-1.}, {"capture_adapter":""}, {"source_identity":"synthetic"}, {"source_proof":"synthetic"},
])
def test_provenance_validator_rejects_complete_required_matrix(changes):
    with pytest.raises(ValueError):
        replace(_synthetic_phi_provenance(), **changes).validate()

def test_provenance_digest_is_mapping_order_invariant():
    first=_synthetic_phi_provenance()
    second=replace(first, source_identity={"id":"fixture","kind":"synthetic"}, source_proof={"method":"synthetic-only"})
    first.validate(); second.validate()
    assert first.canonical_digest() == second.canonical_digest()

def test_phi3_synthetic_post_rope_boundary():
    transformers = pytest.importorskip("transformers")
    from transformers import Phi3Config, Phi3Model
    torch.manual_seed(4)
    # Four heads over hidden_size 384 gives the actual Phase-A head dimension.
    c = Phi3Config(vocab_size=47, pad_token_id=0, eos_token_id=1, hidden_size=384, intermediate_size=768, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32, original_max_position_embeddings=32)
    m = Phi3Model(c).eval(); ids = torch.tensor([[1,2,3,4,5]])
    got = capture_target_layer_post_rope_qkv_low_memory(m, ids, 1, adapter=Phi3CaptureAdapter())
    # Independent direct reconstruction from the actual target module/hook input.
    seen = {}; attn = m.layers[1].self_attn
    def h(mod,args,kwargs): seen.update(hidden=kwargs["hidden_states"], pos=kwargs["position_embeddings"])
    x=attn.register_forward_pre_hook(h, with_kwargs=True)
    with torch.inference_mode(): m(input_ids=ids)
    x.remove()
    with torch.inference_mode():
        packed=attn.qkv_proj(seen["hidden"]); s=(*seen["hidden"].shape[:-1],-1,attn.head_dim); qw=c.num_attention_heads*attn.head_dim; kw=attn.num_key_value_heads*attn.head_dim
        from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb
        q=packed[...,:qw].view(s).transpose(1,2); k=packed[...,qw:qw+kw].view(s).transpose(1,2); v=packed[...,qw+kw:].view(s).transpose(1,2); q,k=apply_rotary_pos_emb(q,k,*seen["pos"])
    for actual, expected in zip(got,(q,k,v)): assert torch.allclose(actual, expected.float().cpu(), atol=1e-6)
    assert got[0].shape == (1,4,5,96) and got[1].shape == got[2].shape == (1,4,5,96) and attn.scaling == pytest.approx(1/math.sqrt(96))

def test_storage_dtype_contract_round_trip():
    q,k,v=(torch.randn(1,2,3,96) for _ in range(3))
    stored=serialize_qkv_for_cache(q,k,v)
    assert all(x.dtype == torch.float16 and x.is_contiguous() for x in stored)
    assert torch.allclose(stored[0].float(),q,atol=1e-3,rtol=1e-3)

def test_shared_evaluation_core_qwen_legacy_regression():
    from experiments.fresh_sequence_benchmark import attention_reference as legacy_reference, output_metrics as legacy_metrics, quantized_flat as legacy_flat, traffic as legacy_traffic
    torch.manual_seed(17); q=torch.randn(128); k=torch.randn(32,128); v=torch.randn(32,128); budget=8
    core_ids=select_flat_k4(q,k,budget); legacy_ids=legacy_flat(q,k,budget)
    assert core_ids == legacy_ids
    core_ref=exact_attention_reference(q,k,v); legacy_ref=legacy_reference(q,k,v,1/math.sqrt(128))
    assert all(torch.equal(a,b) for a,b in zip(core_ref,legacy_ref))
    assert exact_rerank_and_output(q,k,v,core_ids,reference=core_ref) == legacy_metrics(q,k,v,legacy_ids,legacy_ref,1/math.sqrt(128))
    route={"active_root_reads":1,"detail_reads":2,"expanded_internal_nodes":2,"variance_scalar_reads":8}
    assert traffic(route,32,budget,128,variance=True) == legacy_traffic(route,32,budget,variance=True)

def test_actual_lifting_routing_and_exact_rerank_for_gqa_and_mha():
    torch.manual_seed(19); keys=torch.randn(16,96); values=torch.randn(16,96); query=keys[7]+.01
    forest=StreamingLiftingForest(mode="binary_counter",atom_size=8,window=4,group_counts=(4,))
    forest.append_atom(keys[:8]);forest.append_atom(keys[8:])
    for geometry,head in ((ModelGeometry("qwen3",128,1,2,1,64,16,(0,),(15,)),1),(ModelGeometry("phi3",192,1,2,2,96,16,(0,),(15,)),1)):
        assert geometry.kv_head_for_query(head) == (0 if geometry.model_family == "qwen3" else 1)
        state=initialize_route(forest,query,4,.2,reserve_ids(16,1,1,8)); route=advance_until_budget(state,8)
        assert len(route["ids"])==8 and route["active_root_reads"] and route["detail_reads"] >= 0
        reference=exact_attention_reference(query,keys,values)
        metrics=exact_rerank_and_output(query,keys,values,route["ids"],reference=reference)
        assert set(metrics)=={"top8_recall","relative_exact_attention_mass","relative_l2_error","absolute_l2_error","cosine_similarity"}

def test_logical_8k_phi_evaluation_core_iteration_and_traffic():
    # Compact backing is intentional: the logical geometry remains 8K while
    # dispatch/rerank execute for every real scheduled physical query.
    torch.manual_seed(22); compact_k=torch.randn(16,96); compact_v=torch.randn(16,96)
    forest=StreamingLiftingForest(mode="binary_counter",atom_size=8,window=4,group_counts=(4,)); forest.append_atom(compact_k[:8]); forest.append_atom(compact_k[8:])
    executed=[]
    for layer in PHI35_GEOMETRY.sampled_layers:
        for pos in PHI35_GEOMETRY.sampled_positions:
            for qh in range(PHI35_GEOMETRY.num_q_heads):
                q=compact_k[(layer+pos+qh)%16]
                route=hierarchy_route(forest,q,8,4,.1,reserve_ids(16,1,1,8))
                exact_rerank_and_output(q,compact_k,compact_v,route["ids"])
                executed.append((layer,pos,qh,PHI35_GEOMETRY.kv_head_for_query(qh)))
    assert len(executed)==96*5 and len({(l,h) for l,_,h,_ in executed})==160
    assert all(qh==kv for _,_,qh,kv in executed)
    t=traffic(route,16,8,96,variance=True)
    assert traffic_accounting(96).k4_groups==6 and t["candidate_k4_bytes"]==8*60 and t["selected_v_fp16_bytes"]==8*192
    assert t["total_kv_bytes"] == t["total_k_bytes"] + t["selected_v_fp16_bytes"]
    assert t["total_kv_vs_dense"] == t["total_kv_bytes"] / (16 * 96 * 4)

@pytest.mark.parametrize("head_dim", [96, 128])
def test_total_kv_traffic_is_mechanical(head_dim):
    route={"active_root_reads":1,"detail_reads":2,"expanded_internal_nodes":2,"variance_scalar_reads":8}
    got=traffic(route,32,8,head_dim,variance=True)
    spec=traffic_accounting(head_dim)
    assert got["total_kv_bytes"] == got["total_k_bytes"] + got["selected_v_fp16_bytes"]
    assert got["total_kv_vs_dense"] == got["total_kv_bytes"] / (32 * (spec.fp16_key_bytes + spec.fp16_value_bytes))

def test_qwen_synthetic_adapter_matches_old_formula():
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3Model
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    torch.manual_seed(5); c=Qwen3Config(vocab_size=47, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32)
    m=Qwen3Model(c).eval(); ids=torch.tensor([[1,2,3,4]])
    got=capture_target_layer_post_rope_qkv_low_memory(m,ids,0,adapter=Qwen3CaptureAdapter()); a=m.layers[0].self_attn
    hidden=m.layers[0].input_layernorm(m.embed_tokens(ids)); cos,sin=m.rotary_emb(hidden,torch.arange(4).unsqueeze(0)); s=(*hidden.shape[:-1],-1,a.head_dim)
    q=a.q_norm(a.q_proj(hidden).view(s)).transpose(1,2); k=a.k_norm(a.k_proj(hidden).view(s)).transpose(1,2); q,k=apply_rotary_pos_emb(q,k,cos,sin); v=a.v_proj(hidden).view(s).transpose(1,2)
    for actual, old in zip(got,(q,k,v)): assert torch.allclose(actual,old.float().cpu(),atol=1e-6)
    # The closed Qwen capture implementation is the compatibility oracle.
    from experiments.qwen_partial_dot import capture_target_layer_post_rope_qkv_low_memory as legacy_capture
    legacy = legacy_capture(m, ids, 0)
    for actual, expected in zip(got, legacy): assert torch.allclose(actual, expected, atol=1e-6)

def test_small_attention_and_mapping_regression():
    # Exact attention and selected-V computation have identical semantics for GQA/MHA after mapping.
    q=torch.tensor([[[[1.,0.]], [[0.,1.]]]]); k=torch.tensor([[[[1.,0.],[0.,1.]], [[0.,1.],[1.,0.]]]]); v=k.clone()
    scores=torch.einsum("bhqd,bhkd->bhqk",q,k)/math.sqrt(2); chosen=scores.topk(1,-1).indices
    out=torch.gather(v,2,chosen.squeeze(-1).unsqueeze(-1).expand(-1,-1,-1,2))
    assert out.shape == (1,2,1,2) and torch.equal(out[0,0,0],v[0,0,0])
    # Flat exact top-k and hierarchy's exact branch-and-bound agree with brute force.
    from cascadekv.hierarchical_index import HierarchicalIndex
    from cascadekv.scorer import full_dot_scores, relative_attention_mass_recall
    keys = torch.tensor([[1.,0.,0.,0.], [0.,2.,0.,0.], [0.,0.,3.,0.], [0.,0.,0.,4.]])
    query = torch.tensor([.1, .2, 1., .3])
    reference = full_dot_scores(query, keys)
    flat = reference.topk(2).indices
    hierarchical = HierarchicalIndex(keys, fanout=2).best_first_search(query, k=2, summary="full_box_bound")
    assert torch.equal(hierarchical.ids, flat)
    assert relative_attention_mass_recall(reference, reference, 2) == pytest.approx(1.0)
