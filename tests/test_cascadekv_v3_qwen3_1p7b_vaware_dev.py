import hashlib, json, sys, types
from pathlib import Path
import pytest
import torch
from experiments import cascadekv_v3_qwen3_1p7b_vaware_dev as v

def _traffic(**overrides):
    """A real-traffic-shaped synthetic record, never a reduced test schema."""
    x={k:1.0 for k in v.TRAFFIC_FIELDS}
    x.update(overrides)
    return x

def test_manifest_freeze_identity_and_target_bindings():
    d=v.load_manifest()
    assert hashlib.sha256(v.MANIFEST.read_bytes()).hexdigest()==v.MANIFEST_SHA
    assert d["target_model"]["resolved_commit_sha"]==v.MODEL_SHA
    assert d["target_model"]["config_sha256"]==v.CONFIG_SHA
    used={v.identity(x) for x in d["complete_consumed_identity_inventory"]}
    assert len(d["sources"])==6 and all(v.identity(x) not in used for x in d["sources"].values())
    assert {x["role"] for x in d["sources"].values()}=={"calibration","validation"}
    assert all(x["token_length_proof"]["at_least"] for x in d["sources"].values())

def test_all_live_and_archived_provenance_inputs_are_pinned():
    assert hashlib.sha256(v.V1.read_bytes()).hexdigest()==v.V1_SHA
    assert hashlib.sha256(v.V2.read_bytes()).hexdigest()==v.V2_SHA
    assert hashlib.sha256(v.V3.read_bytes()).hexdigest()==v.V3_SHA
    assert hashlib.sha256(v.ARCHIVED.read_bytes()).hexdigest()==v.ARCHIVED_SHA
    assert hashlib.sha256(v.ARCHIVED_MANIFEST.read_bytes()).hexdigest()==v.ARCHIVED_MANIFEST_SHA

def test_wrong_v1_fails_preflight_before_any_execution(tmp_path, monkeypatch):
    wrong=tmp_path/'wrong-v1.json'; wrong.write_text('{}')
    monkeypatch.setattr(v, 'V1', wrong)
    with pytest.raises(RuntimeError, match='immutable input hash differs'):
        v.require_inputs()

def test_real_traffic_and_wrapper_schema_match_validator_exactly():
    hierarchy=v.traffic({'active_root_reads':2,'detail_reads':3,'expanded_internal_nodes':4,
                         'variance_scalar_reads':5}, 64, 8, variance=True)
    flat=v.traffic({'active_root_reads':0,'detail_reads':0,'expanded_internal_nodes':0,
                    'variance_scalar_reads':0}, 64, 8, variance=False, flat=True)
    produced=set(hierarchy)
    assert produced==set(flat)
    assert set(v.tr(hierarchy,64))==set(v.TRAFFIC_FIELDS)
    assert set(v.tr(flat,64))==set(v.TRAFFIC_FIELDS)
    # All production method families flow through tr(), so their resulting
    # traffic records are precisely the validator's accounting schema.
    assert {'total_kv_bytes','total_kv_vs_dense'} <= set(v.tr(hierarchy,64))

def test_architecture_actions_schedule_and_no_reselection():
    d=v.load_manifest(); assert d["geometry"]["static_schedule_cells"]==40
    assert d["immutable_architecture"]["actions"]=={"A0":[.05,"hierarchy"],"A1":[.075,"hierarchy"],"A2":[.1,"hierarchy"],"A3":[.15,"hierarchy"],"A4":[.05,"flat"]}
    assert v.load_manifest()==d
    source=Path(v.__file__).read_text()
    assert "AutoModel.from_pretrained(MODEL,revision=MODEL_SHA" in source
    assert "Qwen/Qwen3-0.6B" not in source
    assert hashlib.sha256(v.V3.read_bytes()).hexdigest()==v.V3_SHA
    assert "weak_layer" not in source.lower() and "weak_head" not in source.lower()

def test_inherited_target_semantics_and_calibration_only_dp():
    assert v.inherited_targets(10,20)=={"T0":10,"T1":15.,"T2":20}
    groups=[{a:{"total_kv_bytes":10+i,"relative_l2":5-i,"cosine":i} for i,a in enumerate(v.ACTIONS)} for _ in range(40)]
    out=v.optimize_calibration(groups,14)
    assert out and len(out[3])==40 and set(out[3])<=set(v.ACTIONS)
    assert "validation" not in v.optimize_calibration.__code__.co_varnames
    assert "ARCHIVED" not in v.optimize_calibration.__code__.co_names

def test_gates_boundaries_and_strict_comparative_improvements():
    base={"cosine":.985,"relative_l2":.120,"kv":100}
    assert v.passes(base,100,.121,.121)
    assert not v.passes(base,99,.121,.121)
    assert not v.passes(base,100,.120,.121) # equal v2 is not improvement
    assert not v.passes(base,100,.121,.120) # equal zero-shot v3 is not improvement

def candidates(t0, t1, t2):
    return [
        {"name":"target_vaware_T0", **t0},
        {"name":"target_vaware_T1", **t1},
        {"name":"target_vaware_T2", **t2},
    ]

def test_winner_is_first_passing_target_not_best_realized_metric():
    # A: T0 wins even though later candidates realize better validation values.
    xs=candidates(
        {"cosine":.985,"relative_l2":.119,"kv":100},
        {"cosine":.999,"relative_l2":.01,"kv":1},
        {"cosine":.999,"relative_l2":.02,"kv":2},
    )
    assert v.choose_winner(xs,100,.121,.121)["name"]=="target_vaware_T0"
    # B: T1 wins over a better-realized T2 after T0 fails.
    xs=candidates(
        {"cosine":.984,"relative_l2":.01,"kv":1},
        {"cosine":.985,"relative_l2":.119,"kv":100},
        {"cosine":.999,"relative_l2":.01,"kv":1},
    )
    assert v.choose_winner(xs,100,.121,.121)["name"]=="target_vaware_T1"
    # C: T2 wins when T0 and T1 fail.
    xs=candidates(
        {"cosine":.984,"relative_l2":.01,"kv":1},
        {"cosine":.984,"relative_l2":.01,"kv":1},
        {"cosine":.985,"relative_l2":.119,"kv":100},
    )
    assert v.choose_winner(xs,100,.121,.121)["name"]=="target_vaware_T2"
    # D: no passing target means no winner.
    assert v.choose_winner(candidates(*([{"cosine":.984,"relative_l2":.01,"kv":1}]*3)),100,.121,.121) is None

def test_winner_order_is_independent_of_input_order():
    xs=candidates(
        {"cosine":.985,"relative_l2":.119,"kv":100},
        {"cosine":.999,"relative_l2":.01,"kv":1},
        {"cosine":.999,"relative_l2":.02,"kv":2},
    )
    assert v.choose_winner(list(reversed(xs)),100,.121,.121)["name"]=="target_vaware_T0"

@pytest.mark.parametrize("invalid",[
    [],
    [
        {"name":"target_vaware_T0","cosine":.99,"relative_l2":.01,"kv":1},
        {"name":"target_vaware_T1","cosine":.99,"relative_l2":.01,"kv":1},
    ],
    [{"name":"target_vaware_T0","cosine":.99,"relative_l2":.01,"kv":1}]*3,
    candidates({"cosine":.99,"relative_l2":.01,"kv":1},{"cosine":.99,"relative_l2":.01,"kv":1},{"cosine":.99,"relative_l2":.01,"kv":1})+[{"name":"other","cosine":.99,"relative_l2":.01,"kv":1}],
])
def test_invalid_target_candidate_sets_fail_closed(invalid):
    assert v.choose_winner(invalid,100,.121,.121) is None

def test_status_is_data_free_and_empty_artifacts_are_zero(tmp_path):
    s=v.status(tmp_path/"cache",tmp_path/"shards")
    assert (s["valid_caches"],s["total_caches"],s["valid_action_shards"],s["total_action_shards"])==(0,30,0,30)
    assert "AutoModel" not in v.status.__code__.co_names

def test_action_shard_requires_all_physical_action_observations(tmp_path):
    d=v.load_manifest(); source="narrative_calibration"; layer=0
    rows=[]
    for pos in v.POSITIONS:
      for head in range(16):
       for action in v.ACTIONS:
            rows.append({"source":source,"role":"calibration","layer":layer,"context_length":4096,"position":pos,"q_head":head,"kv_head":head//2,"action":action,"metrics":{x:0.0 for x in v.METRICS},"traffic":_traffic()})
    path=v.shard_path(tmp_path,source,layer)
    baseline_rows=[]
    for pos in v.POSITIONS:
      for head in range(16):
       for method in v.BASELINES:
        baseline_rows.append({"source":source,"role":"calibration","layer":layer,"context_length":4096,"position":pos,"q_head":head,"kv_head":head//2,"method":method,"metrics":{x:0.0 for x in v.METRICS},"traffic":_traffic()})
    meta={"manifest_sha256":v.MANIFEST_SHA,"protocol_tag":v.TAG,"protocol_commit":v.TAG_COMMIT,"model_name":v.MODEL,"model_sha":v.MODEL_SHA,"model_revision":v.MODEL_SHA,"target_config_sha256":v.CONFIG_SHA,"source_identity":d["sources"][source],"role":"calibration","layer":layer,"context_length":4096}
    path.write_text(json.dumps({**meta,"rows":rows,"baseline_rows":baseline_rows}))
    assert v.validate_shard(path,source,layer)[0]
    rows.pop(); path.write_text(json.dumps({"manifest_sha256":v.MANIFEST_SHA,"model_sha":v.MODEL_SHA,"source_identity":d["sources"][source],"role":"calibration","layer":layer,"context_length":4096,"rows":rows}))
    assert not v.validate_shard(path,source,layer)[0]

def test_original_v3_action_profile_semantics_are_not_budget_semantics(monkeypatch):
    """Regression against original v3: A2 is 10%-budget with sch5/lam5."""
    budgets=[]; profiles=[]
    monkeypatch.setattr(v,'candidate_budget',lambda n,f:budgets.append(f) or 7)
    monkeypatch.setattr(v,'frozen_routing_config',lambda cfg,l,f:profiles.append(f) or ({'sink':0,'local':0},1.0))
    monkeypatch.setattr(v,'route',lambda *args:{'ids':set(),'active_root_reads':0,'detail_reads':0,'expanded_internal_nodes':0,'variance_scalar_reads':0})
    monkeypatch.setattr(v,'traffic',lambda *args,**kwargs:{'total_k_bytes':1.,'selected_v_fp16_bytes':1.})
    monkeypatch.setattr(v,'quantized_flat',lambda *args:[])
    for action in v.ACTIONS:
        v.evaluate_frozen_action(action,None,None,None,100,0,{})
    assert budgets==[.05,.075,.10,.15,.05]
    assert profiles==[.05,.05,.05,.05]
    budgets.clear();profiles.clear()
    v.evaluate_uniform_v1(.05,None,None,None,100,0,{})
    v.evaluate_uniform_v1(.10,None,None,None,100,0,{})
    assert budgets==[.05,.10] and profiles==[.05,.10]
    # A frozen table cell resolving to A2 has the same action semantics.
    budgets.clear();profiles.clear();v.evaluate_frozen_action('A2',None,None,None,100,0,{})
    assert budgets==[.10] and profiles==[.05]

def _metadata(source, layer):
    src=v.load_manifest()['sources'][source]
    return {'manifest_sha256':v.MANIFEST_SHA,'protocol_tag':v.TAG,'protocol_commit':v.TAG_COMMIT,
            'model_name':v.MODEL,'model_sha':v.MODEL_SHA,'model_revision':v.MODEL_SHA,
            'target_config_sha256':v.CONFIG_SHA,'source_identity':src,'role':src['role'],
            'layer':layer,'context_length':4096}

def _synthetic_artifacts(root, validation_quality='normal'):
    """Write complete schema-valid artifacts without weights or real Q/K/V data."""
    cache,shards=root/'cache',root/'shards';cache.mkdir(parents=True);shards.mkdir()
    for source,layer in __import__('itertools').product(v.CAL+v.VAL,v.LAYERS):
        md=_metadata(source,layer)
        # Meta tensors preserve the precise storage geometry without allocating
        # the 30 real captures; validate_cache intentionally accepts their type.
        torch = __import__('torch')
        torch.save({'metadata':{**md,'tensor_shapes':{'q':[1,16,4096,128],'k':[1,8,4096,128],'v':[1,8,4096,128]}},
                    'query':torch.empty((1,16,4096,128),dtype=torch.float16,device='meta'),
                    'key':torch.empty((1,8,4096,128),dtype=torch.float16,device='meta'),
                    'value':torch.empty((1,8,4096,128),dtype=torch.float16,device='meta')},v.cache_path(cache,source,layer))
        rows=[];baselines=[]
        for position in v.POSITIONS:
          for head in range(16):
            common={'source':source,'role':md['role'],'layer':layer,'context_length':4096,'position':position,'q_head':head,'kv_head':head//2}
            for i,action in enumerate(v.ACTIONS):
                # Calibration makes T0/A0, T1/A2, T2/A4; validation then proves
                # fixed T0-first selection even where later targets are better.
                rel=(.119,.05,.01,.02,.005)[i] if (md['role']=='calibration' or validation_quality=='normal') else (.13,.05,.01,.02,.005)[i]
                rows.append({**common,'action':action,'metrics':{'relative_exact_attention_mass':.8,'top8_recall':.8,'cosine_similarity':.986+i*.002,'relative_l2_error':rel,'absolute_l2_error':rel},'traffic':_traffic(total_kv_bytes=50+i*10,total_k_bytes=20,selected_v_fp16_bytes=30)})
            for method in v.BASELINES:
                rel=.121 if method in ('frozen_cascadekv_v2','original_frozen_cascadekv_v3_zero_shot') else .2
                kv=50 if method=='frozen_cascadekv_v2' else 100
                baselines.append({**common,'method':method,'metrics':{'relative_exact_attention_mass':.8,'top8_recall':.8,'cosine_similarity':.99,'relative_l2_error':rel,'absolute_l2_error':rel},'traffic':_traffic(total_kv_bytes=kv,total_k_bytes=20,selected_v_fp16_bytes=30)})
        v.atomic_json({**md,'rows':rows,'baseline_rows':baselines},v.shard_path(shards,source,layer))
    return cache,shards

def test_complete_synthetic_merge_uses_single_fixed_t0_first_winner(tmp_path):
    cache,shards=_synthetic_artifacts(tmp_path)
    out=tmp_path/'merged.json';v.merge(cache,shards,out);got=json.loads(out.read_text())
    assert got['status']=='complete'
    assert got['classification']=='1P7B-V-AWARE-STATIC-PROMISING'
    assert got['winner_target_name']=='target_vaware_T0'
    assert len(got['validation_candidates'])==3
    assert got['development_success_rule']
    assert got['development_gates']['min_mean_cosine']==.985
    assert set(got['winner_validation_deltas'])=={'original_frozen_cascadekv_v3_zero_shot','frozen_cascadekv_v2','uniform_v1_10','flat_q8k4_5'}

def test_validation_cannot_change_calibration_targets_or_40_cell_tables(tmp_path):
    cache_a,shards_a=_synthetic_artifacts(tmp_path/'a','normal')
    cache_b,shards_b=_synthetic_artifacts(tmp_path/'b','poor')
    out_a,out_b=tmp_path/'a.json',tmp_path/'b.json'
    v.merge(cache_a,shards_a,out_a);v.merge(cache_b,shards_b,out_b)
    a,b=json.loads(out_a.read_text()),json.loads(out_b.read_text())
    assert a['winner_target_name']=='target_vaware_T0'
    assert b['winner_target_name']=='target_vaware_T1'
    assert a['traffic_targets']==b['traffic_targets']
    for target in ('T0','T1','T2'):
        assert a['calibration_frozen_tables'][target]['table']==b['calibration_frozen_tables'][target]['table']
        assert len(a['calibration_frozen_tables'][target]['table'])==40

def test_archived_zero_shot_result_is_provenance_only_not_candidate_input(tmp_path, monkeypatch):
    # The archived result is intentionally not opened by candidate fitting;
    # this controlled replacement would be rejected by production preflight.
    poisoned=tmp_path/'archived.json'; poisoned.write_text('{"metrics":{"relative_l2_error":999999}}')
    monkeypatch.setattr(v,'ARCHIVED',poisoned)
    ca=[];cb=[]
    for layer,kv in __import__('itertools').product(v.LAYERS,range(8)):
        for action in v.ACTIONS:
            ca.append({'layer':layer,'kv_head':kv,'action':action,
                       'metrics':{'relative_l2_error':.1,'cosine_similarity':.99},
                       'traffic':{'total_kv_bytes':10.}})
        for method,bytes_ in (('frozen_cascadekv_v2',10.),('uniform_v1_10',20.)):
            cb.append({'method':method,'traffic':{'total_kv_bytes':bytes_}})
    assert v.construct_candidates(ca,cb)==v.construct_candidates(ca,cb)
    assert 'ARCHIVED' not in v.construct_candidates.__code__.co_names

def test_partial_merge_can_never_claim_completion_or_promising(tmp_path):
    cache,shards=_synthetic_artifacts(tmp_path)
    v.shard_path(shards,v.VAL[-1],v.LAYERS[-1]).unlink()
    out=tmp_path/'partial.json';v.merge(cache,shards,out);got=json.loads(out.read_text())
    assert got['status']=='partial' and got['classification'] is None

def test_complete_synthetic_merge_no_passing_candidate_has_no_winner(tmp_path):
    cache,shards=_synthetic_artifacts(tmp_path)
    # Preserve calibration entirely; make every validation candidate fail the
    # frozen cosine gate through complete, schema-valid validation shards.
    for source,layer in __import__('itertools').product(v.VAL,v.LAYERS):
        p=v.shard_path(shards,source,layer);d=json.loads(p.read_text())
        for row in d['rows']: row['metrics']['cosine_similarity']=.1
        v.atomic_json(d,p)
    out=tmp_path/'none.json';v.merge(cache,shards,out);got=json.loads(out.read_text())
    assert got['classification']=='1P7B-V-AWARE-STATIC-NOT-PROMISING'
    assert got['winner_target_name'] is None and got['winner_table'] is None

def test_capture_wrapper_binds_the_frozen_1p7b_model_and_writes_valid_cache(tmp_path,monkeypatch):
    calls=[]
    class Tokenizer:
        def __call__(self,*args,**kwargs): return types.SimpleNamespace(input_ids=torch.zeros((1,4096),dtype=torch.long))
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args,**kwargs): calls.append(('tokenizer',args,kwargs));return Tokenizer()
    class Model:
        layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(scaling=1.0))]
        def eval(self): return self
    class AutoModel:
        @staticmethod
        def from_pretrained(*args,**kwargs): calls.append(('model',args,kwargs));return Model()
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoModel=AutoModel,AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(v,'resolve',lambda *args:'synthetic')
    monkeypatch.setattr(v,'capture_target_layer_post_rope_qkv_low_memory',lambda *args:(torch.empty((1,16,4096,128),dtype=torch.float16,device='meta'),torch.empty((1,8,4096,128),dtype=torch.float16,device='meta'),torch.empty((1,8,4096,128),dtype=torch.float16,device='meta')))
    v.capture('narrative_calibration',0,tmp_path)
    assert all(args[0]==v.MODEL and kw['revision']==v.MODEL_SHA for _,args,kw in calls)
    assert v.validate_cache(v.cache_path(tmp_path,'narrative_calibration',0),'narrative_calibration',0)[0]
    # Failure before promotion cannot turn a pre-existing corrupt file valid.
    p=v.cache_path(tmp_path,'narrative_calibration',0);p.write_bytes(b'corrupt')
    monkeypatch.setattr(v,'capture_target_layer_post_rope_qkv_low_memory',lambda *a: (_ for _ in ()).throw(RuntimeError('forced')))
    with pytest.raises(RuntimeError,match='forced'): v.capture('narrative_calibration',0,tmp_path)
    assert not v.validate_cache(p,'narrative_calibration',0)[0]

def test_real_evaluate_composition_emits_complete_240_plus_336_schema(tmp_path,monkeypatch):
    source,layer='narrative_calibration',0;cache=tmp_path/'cache';cache.mkdir();p=v.cache_path(cache,source,layer)
    md={**_metadata(source,layer),'tensor_shapes':{'q':[1,16,4096,128],'k':[1,8,4096,128],'v':[1,8,4096,128]},'attention_scaling':1.0}
    torch.save({'metadata':md,'query':torch.empty((1,16,4096,128),dtype=torch.float16,device='meta'),'key':torch.empty((1,8,4096,128),dtype=torch.float16,device='meta'),'value':torch.empty((1,8,4096,128),dtype=torch.float16,device='meta')},p)
    real_load=torch.load
    payload={'metadata':md,'query':torch.zeros((1,16,4096,128),dtype=torch.float16),'key':torch.zeros((1,8,4096,128),dtype=torch.float16),'value':torch.zeros((1,8,4096,128),dtype=torch.float16)}
    monkeypatch.setattr(v.torch,'load',lambda *args,**kwargs: payload)
    class Forest:
        def __init__(self,**kwargs): pass
        def append_atom(self,x): pass
    monkeypatch.setattr(v,'StreamingLiftingForest',Forest)
    monkeypatch.setattr(v,'attention_reference',lambda *args:(None,None,None))
    monkeypatch.setattr(v,'output_metrics',lambda *args:{m:.5 for m in v.METRICS})
    profiles=[]
    monkeypatch.setattr(v,'candidate_budget',lambda n,f:max(1,int(n*f)))
    monkeypatch.setattr(v,'frozen_routing_config',lambda cfg,l,f:profiles.append(f) or ({'sink':0,'local':0},1.0))
    monkeypatch.setattr(v,'route',lambda *args:{'ids':{0},'active_root_reads':0,'detail_reads':0,'expanded_internal_nodes':0,'variance_scalar_reads':0})
    monkeypatch.setattr(v,'traffic',lambda *args,**kwargs:{'total_k_bytes':1.,'selected_v_fp16_bytes':1.})
    monkeypatch.setattr(v,'quantized_flat',lambda *args:[0])
    v.evaluate(source,layer,cache,tmp_path/'shards')
    d=json.loads(v.shard_path(tmp_path/'shards',source,layer).read_text())
    assert len(d['rows'])==240 and len(d['baseline_rows'])==336
    assert {r['action'] for r in d['rows']}==set(v.ACTIONS)
    assert {r['method'] for r in d['baseline_rows']}==set(v.BASELINES)
    assert .10 in profiles and all(x in (.05,.10) for x in profiles)
    assert v.validate_shard(v.shard_path(tmp_path/'shards',source,layer),source,layer)[0]
    # Evaluation failure happens before atomic promotion, so an old invalid
    # shard never becomes a centrally valid final artifact by accident.
    final=v.shard_path(tmp_path/'shards',source,layer);final.write_text('{}')
    monkeypatch.setattr(v,'output_metrics',lambda *a: (_ for _ in ()).throw(RuntimeError('forced')))
    with pytest.raises(RuntimeError,match='forced'): v.evaluate(source,layer,cache,tmp_path/'shards')
    assert not v.validate_shard(final,source,layer)[0]

def test_resume_wrappers_skip_valid_artifacts_and_regenerate_corrupt_ones(tmp_path, monkeypatch):
    """Exercise wrapper decisions, without loading a real model or dataset."""
    source,layer='narrative_calibration',0
    cache,shards=_synthetic_artifacts(tmp_path)
    valid_cache=v.cache_path(cache,source,layer)
    # A valid cache takes the return path before the expensive imports/capture.
    monkeypatch.setattr(v,'capture_target_layer_post_rope_qkv_low_memory',lambda *a: pytest.fail('capture called'))
    v.capture(source,layer,cache)
    # A valid shard likewise takes the return path before routing/evaluation.
    monkeypatch.setattr(v,'attention_reference',lambda *a: pytest.fail('evaluation called'))
    v.evaluate(source,layer,cache,shards)
    # Corruption is not accepted as resumable state.
    valid_cache.write_bytes(b'not a torch cache')
    assert not v.validate_cache(valid_cache,source,layer)[0]
    # The capture wrapper now enters regeneration rather than returning.
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k: pytest.fail('regeneration entered')),
        AutoModel=object))
    with pytest.raises(pytest.fail.Exception): v.capture(source,layer,cache)

def test_fail_closed_validator_matrix_for_cache_and_shard(tmp_path):
    cache,shards=_synthetic_artifacts(tmp_path)
    source,layer='narrative_calibration',0
    cp=v.cache_path(cache,source,layer); sp=v.shard_path(shards,source,layer)
    good=torch.load(cp,map_location='cpu',weights_only=False)
    # Every metadata/tensor binding mutation must invalidate a cache.
    for key,bad in [('manifest_sha256','bad'),('protocol_commit','bad'),('model_sha','bad'),
                    ('target_config_sha256','bad'),('source_identity',{}),('role','validation'),
                    ('layer',7),('context_length',1)]:
        d=dict(good); d['metadata']=dict(good['metadata']); d['metadata'][key]=bad
        torch.save(d,cp); assert not v.validate_cache(cp,source,layer)[0],key
    d=dict(good);d['metadata']=dict(good['metadata']);d['metadata']['tensor_shapes']['q']=[1]
    torch.save(d,cp);assert not v.validate_cache(cp,source,layer)[0]
    d=dict(good);d['query']=good['query'].float();torch.save(d,cp);assert not v.validate_cache(cp,source,layer)[0]
    torch.save(good,cp)
    original=json.loads(sp.read_text())
    def reject(mut):
        d=json.loads(json.dumps(original));mut(d);sp.write_text(json.dumps(d));assert not v.validate_shard(sp,source,layer)[0]
    reject(lambda d:d.update(manifest_sha256='bad'))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'source':'wrong'}))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'role':'validation'}))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'layer':7}))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'context_length':1}))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'kv_head':7}))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'metrics':{**d['rows'][0]['metrics'],'cosine_similarity':float('nan')} }))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'traffic':{**d['rows'][0]['traffic'],'total_k_bytes':float('inf')} }))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'traffic':{k:x for k,x in d['rows'][0]['traffic'].items() if k!='total_k_bytes'}}))
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'traffic':{**d['rows'][0]['traffic'],'unexpected':0}}))
    reject(lambda d:d['rows'].pop())                 # missing action row
    reject(lambda d:d['rows'].append(dict(d['rows'][0]))) # duplicate action row
    reject(lambda d:d['rows'].__setitem__(0,{**d['rows'][0],'action':'unknown'}))
    reject(lambda d:d['baseline_rows'].pop())
    reject(lambda d:d['baseline_rows'].append(dict(d['baseline_rows'][0])))
    reject(lambda d:d['baseline_rows'].__setitem__(0,{**d['baseline_rows'][0],'method':'unknown'}))
