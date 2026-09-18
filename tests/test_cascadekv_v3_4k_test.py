import hashlib, json
from pathlib import Path
import pytest
import torch
from experiments import cascadekv_v3_4k_test as t

BASE_BINDINGS = ("MANIFEST","V2_CONFIG","V2_SHA","TEST_MANIFEST_SHA","METHODS","require_inputs","load_manifest","POSITIONS","validate_cache")

def valid_cache(tmp_path, sequence="narrative", layer=0):
    """Exact-shape cache that reaches both the preserved base and v3 checks."""
    q=torch.zeros((1,16,4096,128),dtype=torch.float16)
    k=torch.zeros((1,8,4096,128),dtype=torch.float16)
    v=torch.zeros((1,8,4096,128),dtype=torch.float16)
    p=t.cache_path(tmp_path,sequence,layer)
    p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"metadata":{"manifest_sha256":t.TEST_MANIFEST_SHA,"v2_config_sha256":t.V2_SHA,
                "v3_config_sha256":t.V3_SHA,"model_sha":t.MODEL_SHA,"model_revision":t.MODEL_SHA,
                "sequence":sequence,"layer":layer,"context_length":4096,
                "source_identity":t.load_manifest()["sources"][sequence],
                "tensor_shapes":{"q":list(q.shape),"k":list(k.shape),"v":list(v.shape)}},
                "query":q,"key":k,"value":v},p)
    return p

def test_frozen_manifest_and_inputs_are_exact():
    t.require_inputs(); assert hashlib.sha256(t.MANIFEST.read_bytes()).hexdigest()==t.TEST_MANIFEST_SHA
    assert t.load_manifest()["git_freeze_tag"]["resolved_pre_test_commit_sha"]==t.TAG_COMMIT
def test_explicit_consumed_identities_exclude_new_sources():
    d=t.load_manifest()
    for x in d["sources"].values(): assert not any(t.overlaps(x,p) for p in d["complete_prior_explicit_identity_inventory"])
def test_existing_manifest_never_reselects(tmp_path,monkeypatch):
    m=tmp_path/'m.json';m.write_bytes(t.MANIFEST.read_bytes());monkeypatch.setattr(t,'MANIFEST',m);monkeypatch.setattr(t,'require_inputs',lambda:None);monkeypatch.setattr(t,'build_manifest',lambda:pytest.fail('reselection'))
    t.preflight()
def test_bad_manifest_stops_before_parse(tmp_path,monkeypatch):
    m=tmp_path/'m.json';m.write_text('{ bad');monkeypatch.setattr(t,'MANIFEST',m);monkeypatch.setattr(t,'require_inputs',lambda:None)
    with pytest.raises(RuntimeError,match='hash differs'): t.load_manifest()
def test_classification_uses_only_predeclared_gates():
    def p(c,r,b): return {"metrics":{"cosine_similarity":{"mean":c},"relative_l2_error":{"mean":r}},"traffic":{"total_kv_bytes":{"mean":b}}}
    x={"frozen_cascadekv_v3":p(.99,.1,1),"uniform_v1_10":p(.9,.2,2),"frozen_cascadekv_v2":p(.9,.11,1)}
    assert t.classify(x)=="TEST-PASSED"; x["frozen_cascadekv_v3"]["tails"]={"worst":999}; assert t.classify(x)=="TEST-PASSED"
    x["frozen_cascadekv_v3"]=p(.985,.12,2); x["uniform_v1_10"]=p(.9,.2,2); x["frozen_cascadekv_v2"]=p(.9,.13,1)
    assert t.classify(x)=="TEST-PASSED"
    x["frozen_cascadekv_v2"]=p(.9,.12,1)
    assert t.classify(x)=="TEST-COMPARATIVE-GATE-FAILED"
def test_shard_cardinality(tmp_path):
    rows=[{"position":z,"q_head":h,"kv_head":h//2,"method":m,"sequence":"n","layer":0,"context_length":4096,"metrics":{k:0 for k in t.METRIC_FIELDS},"traffic":{k:0 for k in t.TRAFFIC_FIELDS}} for z in t.POSITIONS for h in range(16) for m in t.METHODS]
    d={"manifest_sha256":t.TEST_MANIFEST_SHA,"model_sha":t.MODEL_SHA,"v2_config_sha256":t.V2_SHA,"v3_config_sha256":t.V3_SHA,"sequence":"n","layer":0,"context_length":4096,"rows":rows}; p=tmp_path/'s';p.write_text(json.dumps(d));assert t.validate_shard(p,'n',0)[0]
    d['rows'].append(rows[0]);p.write_text(json.dumps(d));assert not t.validate_shard(p,'n',0)[0]

def test_base_evaluation_cache_adapter_is_nonrecursive_under_both_pass_bindings(tmp_path):
    p=valid_cache(tmp_path)
    original={name:getattr(t.base,name) for name in BASE_BINDINGS}
    for config, digest in ((t.V2_CONFIG,t.V2_SHA),(t.V3_CONFIG,t.V3_SHA)):
        with t.bound_base(config,digest,t.BASE_METHODS,validate_cache=t.base_evaluate_cache_adapter) as bound:
            assert bound.validate_cache(p,"narrative",0)==(True,"exact v3 cache binding")
    assert {name:getattr(t.base,name) for name in BASE_BINDINGS}==original

def test_wrapper_capture_uses_v2_provenance_and_adds_independent_v3_binding(tmp_path, monkeypatch):
    """Exercise the real wrapper capture/validation path without model capture."""
    original={name:getattr(t.base,name) for name in BASE_BINDINGS}
    seen=[]
    def fake_base_capture(sequence, layer, root):
        # This is the exact metadata surface of base.capture, read from the
        # globals which the wrapper temporarily binds around that routine.
        seen.append((t.base.MANIFEST,t.base.V2_CONFIG,t.base.V2_SHA,
                     t.base.TEST_MANIFEST_SHA,t.base.METHODS,
                     t.base.require_inputs,t.base.load_manifest))
        manifest=t.base.load_manifest()
        q=torch.zeros((1,16,4096,128),dtype=torch.float16)
        k=torch.zeros((1,8,4096,128),dtype=torch.float16)
        v=torch.zeros((1,8,4096,128),dtype=torch.float16)
        meta={"manifest_sha256":hashlib.sha256(t.base.MANIFEST.read_bytes()).hexdigest(),
              "v2_config_sha256":t.base.V2_SHA,"model_sha":t.base.MODEL_SHA,
              "sequence":sequence,"source_identity":manifest["sources"][sequence],
              "context_length":4096,"layer":layer,
              "tensor_shapes":{"q":list(q.shape),"k":list(k.shape),"v":list(v.shape)},
              "attention_scaling":1.0,"storage":"authoritative FP16 Q/K/V"}
        root.mkdir(parents=True,exist_ok=True)
        torch.save({"metadata":meta,"query":q,"key":k,"value":v},t.base.cache_path(root,sequence,layer))
    monkeypatch.setattr(t.base,"capture",fake_base_capture)
    t.capture("narrative",0,tmp_path)
    p=t.cache_path(tmp_path,"narrative",0)
    assert t.validate_cache(p,"narrative",0)==(True,"exact v3 cache binding")
    x=torch.load(p,map_location="cpu",weights_only=False); m=x["metadata"]
    assert m["manifest_sha256"]==t.TEST_MANIFEST_SHA
    assert m["model_sha"]==m["model_revision"]==t.MODEL_SHA
    assert m["v2_config_sha256"]==t.V2_SHA and m["v3_config_sha256"]==t.V3_SHA
    assert m["source_identity"]==t.load_manifest()["sources"]["narrative"]
    assert (m["sequence"],m["layer"],m["context_length"])==("narrative",0,4096)
    assert [tuple(x[n].shape) for n in ("query","key","value")]==[(1,16,4096,128),(1,8,4096,128),(1,8,4096,128)]
    assert all(x[n].dtype==torch.float16 for n in ("query","key","value"))
    assert [(manifest,config,digest,manifest_sha,methods) for manifest,config,digest,manifest_sha,methods,*_ in seen]==[(t.MANIFEST,t.V2_CONFIG,t.V2_SHA,t.TEST_MANIFEST_SHA,t.BASE_METHODS)]
    # A V3 capture binding would make fake_base_capture write V3_SHA here and
    # fail this preserved-base validation under its fixed V2 binding.
    for config,digest in ((t.V2_CONFIG,t.V2_SHA),(t.V3_CONFIG,t.V3_SHA)):
        with t.bound_base(config,digest,t.BASE_METHODS,validate_cache=t.base_evaluate_cache_adapter) as bound:
            assert bound.validate_cache(p,"narrative",0)==(True,"exact v3 cache binding")
    assert {name:getattr(t.base,name) for name in BASE_BINDINGS}==original

def test_cache_validator_exception_restores_every_base_binding(monkeypatch, tmp_path):
    original={name:getattr(t.base,name) for name in BASE_BINDINGS}
    def fail(*args):
        raise RuntimeError("forced preserved validator failure")
    monkeypatch.setattr(t,"BASE_VALIDATE_CACHE",fail)
    with pytest.raises(RuntimeError,match="forced preserved validator failure"):
        t.validate_cache(tmp_path/"cache","narrative",0)
    assert {name:getattr(t.base,name) for name in BASE_BINDINGS}==original

def test_two_pass_evaluation_remaps_exactly_and_restores_base_globals(tmp_path, monkeypatch):
    seen=[]
    original={name:getattr(t.base,name) for name in BASE_BINDINGS}
    def fake_evaluate(sequence, layer, cache, shards):
        seen.append((t.base.V2_CONFIG,t.base.V2_SHA,t.base.METHODS))
        assert t.base.validate_cache(t.cache_path(cache,sequence,layer),sequence,layer) == (True,"exact v3 cache binding")
        rows=[{"sequence":sequence,"layer":layer,"context_length":4096,"position":z,"q_head":h,"kv_head":h//2,"method":method,
               "metrics":{k:0.0 for k in t.METRIC_FIELDS},"traffic":{k:0.0 for k in t.TRAFFIC_FIELDS}}
              for z in t.POSITIONS for h in range(16) for method in t.base.METHODS]
        shards.mkdir(parents=True,exist_ok=True)
        (shards/f'{sequence}_L4096_layer{layer}.json').write_text(json.dumps({'sequence':sequence,'layer':layer,'context_length':4096,'rows':rows}))
    monkeypatch.setattr(t.base,'evaluate',fake_evaluate)
    monkeypatch.setattr(t.base,'frozen_routing_config',lambda *args:pytest.fail('schedule construction called'))
    valid_cache(tmp_path/'cache')
    t.evaluate('narrative',0,tmp_path/'cache',tmp_path/'shards')
    shard=json.loads((tmp_path/'shards'/'narrative_L4096_layer0.json').read_text())
    methods=[r['method'] for r in shard['rows']]
    assert len(methods)==3*16*7 and set(methods)==set(t.METHODS)
    assert methods.count('frozen_cascadekv_v2')==3*16
    assert methods.count('frozen_cascadekv_v3')==3*16
    assert 'cascadekv_v2' not in methods
    for z in t.POSITIONS:
        for h in range(16):
            assert [r['method'] for r in shard['rows'] if r['position']==z and r['q_head']==h].count('frozen_cascadekv_v2')==1
            assert [r['method'] for r in shard['rows'] if r['position']==z and r['q_head']==h].count('frozen_cascadekv_v3')==1
    assert [(p,sha) for p,sha,_ in seen]==[(t.V2_CONFIG,t.V2_SHA),(t.V3_CONFIG,t.V3_SHA)]
    assert all(methods == t.BASE_METHODS for _,_,methods in seen)
    assert {name:getattr(t.base,name) for name in BASE_BINDINGS}==original
def test_status_is_read_only(tmp_path):
    before=list(tmp_path.iterdir()); t.status(tmp_path/'cache',tmp_path/'shards'); assert list(tmp_path.iterdir())==before
