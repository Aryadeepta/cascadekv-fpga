import hashlib, json, math
from pathlib import Path
import pytest, torch
from experiments import cascadekv_v3_qwen3_1p7b_4k_test as t

def test_frozen_target_architecture_and_literal_table():
    t.require_inputs(); t.verify_architecture(dict(t.REQUIRED_ARCHITECTURE))
    with pytest.raises(RuntimeError): t.verify_architecture({**t.REQUIRED_ARCHITECTURE,"head_dim":64})
    cfg=json.loads(t.V3_CONFIG.read_text()); assert len(cfg["exact_layer_head_action_table"]) == 40
    assert t.MODEL == "Qwen/Qwen3-1.7B" and t.MODEL_SHA == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert t.TAG == "cascadekv-v3-qwen3-1p7b-4k-test-freeze" and t.TAG_COMMIT == "890bdb1cbc3e097e324b480debbcb8ffd7385cd1"
def test_manifest_hash_sha_and_no_reselection(monkeypatch):
    assert hashlib.sha256(t.MANIFEST.read_bytes()).hexdigest() == t.TEST_MANIFEST_SHA
    assert t.preflight() == t.load_manifest()  # It is load-only: no selector exists in this module.
def test_consumed_inventory_excludes_sources_and_wrong_sha_fails(monkeypatch):
    d=t.load_manifest(); used={t.identity(x) for x in d["complete_prior_explicit_identity_inventory"]}; assert all(t.identity(x) not in used for x in d["sources"].values())
    monkeypatch.setattr(t,"TEST_MANIFEST_SHA","0"*64)
    with pytest.raises(RuntimeError): t.load_manifest()
def test_cache_and_shard_validation(tmp_path):
    d=t.load_manifest(); q=torch.zeros((1,16,4096,128),dtype=torch.float16); k=torch.zeros((1,8,4096,128),dtype=torch.float16); v=k.clone(); p=t.cache_path(tmp_path,"narrative",0)
    torch.save({"metadata":{"manifest_sha256":t.TEST_MANIFEST_SHA,"model_sha":t.MODEL_SHA,"model_revision":t.MODEL_SHA,"v2_config_sha256":t.V2_SHA,"v3_config_sha256":t.V3_SHA,"sequence":"narrative","layer":0,"context_length":4096,"source_identity":d["sources"]["narrative"],"tensor_shapes":{"q":list(q.shape),"k":list(k.shape),"v":list(v.shape)}},"query":q,"key":k,"value":v},p); assert t.validate_cache(p,"narrative",0)[0]
    rows=[{"sequence":"narrative","layer":0,"context_length":4096,"position":z,"q_head":h,"kv_head":h//2,"method":m,"metrics":{k:0 for k in t.METRIC_FIELDS},"traffic":{k:0 for k in t.TRAFFIC_FIELDS}} for z in t.POSITIONS for h in range(16) for m in t.METHODS]
    shard=t.shard_path(tmp_path,"narrative",0); shard.write_text(json.dumps({"manifest_sha256":t.TEST_MANIFEST_SHA,"model_sha":t.MODEL_SHA,"v2_config_sha256":t.V2_SHA,"v3_config_sha256":t.V3_SHA,"sequence":"narrative","layer":0,"context_length":4096,"rows":rows})); assert t.validate_shard(shard,"narrative",0)[0]
def test_selection_surface_and_no_target_optimizer_or_config_writer():
    source=Path(t.__file__).read_text(); policy=t.load_manifest()["selection_policy"]
    assert "Q/K/V" in policy and "tokenizer length" in policy
    assert not any(x in source for x in ("optimizer", "calibrate", "V3_CONFIG.write"))
    assert hashlib.sha256(t.V3_CONFIG.read_bytes()).hexdigest() == t.V3_SHA
def test_classification_boundaries_and_diagnostics_do_not_matter():
    def x(c,r,b): return {"metrics":{"cosine_similarity":{"mean":c},"relative_l2_error":{"mean":r}},"traffic":{"total_kv_bytes":{"mean":b}}}
    p={"frozen_cascadekv_v3":x(.985,.12,1),"uniform_v1_10":x(0,0,1),"frozen_cascadekv_v2":x(0,.13,0)}; assert t.classify(p)=="TARGET-MODEL-TRANSFER-PASSED"; p["frozen_cascadekv_v3"]["diagnostics"]={"bad":999}; assert t.classify(p)=="TARGET-MODEL-TRANSFER-PASSED"
    p["frozen_cascadekv_v2"]["metrics"]["relative_l2_error"]["mean"]=.12; assert t.classify(p)=="TARGET-MODEL-COMPARATIVE-GATE-FAILED"

def test_synthetic_capture_binds_target_model_and_restores_every_global(tmp_path, monkeypatch):
    """No weights: fake only the expensive base capture after production binding."""
    seen=[]; original={x:getattr(t.base,x) for x in ("MODEL","MODEL_SHA","MANIFEST","V2_CONFIG","V2_SHA","TEST_MANIFEST_SHA","METHODS","POSITIONS","require_inputs","load_manifest")}
    def fake_capture(sequence, layer, root):
        seen.append((t.base.MODEL,t.base.MODEL_SHA))
        d=t.load_manifest(); q=torch.zeros((1,16,4096,128),dtype=torch.float16); k=torch.zeros((1,8,4096,128),dtype=torch.float16)
        p=t.cache_path(root,sequence,layer); p.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"metadata":{"manifest_sha256":t.TEST_MANIFEST_SHA,"model_sha":t.MODEL_SHA,"v2_config_sha256":t.V2_SHA,"sequence":sequence,"source_identity":d["sources"][sequence],"context_length":4096,"layer":layer,"tensor_shapes":{"q":list(q.shape),"k":list(k.shape),"v":list(k.shape)},"attention_scaling":1.0},"query":q,"key":k,"value":k.clone()},p)
    monkeypatch.setattr(t,"BASE_CAPTURE",fake_capture)
    t.capture("narrative",0,tmp_path/"cache")
    assert seen == [("Qwen/Qwen3-1.7B","70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")]
    assert seen[0] != ("Qwen/Qwen3-0.6B","c1899de289a04d12100db370d81485cdf75e47ca")
    assert t.validate_cache(t.cache_path(tmp_path/"cache","narrative",0),"narrative",0)[0]
    assert {x:getattr(t.base,x) for x in original} == original

def test_bound_base_restores_on_exception():
    original={x:getattr(t.base,x) for x in ("MODEL","MODEL_SHA","MANIFEST","V2_CONFIG","V2_SHA","TEST_MANIFEST_SHA","METHODS","POSITIONS","require_inputs","load_manifest")}
    with pytest.raises(RuntimeError):
        with t.bound_base():
            assert t.base.MODEL == t.MODEL and t.base.MODEL_SHA == t.MODEL_SHA
            raise RuntimeError("forced")
    assert {x:getattr(t.base,x) for x in original} == original

def test_runner_uses_phases_not_file_existence_and_status_has_no_model_load():
    runner=Path("scripts/run_cascadekv_v3_qwen3_1p7b_4k_test.sh").read_text()
    assert "--capture" in runner and "--evaluate" in runner and "--merge" in runner
    assert "[ -e " not in runner and "AutoModel" not in t.status.__code__.co_names

BASE_GLOBALS=("MODEL","MODEL_SHA","MANIFEST","V2_CONFIG","V2_SHA","TEST_MANIFEST_SHA","METHODS","POSITIONS","require_inputs","load_manifest","validate_cache")

def _base_state(): return {x:getattr(t.base,x) for x in BASE_GLOBALS}

def _cache(root, s="narrative", l=0):
    d=t.load_manifest(); q=torch.zeros((1,16,4096,128),dtype=torch.float16); k=torch.zeros((1,8,4096,128),dtype=torch.float16)
    p=t.cache_path(root,s,l); p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"metadata":{"manifest_sha256":t.TEST_MANIFEST_SHA,"model_sha":t.MODEL_SHA,"model_revision":t.MODEL_SHA,"v2_config_sha256":t.V2_SHA,"v3_config_sha256":t.V3_SHA,"sequence":s,"layer":l,"context_length":4096,"source_identity":d["sources"][s],"tensor_shapes":{"q":list(q.shape),"k":list(k.shape),"v":list(k.shape)}},"query":q,"key":k,"value":k.clone()},p)
    assert t.validate_cache(p,s,l)[0]; return p

def _rows(s, l, methods=t.METHODS):
    # Deliberately unambiguous transfer-pass data: v3 beats v2 on L2 and is
    # no larger than uniform10.  All required base summary fields are finite.
    out=[]
    for p in t.POSITIONS:
      for h in range(16):
       for m in methods:
        l2={"frozen_cascadekv_v3":.05,"frozen_cascadekv_v2":.08}.get(m,.07)
        kv={"frozen_cascadekv_v3":80.,"uniform_v1_10":100.}.get(m,120.)
        metrics={k:0.0 for k in t.METRIC_FIELDS}; metrics.update({"cosine_similarity":.99,"relative_l2_error":l2})
        traffic={k:1.0 for k in t.TRAFFIC_FIELDS}; traffic["total_kv_bytes"]=kv
        traffic.update({"total_k_vs_dense_fp16_k":.1,"total_kv_vs_dense":.1,"total_k_vs_flat_q8k4_5":.1,"total_kv_vs_flat_q8k4_5":.1,"total_k_vs_flat_q8k4_10":.1,"total_kv_vs_flat_q8k4_10":.1,"total_k_vs_uniform_v1_10":.1,"total_kv_vs_uniform_v1_10":.1})
        out.append({"sequence":s,"layer":l,"context_length":4096,"position":p,"q_head":h,"kv_head":h//2,"method":m,"metrics":metrics,"traffic":traffic})
    return out

def _shard(root, s="narrative", l=0, methods=t.METHODS):
    p=t.shard_path(root,s,l); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"manifest_sha256":t.TEST_MANIFEST_SHA,"model_sha":t.MODEL_SHA,"v2_config_sha256":t.V2_SHA,"v3_config_sha256":t.V3_SHA,"sequence":s,"layer":l,"context_length":4096,"rows":_rows(s,l,methods)}))
    assert t.validate_shard(p,s,l)[0]; return p

def test_validate_cache_calls_immutable_base_validator_under_target_binding(tmp_path, monkeypatch):
    p=_cache(tmp_path); seen=[]; preserved=t.BASE_VALIDATE_CACHE
    def observed(path,s,l):
        seen.append((t.base.MODEL,t.base.MODEL_SHA,t.base.V2_CONFIG,t.base.V2_SHA,t.base.METHODS)); return preserved(path,s,l)
    monkeypatch.setattr(t,"BASE_VALIDATE_CACHE",observed)
    assert t.validate_cache(p,"narrative",0)[0]
    assert seen == [(t.MODEL,t.MODEL_SHA,t.V2_CONFIG,t.V2_SHA,t.BASE_METHODS)]

def test_synthetic_two_pass_evaluation_and_exact_remapping(tmp_path, monkeypatch):
    cache=tmp_path/"cache"; shards=tmp_path/"shards"; _cache(cache); seen=[]; original=_base_state()
    def fake(sequence,layer,cache_root,out):
        seen.append((t.base.MODEL,t.base.MODEL_SHA,t.base.V2_CONFIG,t.base.V2_SHA,t.base.METHODS))
        p=t.shard_path(out,sequence,layer); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"rows":_rows(sequence,layer,t.BASE_METHODS)}))
    monkeypatch.setattr(t,"BASE_EVALUATE",fake)
    t.evaluate("narrative",0,cache,shards)
    assert seen == [(t.MODEL,t.MODEL_SHA,t.V2_CONFIG,t.V2_SHA,t.BASE_METHODS),(t.MODEL,t.MODEL_SHA,t.V3_CONFIG,t.V3_SHA,t.BASE_METHODS)]
    p=t.shard_path(shards,"narrative",0); assert t.validate_shard(p,"narrative",0)[0]
    d=json.loads(p.read_text()); rows=d["rows"]
    assert len(rows)==336 and {r["method"] for r in rows}==set(t.METHODS) and "cascadekv_v2" not in {r["method"] for r in rows}
    for position in t.POSITIONS:
      for q_head in range(16):
        rs=[r for r in rows if (r["position"],r["q_head"]) == (position,q_head)]
        assert len(rs)==7 and {r["method"] for r in rs}==set(t.METHODS) and {r["kv_head"] for r in rs}=={q_head//2}
    assert (d["model_sha"],d["manifest_sha256"],d["v2_config_sha256"],d["v3_config_sha256"]) == (t.MODEL_SHA,t.TEST_MANIFEST_SHA,t.V2_SHA,t.V3_SHA)
    assert _base_state()==original

@pytest.mark.parametrize("failure_call",[1,2])
def test_evaluate_exception_restores_all_base_globals_and_never_promotes(tmp_path, monkeypatch, failure_call):
    cache=tmp_path/"cache"; shards=tmp_path/"shards"; _cache(cache); original=_base_state(); calls=0
    def boom(*args):
        nonlocal calls; calls+=1
        if calls==failure_call: raise RuntimeError("forced evaluation failure")
        p=t.shard_path(args[3],args[0],args[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({"rows":_rows(args[0],args[1],t.BASE_METHODS)}))
    monkeypatch.setattr(t,"BASE_EVALUATE",boom)
    with pytest.raises(RuntimeError,match="forced evaluation failure"): t.evaluate("narrative",0,cache,shards)
    assert _base_state()==original
    assert not t.validate_shard(t.shard_path(shards,"narrative",0),"narrative",0)[0]

def test_valid_artifact_resume_is_validity_based(tmp_path, monkeypatch):
    cache=tmp_path/"cache"; shards=tmp_path/"shards"; cp=_cache(cache); _shard(shards)
    calls={"capture":0,"evaluate":0}
    def no_capture(*args): calls["capture"]+=1; raise AssertionError("must reuse valid cache")
    def no_eval(*args): calls["evaluate"]+=1; raise AssertionError("must reuse valid shard")
    monkeypatch.setattr(t,"BASE_CAPTURE",no_capture); monkeypatch.setattr(t,"BASE_EVALUATE",no_eval)
    t.capture("narrative",0,cache); t.evaluate("narrative",0,cache,shards); assert calls=={"capture":0,"evaluate":0}
    d=torch.load(cp,weights_only=False); d["metadata"]["model_revision"]="bad"; torch.save(d,cp)
    def synth_capture(s,l,out):
        calls["capture"]+=1; _cache(out,s,l)
    monkeypatch.setattr(t,"BASE_CAPTURE",synth_capture); t.capture("narrative",0,cache); assert calls["capture"]==1
    sd=json.loads(t.shard_path(shards,"narrative",0).read_text()); sd["rows"][0]["traffic"]["total_kv_bytes"]=float("nan"); t.shard_path(shards,"narrative",0).write_text(json.dumps(sd))
    def synth_eval(s,l,cache_root,out):
        calls["evaluate"]+=1; p=t.shard_path(out,s,l); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({"rows":_rows(s,l,t.BASE_METHODS)}))
    monkeypatch.setattr(t,"BASE_EVALUATE",synth_eval); t.evaluate("narrative",0,cache,shards); assert calls["evaluate"]==2

def test_complete_and_fail_closed_synthetic_merges(tmp_path):
    complete=tmp_path/"complete"
    for s in t.SPECS:
      for l in t.LAYERS: _shard(complete,s,l)
    out=tmp_path/"merged.json"; t.merge(complete,out); d=json.loads(out.read_text())
    assert d["status"]=="complete" and d["classification"]=="TARGET-MODEL-TRANSFER-PASSED" and d["methods"]==list(t.METHODS)
    assert {k:len(d["diagnostics"][k]) for k in ("per_source","per_layer","per_position","per_q_head","per_kv_head")} == {"per_source":3,"per_layer":5,"per_position":3,"per_q_head":16,"per_kv_head":8}
    assert set(d["direct_deltas"]) == {"v3_minus_v2","v3_minus_uniform10","v3_minus_flat5"} and d["diagnostics"]["v3_minus_completed_0p6b_v3"]["classification_input"] is False
    assert (d["target_model"]["name"],d["target_model"]["revision"],d["manifest_sha256"],d["v2_config_sha256"],d["parent_v3_config_sha256"],d["protocol_tag"]["commit"]) == (t.MODEL,t.MODEL_SHA,t.TEST_MANIFEST_SHA,t.V2_SHA,t.V3_SHA,t.TAG_COMMIT)
    missing=tmp_path/"missing"; missing.mkdir()
    for p in complete.iterdir():
        if p.name != t.shard_path(complete,"narrative",0).name: (missing/p.name).write_bytes(p.read_bytes())
    for root,corrupt in ((missing,False),(complete,True)):
        if corrupt:
            p=t.shard_path(root,"narrative",0); x=json.loads(p.read_text()); x["rows"][0]["metrics"]["cosine_similarity"]=float("inf"); p.write_text(json.dumps(x))
        target=tmp_path/("corrupt.json" if corrupt else "missing.json"); t.merge(root,target); result=json.loads(target.read_text())
        assert (result["status"],result["classification"]) != ("complete","TARGET-MODEL-TRANSFER-PASSED")
        assert result["status"]=="partial" and result["classification"]=="TARGET-MODEL-TEST-INVALID"

@pytest.mark.parametrize("value",[float("nan"),float("inf")])
def test_nonfinite_shard_values_fail_closed(tmp_path, value):
    p=_shard(tmp_path); d=json.loads(p.read_text()); d["rows"][0]["metrics"]["cosine_similarity"]=value; p.write_text(json.dumps(d))
    ok,msg=t.validate_shard(p,"narrative",0); assert not ok and msg=="non-finite metric/traffic"
