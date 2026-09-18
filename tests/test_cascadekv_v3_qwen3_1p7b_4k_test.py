import hashlib, json
from pathlib import Path
import pytest, torch
from experiments import cascadekv_v3_qwen3_1p7b_4k_test as t

def test_frozen_target_architecture_and_literal_table():
    t.require_inputs(); t.verify_architecture(dict(t.REQUIRED_ARCHITECTURE))
    with pytest.raises(RuntimeError): t.verify_architecture({**t.REQUIRED_ARCHITECTURE,"head_dim":64})
    cfg=json.loads(t.V3_CONFIG.read_text()); assert len(cfg["exact_layer_head_action_table"]) == 40
def test_manifest_hash_sha_and_no_reselection(monkeypatch):
    assert hashlib.sha256(t.MANIFEST.read_bytes()).hexdigest() == t.TEST_MANIFEST_SHA
    assert t.preflight() == t.load_manifest()  # It is load-only: no selector exists in this module.
def test_consumed_inventory_excludes_sources_and_wrong_sha_fails(monkeypatch):
    d=t.load_manifest(); used={t.identity(x) for x in d["complete_prior_explicit_identity_inventory"]}; assert all(t.identity(x) not in used for x in d["sources"].values())
    monkeypatch.setattr(t,"TEST_MANIFEST_SHA","0"*64)
    with pytest.raises(RuntimeError): t.load_manifest()
def test_cache_and_shard_validation(tmp_path):
    d=t.load_manifest(); q=torch.zeros((1,16,4096,128),dtype=torch.float16); k=torch.zeros((1,8,4096,128),dtype=torch.float16); v=k.clone(); p=t.cache_path(tmp_path,"narrative",0)
    torch.save({"metadata":{"manifest_sha256":t.TEST_MANIFEST_SHA,"model_sha":t.MODEL_SHA,"model_revision":t.MODEL_SHA,"v2_config_sha256":t.V2_SHA,"v3_config_sha256":t.V3_SHA,"sequence":"narrative","layer":0,"context_length":4096,"source_identity":d["sources"]["narrative"]},"query":q,"key":k,"value":v},p); assert t.validate_cache(p,"narrative",0)[0]
    rows=[{"position":z,"q_head":h,"kv_head":h//2,"method":m,"metrics":{k:0 for k in t.METRIC_FIELDS},"traffic":{k:0 for k in t.TRAFFIC_FIELDS}} for z in t.POSITIONS for h in range(16) for m in t.METHODS]
    shard=t.shard_path(tmp_path,"narrative",0); shard.write_text(json.dumps({"manifest_sha256":t.TEST_MANIFEST_SHA,"model_sha":t.MODEL_SHA,"v2_config_sha256":t.V2_SHA,"v3_config_sha256":t.V3_SHA,"sequence":"narrative","layer":0,"context_length":4096,"rows":rows})); assert t.validate_shard(shard,"narrative",0)[0]
def test_selection_surface_and_no_target_optimizer_or_config_writer():
    source=Path(t.__file__).read_text(); policy=t.load_manifest()["selection_policy"]
    assert "Q/K/V" in policy and "tokenizer length" in policy
    assert not any(x in source for x in ("optimizer", "calibrate", "V3_CONFIG.write", "json.dump("))
    assert hashlib.sha256(t.V3_CONFIG.read_bytes()).hexdigest() == t.V3_SHA
def test_classification_boundaries_and_diagnostics_do_not_matter():
    def x(c,r,b): return {"metrics":{"cosine_similarity":{"mean":c},"relative_l2_error":{"mean":r}},"traffic":{"total_kv_bytes":{"mean":b}}}
    p={"frozen_cascadekv_v3":x(.985,.12,1),"uniform_v1_10":x(0,0,1),"frozen_cascadekv_v2":x(0,.13,0)}; assert t.classify(p)=="TARGET-MODEL-TRANSFER-PASSED"; p["frozen_cascadekv_v3"]["diagnostics"]={"bad":999}; assert t.classify(p)=="TARGET-MODEL-TRANSFER-PASSED"
