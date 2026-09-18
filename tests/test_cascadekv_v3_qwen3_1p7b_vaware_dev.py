import hashlib, json
from pathlib import Path
import pytest
from experiments import cascadekv_v3_qwen3_1p7b_vaware_dev as v

def test_manifest_freeze_identity_and_target_bindings():
    d=v.load_manifest()
    assert hashlib.sha256(v.MANIFEST.read_bytes()).hexdigest()==v.MANIFEST_SHA
    assert d["target_model"]["resolved_commit_sha"]==v.MODEL_SHA
    assert d["target_model"]["config_sha256"]==v.CONFIG_SHA
    used={v.identity(x) for x in d["complete_consumed_identity_inventory"]}
    assert len(d["sources"])==6 and all(v.identity(x) not in used for x in d["sources"].values())
    assert {x["role"] for x in d["sources"].values()}=={"calibration","validation"}
    assert all(x["token_length_proof"]["at_least"] for x in d["sources"].values())

def test_architecture_actions_schedule_and_no_reselection():
    d=v.load_manifest(); assert d["geometry"]["static_schedule_cells"]==40
    assert d["immutable_architecture"]["actions"]=={"A0":[.05,"hierarchy"],"A1":[.075,"hierarchy"],"A2":[.1,"hierarchy"],"A3":[.15,"hierarchy"],"A4":[.05,"flat"]}
    assert v.load_manifest()==d
    source=Path(v.__file__).read_text()
    assert "AutoModel" not in source and "from_pretrained" not in source and "ARCHIVED.read_text" in source
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
        rows.append({"source":source,"role":"calibration","layer":layer,"position":pos,"q_head":head,"kv_head":head//2,"action":action,"metrics":{x:0.0 for x in v.METRICS},"traffic":{"total_kv_bytes":0.0}})
    path=v.shard_path(tmp_path,source,layer)
    path.write_text(json.dumps({"manifest_sha256":v.MANIFEST_SHA,"model_sha":v.MODEL_SHA,"source_identity":d["sources"][source],"role":"calibration","layer":layer,"context_length":4096,"rows":rows}))
    assert v.validate_shard(path,source,layer)[0]
    rows.pop(); path.write_text(json.dumps({"manifest_sha256":v.MANIFEST_SHA,"model_sha":v.MODEL_SHA,"source_identity":d["sources"][source],"role":"calibration","layer":layer,"context_length":4096,"rows":rows}))
    assert not v.validate_shard(path,source,layer)[0]
