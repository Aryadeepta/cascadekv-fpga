import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from experiments import cascadekv_v3_qwen3_1p7b_t1_4k_test as t


def pool(cosine=.985, l2=.120, v2=.121, v3=.121, traffic=9, flat5=10):
    def method(c=.0, r=.0, b=0):
        return {"metrics":{"cosine_similarity":{"mean":c},"relative_l2_error":{"mean":r}}, "traffic":{"total_kv_bytes":{"mean":b}}}
    return {"frozen_qwen3_1p7b_T1":method(cosine,l2,traffic), "frozen_cascadekv_v2":method(0,v2,0), "original_frozen_cascadekv_v3_zero_shot":method(0,v3,0), "flat_q8k4_5":method(0,0,flat5)}


def test_config_and_development_table_are_exactly_frozen():
    config, manifest, development = t.require_inputs()
    assert hashlib.sha256(t.CONFIG.read_bytes()).hexdigest() == t.CONFIG_SHA
    assert config["exact_layer_head_action_table"] == development["winner_table"] == t.EXPECTED_TABLE
    assert len(t.EXPECTED_TABLE) == config["schedule_cell_count"] == 40
    assert config["composition"] == {"A0":13,"A1":11,"A2":4,"A3":4,"A4":8}
    assert config["action_semantics"]["A2"] == {"candidate_budget":.10,"routing_profile":"frozen 5%-routing profile","mode":"hierarchy"}
    assert config["target_model"] == {"name":t.MODEL,"revision":t.REVISION,"config_sha256":t.MODEL_CONFIG_SHA}


def test_manifest_is_frozen_and_sources_are_exact_fresh_mechanical_choices():
    manifest = t.load_manifest()
    assert hashlib.sha256(t.MANIFEST.read_bytes()).hexdigest() == t.MANIFEST_SHA
    assert {k:v["index"] for k,v in manifest["sources"].items()} == {"narrative":11,"report":12,"qa":11}
    assert all(x["eligibility"]["at_least_4096"] and x["eligibility"]["bounded_frozen_tokenizer_length"] == 4096 for x in manifest["sources"].values())
    for family, source in manifest["sources"].items():
        assert source["index"] not in manifest["complete_consumed_identity_inventory"][family]["unavailable_indices"]
    assert "ascending dataset-index order" in manifest["selection_rule"]
    assert "never reselects" in manifest["selection_rule"]
    assert t.preflight() == manifest


def test_methods_and_predata_status_are_exact():
    assert t.METHODS == ("dense_exact","flat_q8k4_5","flat_q8k4_10","uniform_v1_5","uniform_v1_10","frozen_cascadekv_v2","original_frozen_cascadekv_v3_zero_shot","frozen_qwen3_1p7b_T1")
    state=t.status()
    assert (state["valid_caches"],state["total_caches"],state["valid_shards"],state["total_shards"],state["result_exists"],state["execution_enabled"]) == (0,15,0,15,False,False)
    source=Path(t.__file__).read_text()
    assert "def capture" not in source and "def evaluate" not in source and "def merge" not in source and "from_pretrained" not in source


def test_gate_boundaries_and_strict_comparators():
    assert t.classify(pool()) == t.LABELS["pass"]
    assert t.classify(pool(v2=.120)) == t.LABELS["comparative_fail"]
    assert t.classify(pool(v3=.120)) == t.LABELS["comparative_fail"]
    assert t.classify(pool(traffic=10,flat5=10)) == t.LABELS["comparative_fail"]
    assert t.classify(pool(cosine=.984)) == t.LABELS["absolute_fail"]
    assert t.classify(pool(l2=.121)) == t.LABELS["absolute_fail"]
    assert t.classify({}) == t.LABELS["invalid"]
    assert "uniform_v1_10" not in json.dumps(t.load_manifest()["pass_gates"])


def test_wrong_immutable_input_fails_closed(monkeypatch):
    monkeypatch.setattr(t, "MANIFEST_SHA", "0" * 64)
    with pytest.raises(RuntimeError, match="T1-CONFIRMATORY-TEST-INVALID"):
        t.preflight()


RUNNER = t.ROOT / "scripts/run_cascadekv_v3_qwen3_1p7b_t1_4k_test.sh"


def run_runner(*args):
    return subprocess.run(["bash", str(RUNNER), *args], cwd=t.ROOT, text=True, capture_output=True)


def test_runner_is_preparation_only_and_uses_python3():
    runner = RUNNER.read_text()
    assert "uv run python3" in runner
    assert "uv run python " not in runner
    assert not any(option in runner for option in ("--capture", "--evaluate", "--merge"))


@pytest.mark.parametrize("args", [(), ("--preflight",), ("--status",)])
def test_runner_allowed_modes_are_safe_and_leave_no_artifacts(args):
    before = (t.CACHE_DIR.exists(), t.SHARD_DIR.exists(), t.RESULT.exists())
    completed = run_runner(*args)
    assert completed.returncode == 0, completed.stderr
    assert (t.CACHE_DIR.exists(), t.SHARD_DIR.exists(), t.RESULT.exists()) == before == (False, False, False)
    state = t.status()
    assert (state["valid_caches"], state["total_caches"], state["valid_shards"], state["total_shards"], state["result_exists"]) == (0, 15, 0, 15, False)


def test_runner_rejects_unsupported_arguments():
    completed = run_runner("--capture")
    assert completed.returncode != 0
