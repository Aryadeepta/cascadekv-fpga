import hashlib

import pytest

from experiments import cascadekv_v3_qwen3_1p7b_wide_vaware_dev as w


def candidate(i, **overrides):
    return {"name": f"target_wide_T{i}", "cosine": 0.985, "relative_l2": 0.119, "kv": 99, **overrides}


def test_frozen_inputs_target_bindings_and_fresh_identities():
    manifest = w.require_inputs()
    assert hashlib.sha256(w.MANIFEST.read_bytes()).hexdigest() == w.MANIFEST_SHA
    assert w.MODEL == "Qwen/Qwen3-1.7B"
    assert w.MODEL_SHA == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert w.CONFIG_SHA == "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
    assert manifest["target_model"] == {"name": w.MODEL, "resolved_commit_sha": w.MODEL_SHA, "config_sha256": w.CONFIG_SHA}
    assert w.sha(w.V1) == w.V1_SHA and w.sha(w.V2) == w.V2_SHA and w.sha(w.V3) == w.V3_SHA
    assert w.sha(w.NARROW) == w.NARROW_SHA

    sources = manifest["sources"]
    assert {name: source["index"] for name, source in sources.items()} == {
        "narrative_calibration": 9, "narrative_validation": 10,
        "report_calibration": 10, "report_validation": 11,
        "qa_calibration": 9, "qa_validation": 10,
    }
    calibration = {w.identity(source) for source in sources.values() if source["role"] == "calibration"}
    validation = {w.identity(source) for source in sources.values() if source["role"] == "validation"}
    consumed = {w.identity(row) for row in manifest["complete_consumed_identity_inventory"]}
    assert {name for name, source in sources.items() if source["role"] == "calibration"} == {
        "narrative_calibration", "report_calibration", "qa_calibration"
    }
    assert {name for name, source in sources.items() if source["role"] == "validation"} == {
        "narrative_validation", "report_validation", "qa_validation"
    }
    assert len(calibration) == 3 and len(validation) == 3
    assert calibration.isdisjoint(validation)
    assert (calibration | validation).isdisjoint(consumed)


def test_action_semantics_are_exactly_frozen():
    assert w.ACTION_SEMANTICS == {
        "A0": [0.05, "hierarchy using frozen 5%-routing profile"],
        "A1": [0.075, "hierarchy using frozen 5%-routing profile"],
        "A2": [0.10, "hierarchy using frozen 5%-routing profile"],
        "A3": [0.15, "hierarchy using frozen 5%-routing profile"],
        "A4": [0.05, "flat"],
    }
    assert w.ACTIONS == ("A0", "A1", "A2", "A3", "A4")


def test_exact_targets_and_each_invalid_order_fails_closed():
    assert w.wide_targets(10, 20, 50) == {"T0": 10, "T1": 15.0, "T2": 20, "T3": 30.0, "T4": 40.0, "T5": 50}
    for baselines in ((10, 10, 20), (10, 20, 20), (20, 10, 50), (10, 50, 20)):
        with pytest.raises(RuntimeError, match="WIDE-TARGET-BASELINE-ORDER-INVALID"):
            w.wide_targets(*baselines)
    assert w.wide_targets.__code__.co_argcount == 3
    assert "NARROW" not in w.wide_targets.__code__.co_names


def test_absolute_quality_boundaries_pass_when_comparators_are_strictly_worse():
    assert w.passes(candidate(0, cosine=0.985, relative_l2=0.120, kv=99), 100, 0.121, 0.121) is True


def test_equal_flat5_bandwidth_fails():
    assert w.passes(candidate(0, kv=100), 100, 0.121, 0.121) is False


def test_equal_v2_relative_l2_fails():
    assert w.passes(candidate(0, relative_l2=0.120), 100, 0.120, 0.121) is False


def test_equal_frozen_v3_relative_l2_fails():
    assert w.passes(candidate(0, relative_l2=0.120), 100, 0.121, 0.120) is False


def test_fixed_winner_order_and_malformed_fail_closed():
    candidates = [candidate(i) for i in range(6)]
    candidates[0]["cosine"] = 0.984
    candidates[1]["relative_l2"] = 0.01
    assert w.choose_winner(list(reversed(candidates)), 100, 0.121, 0.121)["name"] == "target_wide_T1"
    assert w.choose_winner(candidates[:-1], 100, 0.121, 0.121) is None
    candidates[0] = {**candidates[1]}
    assert w.choose_winner(candidates, 100, 0.121, 0.121) is None
    unknown = [candidate(i) for i in range(6)]
    unknown[-1]["name"] = "target_wide_T6"
    assert w.choose_winner(unknown, 100, 0.121, 0.121) is None
    assert w.choose_winner([candidate(0)] + [candidate(i, relative_l2=0.0001, cosine=0.999) for i in range(1, 6)], 100, 0.121, 0.121)["name"] == "target_wide_T0"


def test_all_required_candidates_failing_has_no_winner():
    candidates = [candidate(i, cosine=0.984) for i in range(6)]
    assert {row["name"] for row in candidates} == set(w.TARGET_NAMES)
    assert w.choose_winner(candidates, 100, 0.121, 0.121) is None


def test_status_is_zero_and_harness_has_only_safe_commands():
    status = w.status()
    assert (status["valid_caches"], status["valid_action_shards"], status["total_caches"], status["total_action_shards"]) == (0, 0, 30, 30)
    assert status["result_exists"] is False
    source = w.Path(w.__file__).read_text()
    harness = w.Path("scripts/run_cascadekv_v3_qwen3_1p7b_wide_vaware_dev.sh").read_text()
    safe_only = source + harness
    for forbidden in ("--capture", "--evaluate", "--merge", "AutoModel.from_pretrained", "from_pretrained"):
        assert forbidden not in safe_only
    assert "optimizer" not in source.lower()
