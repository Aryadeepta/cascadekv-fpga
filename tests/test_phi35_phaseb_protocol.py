import copy
import json
import subprocess
from pathlib import Path

import pytest

from cascadekv.phi35_phaseb import (
    ACTIONS, PROTOCOL, REVISION, TARGETS, PhaseBError, optimize_static_schedule,
    QWEN_PROTOCOL_COMMIT, QWEN_PROTOCOL_TAG, QWEN_PROTOCOL_TAG_OBJECT_SHA,
    runner_preflight, select_mechanically_inspected, validate_protocol,
    verify_phase_a, verify_runtime_closure,
)


def frozen():
    return json.loads(PROTOCOL.read_text())


def test_exact_target_geometry_mapping_and_observation_counts():
    p = validate_protocol()
    assert p["target_model"]["revision"] == REVISION
    assert p["geometry"] == {"context_length": 8192, "sampled_layers": [0, 8, 16, 24, 31], "sampled_query_positions": [4095, 6143, 8191], "q_heads": 32, "kv_heads": 32, "head_dimension": 96, "kv_head_mapping": "kv_head = q_head", "static_schedule_cells": 160, "observations_per_source_per_method": 480, "observations_per_method_three_sources": 1440}


def test_prior_only_routing_provenance_depth_transfer_and_reserves():
    p = frozen()
    assert p["prior_qwen_provenance"]["protocol_tag"] == QWEN_PROTOCOL_TAG
    assert p["prior_qwen_provenance"]["protocol_tag_object_sha"] == QWEN_PROTOCOL_TAG_OBJECT_SHA
    assert p["prior_qwen_provenance"]["protocol_commit"] == QWEN_PROTOCOL_COMMIT
    assert p["prior_qwen_provenance"]["config_sha256"] == "5d364bc4e351061c243a9166c0a9ccdd68c7cefd74265e4135d937b323e6e505"
    assert p["prior_qwen_provenance"]["manifest_sha256"] == "654596830b06bfc6e9740c5363273bcbc408b769fabd92a69ef80fb182a84fb9"
    five, ten = p["routing_profiles"]["phi35-8k-depth-transfer-qwen5-v1"], p["routing_profiles"]["phi35-8k-depth-transfer-qwen10-v1"]
    assert five["depth_mapping"] == ten["depth_mapping"] == {"0": 0, "8": 7, "16": 14, "24": 21, "31": 27}
    assert five["lambdas"] == {"0": .25, "8": 1.5, "16": .5, "24": 1.5, "31": .25}
    assert ten["lambdas"] == {"0": 1., "8": 1., "16": 1., "24": .75, "31": .25}
    assert five["reserve"]["sink"] == 16 and five["reserve"]["local"] == 64
    assert ten["reserve"]["sink"] == 32 and ten["reserve"]["local"] == 128


def test_prior_qwen_annotated_tag_object_and_dereferenced_commit_are_distinct():
    tag_object = subprocess.check_output(["git", "rev-parse", QWEN_PROTOCOL_TAG], text=True).strip()
    protocol_commit = subprocess.check_output(["git", "rev-parse", f"{QWEN_PROTOCOL_TAG}^{{commit}}"], text=True).strip()
    assert tag_object == QWEN_PROTOCOL_TAG_OBJECT_SHA
    assert protocol_commit == QWEN_PROTOCOL_COMMIT


def test_exact_action_menu_and_uniform10():
    p = frozen()
    assert tuple(p["actions"]) == ACTIONS
    assert p["actions"]["A0"] == {"kind": "hierarchy", "candidate_fraction": .05, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}
    assert p["actions"]["A1"]["candidate_fraction"] == .075
    assert p["actions"]["A2"] == {"kind": "hierarchy", "candidate_fraction": .10, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}
    assert p["actions"]["A3"]["candidate_fraction"] == .15
    assert p["actions"]["A4"] == {"kind": "flat_q8k4", "candidate_fraction": .05, "routing_profile": None}
    assert p["uniform_hierarchy_10"]["routing_profile"] == "phi35-8k-depth-transfer-qwen10-v1"


def test_fixed_gates_targets_frontiers_inventory_and_firewall():
    p = frozen()
    assert p["quality_gates"]["min_mean_cosine"] == .985
    assert p["quality_gates"]["max_mean_relative_l2"] == .120
    assert [p["bandwidth"]["targets"][x] for x in TARGETS] == [.115, .13, .145, .16, .18, "highest_traffic_strictly_below_calibration_flat5"]
    assert p["bandwidth"]["primary_gate"] == "strictly_less_than_fresh_validation_flat_q8k4_5"
    assert [p["source_specs"][x]["first_permitted_index"] for x in ("narrative", "report", "qa")] == [13, 14, 13]
    assert p["development_inventory"] == {"calibration": {"narrative": 2, "report": 2, "qa": 2}, "validation": {"narrative": 1, "report": 1, "qa": 1}}
    assert "modify action cells" in p["firewall"]["validation_forbidden"]


def test_protocol_rejects_selected_identity_or_wrong_a2():
    p = frozen(); p["selected_sources"] = {"narrative": 13}
    with pytest.raises(PhaseBError, match="no selected identities"):
        validate_protocol(p)
    p = frozen(); p["actions"]["A2"]["routing_profile"] = "phi35-8k-depth-transfer-qwen10-v1"
    with pytest.raises(PhaseBError, match="action menu"):
        validate_protocol(p)


def test_injected_mechanical_selection_is_ascending_and_text_free():
    rows = [
        {"dataset_index": 12, "field_exists": True, "is_python_string": True, "character_count": 9000, "bounded_tokenizer_length": 8192},
        {"dataset_index": 13, "field_exists": True, "is_python_string": True, "character_count": 9000, "bounded_tokenizer_length": 8191},
        {"dataset_index": 14, "field_exists": True, "is_python_string": True, "character_count": 9000, "bounded_tokenizer_length": 8192},
        {"dataset_index": 15, "field_exists": True, "is_python_string": True, "character_count": 9000, "bounded_tokenizer_length": 8192},
        {"dataset_index": 16, "field_exists": True, "is_python_string": True, "character_count": 9000, "bounded_tokenizer_length": 8192},
    ]
    chosen, rejected, last_inspected = select_mechanically_inspected("narrative", reversed(rows))
    assert [x["dataset_index"] for x in chosen] == [14, 15, 16]
    assert [x["role"] for x in chosen] == ["calibration", "calibration", "validation"]
    assert rejected[0]["dataset_index"] == 13
    assert last_inspected == 16
    assert all("text" not in json.dumps(x) for x in chosen + rejected)


def test_selection_stops_after_third_eligible_and_never_consumes_future_identities():
    def proof(index, eligible):
        return {"dataset_index": index, "field_exists": True, "is_python_string": True,
                "character_count": 9000, "bounded_tokenizer_length": 8192 if eligible else 8191}

    rows = [proof(13, False), proof(14, True), proof(15, True), proof(16, True), proof(17, True), proof(18, False)]
    accepted, rejected, last_inspected = select_mechanically_inspected("narrative", reversed(rows))
    assert [row["dataset_index"] for row in accepted] == [14, 15, 16]
    assert [row["dataset_index"] for row in rejected] == [13]
    assert last_inspected == 16
    output_indices = {row["dataset_index"] for row in accepted + rejected}
    assert output_indices.isdisjoint({17, 18})


def test_selection_records_multiple_ineligible_before_third_eligible():
    def proof(index, eligible):
        return {"dataset_index": index, "field_exists": True, "is_python_string": True,
                "character_count": 9000, "bounded_tokenizer_length": 8192 if eligible else 8191}

    rows = [proof(13, False), proof(14, False), proof(15, True), proof(16, False), proof(17, True), proof(18, True)]
    accepted, rejected, last_inspected = select_mechanically_inspected("narrative", rows)
    assert [row["dataset_index"] for row in accepted] == [15, 17, 18]
    assert [row["dataset_index"] for row in rejected] == [13, 14, 16]
    assert last_inspected == 18


def test_deterministic_160_cell_optimizer_and_strict_budget_rule():
    group = {action: {"total_kv_bytes": 10.0 + i, "relative_l2": 5.0 - i, "cosine": .9 + i / 100} for i, action in enumerate(ACTIONS)}
    result = optimize_static_schedule([group] * 160, 14.0)
    assert result is not None and len(result[3]) == 160 and set(result[3]) == {"A4"}
    assert optimize_static_schedule([group] * 160, 10.0, strict=True) is None
    assert optimize_static_schedule([group] * 160, 14.0) == result


def test_phase_a_and_phase_b_runtime_closures_succeed_and_runner_is_fail_closed():
    assert len(verify_phase_a()) == 6
    assert len(verify_runtime_closure()) == 12
    with pytest.raises(PhaseBError, match="selected-source manifest"):
        runner_preflight()
