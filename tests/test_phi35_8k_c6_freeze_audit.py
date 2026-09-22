"""Pure, local freeze-boundary matrices for C6."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cascadekv import phi35_8k_c6 as c6


def _set(value, path, replacement):
    target = value
    for key in path[:-1]: target = target[key]
    target[path[-1]] = replacement


@pytest.mark.parametrize("path", [
    ("schema_version",), ("c5_result_parent", "recovered_result_tag"), ("c5_result_parent", "closure_commit"),
    ("c5_result_parent", "freeze_tag"), ("c5_result_parent", "freeze_commit"), ("c5_result_parent", "protocol_sha256"),
    ("c5_result_parent", "runtime_sha256"), ("c5_result_parent", "recovered_result_sha256"),
    ("c5_result_parent", "recovered_postmortem_sha256"), ("c5_result_parent", "stdout_reconstruction_is_not_original_kaggle_container"),
    ("geometry", "context_length"), ("geometry", "layers"), ("geometry", "query_positions"), ("geometry", "heads"),
    ("geometry", "head_dim"), ("geometry", "schedule_cells"), ("geometry", "observations_per_source_per_method"),
    ("geometry", "development_observations_per_method"), ("geometry", "holdout_observations_per_method"),
    ("methods", "actions"), ("development_sources", "identities"), ("development_sources", "report20_excluded"),
    ("development_sources", "source_text_stored"), ("prospective_frontier", "start_indices"),
    ("prospective_frontier", "untouched_during_preparation"), ("targets", "fractions"), ("quality_gates", "mean_cosine_gte"),
    ("quality_gates", "mean_relative_l2_lte"), ("quality_gates", "winner_relative_l2_lt"),
    ("quality_gates", "winner_modeled_kv_traffic_lt"), ("optimizer", "version"), ("optimizer", "scope"),
    ("optimizer", "objective"), ("optimizer", "action_change_rule"), ("optimizer", "tie_breaking"), ("optimizer", "ordering"),
    ("development_gate", "no_go_classification"), ("development_gate", "go_requires_freeze_before_selection"),
    ("holdout_selection", "rule"), ("firewall", "test_optimizer_rerun"), ("firewall", "test_opens_development_tensors"),
    ("firewall", "schedule_sha256_unchanged"), ("lifecycle", "development_go"), ("lifecycle", "development_no_go"),
    ("lifecycle", "freeze_before_prospective"), ("lifecycle", "selected_target_locks_prospective_verdict"),
    ("raw_text_persisted",), ("scientific_state", "c6_metrics_observed"), ("action_validity", "enforcement"),
])
def test_protocol_single_field_mutation_rejected(path):
    value = c6.validate_protocol()
    old = path[-1]
    _set(value, path, "MUTATED" if old not in {"layers", "query_positions", "identities", "actions", "fractions", "start_indices"} else [])
    with pytest.raises(c6.C6Error): c6.validate_protocol(value)


def test_protocol_missing_and_extra_keys_rejected():
    value = c6.validate_protocol(); value.pop("firewall")
    with pytest.raises(c6.C6Error): c6.validate_protocol(value)
    value = c6.validate_protocol(); value["unexpected"] = True
    with pytest.raises(c6.C6Error): c6.validate_protocol(value)


def test_selected_target_lock_verdict_matrix():
    assert c6.prospective_verdict("T0", {"T0": {"passes": False}, "T1": {"passes": True}}) == "C6-PROSPECTIVE-NO-PASS"
    assert c6.prospective_verdict("T0", {"T0": {"passes": True}, "T1": {"passes": False}}) == "C6-PROSPECTIVE-PASS"
    with pytest.raises(c6.C6Error): c6.prospective_verdict("T0", {"T1": {"passes": True}})


def test_preflight_tag_matrix(monkeypatch):
    monkeypatch.setattr(c6, "verify_runtime_closure", lambda: ())
    monkeypatch.setattr(c6, "validate_protocol", lambda: {})
    monkeypatch.setattr(c6, "sha256_path", lambda path: c6.C5_PARENT["recovered_result_sha256"] if "recovered_result" in str(path) else c6.C5_PARENT["recovered_postmortem_sha256"])
    commits = {c6.C5_PARENT["recovered_result_tag"]: c6.C5_PARENT["closure_commit"], "HEAD": "h", c6.C6_TAG: "h"}
    def git(ref):
        if ref not in commits: raise c6.C6Error("missing test ref")
        return commits[ref]
    monkeypatch.setattr(c6, "_git", git)
    assert c6.preflight(execution=False)["local_only"]
    assert not c6.preflight(execution=True)["local_only"]
    commits.pop(c6.C6_TAG)
    with pytest.raises(c6.C6Error): c6.preflight(execution=True)
    commits[c6.C6_TAG] = "other"
    with pytest.raises(c6.C6Error): c6.preflight(execution=True)
    commits[c6.C6_TAG] = "h"; commits[c6.C5_PARENT["recovered_result_tag"]] = "other"
    with pytest.raises(c6.C6Error): c6.preflight(execution=True)


def test_runtime_manifest_structural_mutations_rejected(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.json"; payload = json.loads(c6.RUNTIME.read_text()); runtime.write_text(json.dumps(payload))
    monkeypatch.setattr(c6, "RUNTIME", runtime)
    for change in ("duplicate", "absolute", "traversal", "missing", "malformed", "protocol_path", "protocol_sha"):
        changed = copy.deepcopy(payload)
        if change == "duplicate": changed["bound_files"].append(copy.deepcopy(changed["bound_files"][0]))
        elif change == "absolute": changed["bound_files"][0]["path"] = "/tmp/x"
        elif change == "traversal": changed["bound_files"][0]["path"] = "../x"
        elif change == "missing": changed["bound_files"][0]["path"] = "missing.py"
        elif change == "malformed": changed["bound_files"][0]["sha256"] = "bad"
        elif change == "protocol_path": changed["protocol"]["path"] = "wrong.json"
        else: changed["protocol"]["sha256"] = "0" * 64
        runtime.write_text(json.dumps(changed))
        with pytest.raises(c6.C6Error): c6.verify_runtime_closure()
