import hashlib
import json

import pytest

from cascadekv import c4_v2_bundle_postmortem as pm


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_bundle(tmp_path):
    actions = {a: {"kind": "flat_q8k4" if a == "A4" else "hierarchy"} for a in pm.ACTIONS}
    protocol = {"quality_gates": {"mean_cosine_gte": .985, "mean_relative_l2_lte": .12, "winner_relative_l2_lt": "uniform10", "winner_modeled_kv_traffic_lt": "flat5"}, "targets": {"order": list(pm.TARGETS)}, "methods": {"actions": actions}}
    runtime = {"protocol": {"path": "configs/cascadekv_phi35_8k_c4_protocol.json", "sha256": digest(protocol)}}
    stats = {f"{layer}:{head}:{action}": {"cosine": .99 - layer / 10000, "relative_l2": .1 + layer / 10000, "total_kv_bytes": 100 + head} for layer in pm.LAYERS for head in pm.HEADS for action in pm.ACTIONS}
    table = {f"{layer}:{head}": "A6" if layer == 0 else ("A4" if layer == 8 else "A0") for layer in pm.LAYERS for head in pm.HEADS}
    schedules = {t: {"table": table} for t in pm.TARGETS}
    dev_capture = {"kind": "development", "protocol_sha256": digest(protocol), "qualification_sha256": digest({}), "source_manifest_sha256": digest({}), "artifact_count": 45, "forwards": 9, "holdout_manifest_sha256": None, "development_result_sha256": None, "development_schedule_sha256": None}
    dev = {"protocol_sha256": digest(protocol), "runtime_manifest_sha256": digest(runtime), "development_source_manifest_sha256": digest({}), "capture_manifest_sha256": digest(dev_capture), "schedule_sha256": digest(schedules), "schedules": schedules, "action_cell_statistics": stats, "opened_development_tensor_identities": pm._identities("development"), "opened_holdout_tensor_identities": []}
    selected = [{"family": family, "dataset_index": index} for family, index in pm.HOLDOUT.items()]
    holdout = {"protocol_sha256": digest(protocol), "development_source_manifest_sha256": digest({}), "development_result_sha256": digest(dev), "development_schedule_sha256": dev["schedule_sha256"], "selected_sources": selected, "next_untouched_frontier": pm.FRONTIER, "raw_text_persisted": False}
    capture = {"kind": "holdout", "protocol_sha256": digest(protocol), "qualification_sha256": digest({}), "source_manifest_sha256": None, "holdout_manifest_sha256": digest(holdout), "development_result_sha256": digest(dev), "development_schedule_sha256": dev["schedule_sha256"], "artifact_count": 15, "forwards": 3}
    def metric(c, l2, kv): return {"mean_cosine": c, "mean_relative_l2": l2, "mean_total_kv_bytes": kv}
    baselines = {"dense": metric(1, 0, 1000), "flat5": metric(.94, .37, 500), "flat10": metric(.95, .28, 600), "uniform5": metric(.95, .27, 150), "uniform10": metric(.97, .22, 290)}
    targets = {t: metric(.98 + n / 1000, .18 - n / 100, 270 + n * 20) for n, t in enumerate(pm.TARGETS)}
    test = {"protocol_sha256": digest(protocol), "runtime_manifest_sha256": digest(runtime), "development_result_sha256": digest(dev), "development_schedule_sha256": dev["schedule_sha256"], "holdout_manifest_sha256": digest(holdout), "capture_manifest_sha256": digest(capture), "target_order": list(pm.TARGETS), "gate_values": protocol["quality_gates"], "opened_development_tensor_identities": [], "opened_holdout_tensor_identities": pm._identities("holdout", holdout), "no_schedule_mutation_evidence": {"schedule_sha256_before": dev["schedule_sha256"], "schedule_sha256_after": dev["schedule_sha256"], "schedule_unchanged": True, "optimizer_rerun": False}, "first_passing_target": None, "baseline_test_metrics": baselines, "target_test_metrics": targets}
    data = {"qualification.json": {}, "cascadekv_phi35_8k_dev_sources_v2.json": {}, "development_capture_manifest.json": dev_capture, "development.json": dev, "holdout.json": holdout, "holdout_capture_manifest.json": capture, "test.json": test, "c4_protocol.json": protocol, "c4_runtime_manifest.json": runtime}
    for name, value in data.items(): (tmp_path / name).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    refresh(tmp_path)
    return {name: pm._sha(tmp_path / name) for name in pm.REQUIRED}


def refresh(root):
    (root / "SHA256SUMS").write_text("".join(f"{pm._sha(root / name)}  {name}\n" for name in pm.REQUIRED))


def test_valid_bundle_gate_math_aggregation_and_deterministic_report(tmp_path):
    expected = write_bundle(tmp_path)
    result = pm.postmortem(pm.validate_bundle(tmp_path, expected=expected))
    assert result["closest_target"]["target"] == "T5"
    assert result["gate_margins"]["T5"]["margins"]["relative_l2_gate_minus_value"] == pytest.approx(-.01)
    assert result["t5_development_layers"]["0"]["action_histogram"]["A6"] == 32
    assert result["t5_development_layers"]["8"]["action_histogram"]["A4"] == 32
    assert pm.markdown(result) == pm.markdown(result)


def test_bad_sha_and_missing_file_rejected(tmp_path):
    expected = write_bundle(tmp_path)
    (tmp_path / "test.json").write_text("{}")
    with pytest.raises(pm.BundleError, match="SHA256SUMS mismatch"):
        pm.validate_bundle(tmp_path, expected=expected)
    (tmp_path / "test.json").unlink()
    with pytest.raises(pm.BundleError, match="required file"):
        pm.validate_bundle(tmp_path, expected=expected)


@pytest.mark.parametrize("mutation, message", [
    (lambda p: p["development.json"].__setitem__("schedule_sha256", "0" * 64), "schedule digest"),
    (lambda p: p["holdout.json"].__setitem__("development_result_sha256", "0" * 64), "holdout/development"),
    (lambda p: p["test.json"]["no_schedule_mutation_evidence"].__setitem__("optimizer_rerun", True), "no-schedule"),
    (lambda p: p["test.json"]["no_schedule_mutation_evidence"].__setitem__("schedule_sha256_after", "0" * 64), "no-schedule"),
    (lambda p: p["test.json"].__setitem__("opened_development_tensor_identities", ["bad"]), "tensor firewall"),
    (lambda p: p["test.json"].__setitem__("opened_holdout_tensor_identities", []), "tensor firewall"),
])
def test_binding_and_firewall_fail_closed(tmp_path, mutation, message):
    expected = write_bundle(tmp_path)
    payloads = {name: json.loads((tmp_path / name).read_text()) for name in pm.REQUIRED}
    mutation(payloads)
    for name, value in payloads.items(): (tmp_path / name).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    refresh(tmp_path)
    expected = {name: pm._sha(tmp_path / name) for name in pm.REQUIRED}
    with pytest.raises(pm.BundleError, match=message):
        pm.validate_bundle(tmp_path, expected=expected)
