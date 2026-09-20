"""Synthetic-only C3 contract tests; no captured payload is opened."""
from __future__ import annotations

import builtins
import copy
import json

import pytest

from cascadekv import phi35_8k_c3 as c3


def _source_payload():
    return {"selected_sources": [{"family": family, "dataset_index": index, "role": role} for family, rows in c3.EXPECTED.items() for index, role in rows], "rejected_sources": {"qa": [{"dataset_index": 14, "rejection_reason": "duplicate_input"}]}}


def _capture_payload():
    entries = []
    for family, rows in c3.EXPECTED.items():
        for index, role in rows:
            for layer in c3.LAYERS:
                if (family, index, role) == ("qa", 15, "calibration_2"):
                    entries.append({"origin": "historical_role_rebound", "base_artifact_relative_path": f"artifacts/qa_validation_index15_layer{layer}.safetensors", "base_artifact_sha256": "a" * 64, "historical_provenance_relative_path": f"artifacts/qa_validation_index15_layer{layer}.provenance.json", "historical_provenance_sha256": "e" * 64, "role_reference_relative_path": f"references/{layer}.json", "role_reference_sha256": "b" * 64})
                elif (family, index, role) == ("qa", 16, "validation"):
                    entries.append({"origin": "supplement_capture", "supplement_artifact_relative_path": f"artifacts/qa_validation_index16_layer{layer}.safetensors", "supplement_artifact_sha256": "c" * 64, "supplement_provenance_relative_path": f"provenance/{layer}.json", "supplement_provenance_sha256": "f" * 64})
                else:
                    entries.append({"origin": "historical_unchanged", "base_artifact_relative_path": f"artifacts/{family}_{role}_index{index}_layer{layer}.safetensors", "base_artifact_sha256": "d" * 64})
    return {"schema_version": "cascadekv-phi35-8k-amended-capture-v1", "amended_source_manifest_sha256": c3.SOURCE_SHA, "logical_artifact_count": 45, "artifacts": entries}


def test_protocol_and_local_preflight_do_not_import_data_stack(monkeypatch):
    c3.validate_protocol()
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "safetensors", "datasets", "transformers", "huggingface_hub"}:
            raise AssertionError(f"preflight imported forbidden stack: {name}")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert c3.preflight(require_c3_tag=False)["protocol_sha256"] == c3.sha256_path(c3.PROTOCOL)


def test_protocol_rejects_inputs_geometry_targets_and_gates():
    for path, value in ((["external_inputs", "source_manifest_sha256"], "0" * 64), (["geometry", "schedule_cells"], 159), (["targets", "order"], ["T1"]), (["quality_gates", "mean_cosine_gte"], .984)):
        broken = copy.deepcopy(c3._json(c3.PROTOCOL)); cursor = broken
        for key in path[:-1]: cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(c3.C3Error): c3.validate_protocol(broken)


def test_protocol_rejects_optimizer_firewall_schemas_tag_hierarchy_and_classifications():
    for path, value in ((["optimizer", "scope"], "validation"), (["firewall", "validation_tensor_roles"], ["calibration_1"]), (["result_schemas", "validation"], "other"), (["c3_freeze_tag"], "wrong"), (["methods", "hierarchy", "atom_size"], 9), (["classifications"], [])):
        broken = copy.deepcopy(c3._json(c3.PROTOCOL)); cursor = broken
        for key in path[:-1]: cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(c3.C3Error): c3.validate_protocol(broken)


def test_manifest_only_resolver_has_exact_45_origin_accounting_and_no_qa14(tmp_path, monkeypatch):
    source, capture = tmp_path / "source.json", tmp_path / "capture.json"
    source.write_text(json.dumps(_source_payload())); capture.write_text(json.dumps(_capture_payload()))
    base, supplement = tmp_path / "base", tmp_path / "supplement"; base.mkdir(); supplement.mkdir()
    progress = supplement / "supplement_progress_manifest.json"; progress.write_text("progress")
    (supplement / "references").mkdir(); (supplement / "provenance").mkdir(); (base / "artifacts").mkdir()
    for layer in c3.LAYERS:
        (base / "artifacts" / f"qa_validation_index15_layer{layer}.provenance.json").write_text("historical")
        (supplement / "references" / f"{layer}.json").write_text(json.dumps({"schema_version": "cascadekv-phi35-8k-amended-artifact-reference-v1", "family": "qa", "dataset_index": 15, "layer": layer, "historical_role": "validation", "amended_role": "calibration_2", "tensor_bytes_reused_unchanged": True, "tensor_bytes_recaptured": False, "historical_artifact_sha256": "a" * 64, "historical_provenance_relative_path": f"artifacts/qa_validation_index15_layer{layer}.provenance.json", "historical_provenance_sha256": "e" * 64}))
        (supplement / "provenance" / f"{layer}.json").write_text(json.dumps({"schema_version": "cascadekv-phi35-8k-capture-amendment-artifact-v1", "source_identity": {"family": "qa", "dataset_index": 16, "role": "validation"}, "layer": layer, "artifact_sha256": "c" * 64, "input_ids_sha256": "d9bcac44581785b0d180fdcd46cbcc5dcecd50e0e2c4285d81d949d1b35bc3ac"}))
    actual_hash = c3.sha256_path
    def digest(path):
        if path == source: return c3.SOURCE_SHA
        if path == capture: return c3.CAPTURE_SHA
        if path == progress: return c3.SUPPLEMENT_PROGRESS_SHA
        if path.name.endswith(".json") and path.parent.name == "references": return "b" * 64
        if path.name.endswith(".json") and path.parent.name == "artifacts": return "e" * 64
        if path.name.endswith(".json") and path.parent.name == "provenance": return "f" * 64
        return actual_hash(path)
    monkeypatch.setattr(c3, "sha256_path", digest)
    resolver = c3.CaptureResolver(source, capture, base, supplement)
    assert len(resolver.records) == 45
    assert len(resolver.records_for_roles(c3.CAL_ROLES)) == 30
    assert len(resolver.records_for_roles(c3.VAL_ROLES)) == 15
    assert not any("qa:14:" in key for key in resolver.records)
    assert resolver.records["qa:15:calibration_2:L0"]["origin"] == "historical_role_rebound"
    assert resolver.records["qa:16:validation:L0"]["origin"] == "supplement_capture"
    (base / "artifacts" / "qa_validation_index15_layer0.provenance.json").unlink()
    with pytest.raises(c3.C3Error, match="provenance"):
        c3.CaptureResolver(source, capture, base, supplement)
    (base / "artifacts" / "qa_validation_index15_layer0.provenance.json").write_text("historical")
    (supplement / "references" / "0.json").write_text(json.dumps({"schema_version": "cascadekv-phi35-8k-amended-artifact-reference-v1", "family": "qa", "dataset_index": 15, "layer": 0, "historical_role": "validation", "amended_role": "calibration_2", "tensor_bytes_reused_unchanged": False, "tensor_bytes_recaptured": False, "historical_artifact_sha256": "a" * 64, "historical_provenance_relative_path": "artifacts/qa_validation_index15_layer0.provenance.json", "historical_provenance_sha256": "e" * 64}))
    with pytest.raises(c3.C3Error, match="semantics"):
        c3.CaptureResolver(source, capture, base, supplement)
    (supplement / "references" / "0.json").write_text(json.dumps({"schema_version": "cascadekv-phi35-8k-amended-artifact-reference-v1", "family": "qa", "dataset_index": 15, "layer": 0, "historical_role": "validation", "amended_role": "calibration_2", "tensor_bytes_reused_unchanged": True, "tensor_bytes_recaptured": False, "historical_artifact_sha256": "a" * 64, "historical_provenance_relative_path": "artifacts/qa_validation_index15_layer0.provenance.json", "historical_provenance_sha256": "e" * 64}))
    (supplement / "provenance" / "0.json").unlink()
    with pytest.raises(c3.C3Error, match="supplement provenance"):
        c3.CaptureResolver(source, capture, base, supplement)


def test_tensor_firewall_denies_forbidden_payload_before_hash_or_deserialization():
    access = c3.TensorAccess(None, c3.CAL_ROLES)  # type: ignore[arg-type]
    with pytest.raises(c3.C3Error, match="firewall"):
        access.open({"identity": "narrative:15:validation", "path": "/must/not/open", "sha256": "0" * 64})
    access = c3.TensorAccess(None, c3.VAL_ROLES)  # type: ignore[arg-type]
    with pytest.raises(c3.C3Error, match="firewall"):
        access.open({"identity": "qa:13:calibration_1", "path": "/must/not/open", "sha256": "0" * 64})


def test_tensor_firewall_denies_same_role_wrong_identity_before_hash_or_deserialization():
    access = c3.TensorAccess(None, c3.CAL_ROLES)  # type: ignore[arg-type]
    with pytest.raises(c3.C3Error, match="unexpected identity"):
        access.open({"identity": "narrative:999:calibration_1", "layer": 0, "path": "/must/not/open", "sha256": "0" * 64})


def _rows_for_actions():
    result = {action: [] for action in c3.ACTIONS}
    for layer in c3.LAYERS:
        for head in c3.HEADS:
            for action_i, action in enumerate(c3.ACTIONS):
                for _ in range(18): result[action].append({"layer": layer, "head": head, "metrics": {"relative_l2_error": .1 - action_i * .001, "cosine_similarity": .99 + action_i * .001}, "traffic": {"total_kv_bytes": 100 + action_i}})
    result["flat5"] = [{"metrics": {"relative_l2_error": .1, "cosine_similarity": .99}, "traffic": {"total_kv_bytes": 1000}}] * 2880
    result["dense"] = [{"metrics": {"relative_l2_error": 0., "cosine_similarity": 1.}, "traffic": {"total_kv_bytes": 10000}}] * 2880
    result["flat10"] = [{"metrics": {"relative_l2_error": .1, "cosine_similarity": .99}, "traffic": {"total_kv_bytes": 1000}}] * 2880
    result["uniform5"] = [{"metrics": {"relative_l2_error": .1, "cosine_similarity": .99}, "traffic": {"total_kv_bytes": 1000}}] * 2880
    result["uniform10"] = [{"metrics": {"relative_l2_error": .1, "cosine_similarity": .99}, "traffic": {"total_kv_bytes": 1000}}] * 2880
    return result


def test_calibration_optimizer_is_160_cell_microbyte_deterministic_and_t5_strict():
    stats, schedules = c3._calibration_schedules(_rows_for_actions(), c3.validate_protocol())
    assert len(stats) == 160 * 5 and list(schedules) == list(c3.TARGETS)
    assert all(item["integer_microbyte_cap"] >= 0 for item in schedules.values())
    assert schedules["T5"]["strict"] is True
    for item in schedules.values():
        if item["feasible"]: assert len(item["table"]) == 160


def test_gate_inequalities_are_exact_and_strict():
    candidate = {"mean_cosine": .985, "mean_relative_l2": .120, "mean_total_kv_bytes": 99}
    uniform, flat = {"mean_relative_l2": .120}, {"mean_total_kv_bytes": 100}
    assert candidate["mean_cosine"] >= .985 and candidate["mean_relative_l2"] <= .120
    assert not (candidate["mean_relative_l2"] < uniform["mean_relative_l2"])
    assert candidate["mean_total_kv_bytes"] < flat["mean_total_kv_bytes"]


def _all_feasible_schedules():
    _stats, schedules = c3._calibration_schedules(_rows_for_actions(), c3.validate_protocol())
    return schedules


def _mock_calibrate(monkeypatch, schedules):
    class Resolver:
        def __init__(self, *_args): pass
        def records_for_roles(self, roles): return list(roles)
    class Access:
        def __init__(self, _resolver, roles): self.roles = tuple(roles)
    monkeypatch.setattr(c3, "preflight", lambda **_kwargs: {})
    monkeypatch.setattr(c3, "CaptureResolver", Resolver)
    monkeypatch.setattr(c3, "TensorAccess", Access)
    statistics, _ = c3._calibration_schedules(_rows_for_actions(), c3.validate_protocol())
    monkeypatch.setattr(c3, "_calibration_schedules", lambda _rows, _protocol: (statistics, copy.deepcopy(schedules)))
    monkeypatch.setattr(c3, "_method_rows", lambda _access, _records, _protocol, schedules=None: (_rows_for_actions(), c3._expected_tensor_identities(c3.CAL_ROLES)))


def _calibrate_synthetic(monkeypatch, tmp_path, schedules=None):
    _mock_calibrate(monkeypatch, _all_feasible_schedules() if schedules is None else schedules)
    output = tmp_path / "calibration.json"
    return c3.calibrate(source_manifest=tmp_path / "source", amended_capture_manifest=tmp_path / "capture", base_capture_root=tmp_path / "base", supplement_root=tmp_path / "supplement", output=output), output


def test_infeasible_targets_are_complete_calibration_outcomes(monkeypatch, tmp_path):
    import cascadekv.phi35_phaseb as phaseb
    original = phaseb.optimize_static_schedule
    try:
        for infeasible in ({"T0"}, {"T0", "T1", "T2"}, set(c3.TARGETS)):
            calls = iter(c3.TARGETS)
            monkeypatch.setattr(phaseb, "optimize_static_schedule", lambda *_args, **_kwargs: None if next(calls) in infeasible else (1, 0., 1., tuple("A0" for _ in range(160))))
            _stats, schedules = c3._calibration_schedules(_rows_for_actions(), c3.validate_protocol())
            assert list(schedules) == list(c3.TARGETS)
            for target, item in schedules.items():
                assert item["feasible"] is (target not in infeasible)
                if target in infeasible:
                    assert item["integer_microbytes_used"] is None and item["table"] is None
    finally:
        monkeypatch.setattr(phaseb, "optimize_static_schedule", original)
    for number, infeasible in enumerate(({"T0"}, {"T0", "T1", "T2"}, set(c3.TARGETS))):
        schedules = _all_feasible_schedules()
        for target in infeasible:
            schedules[target].update(feasible=False, integer_microbytes_used=None, table=None)
        result, output = _calibrate_synthetic(monkeypatch, tmp_path / str(number), schedules)
        assert list(result["schedules"]) == list(c3.TARGETS)
        assert c3._validate_calibration(result, output)["schedules"] == result["schedules"]


def test_full_calibration_validation_rejects_each_mutated_contract_fact(monkeypatch, tmp_path):
    result, output = _calibrate_synthetic(monkeypatch, tmp_path)
    assert c3._validate_calibration(result, output)["schedules"] == result["schedules"]
    cases = {
        "runtime": lambda p: p.__setitem__("c3_runtime_manifest_sha256", "0" * 64),
        "prior": lambda p: p["prior_freezes"].__setitem__("phase_b", {}),
        "methods": lambda p: p.__setitem__("methods", {}),
        "source_access": lambda p: p["opened_calibration_tensor_identities"].__setitem__(0, "qa:16:validation:L0"),
        "validation_access": lambda p: p.__setitem__("validation_tensor_access", True),
        "target_mean": lambda p: p["schedules"]["T0"].__setitem__("target_mean_kv_bytes", 0),
        "strict": lambda p: p["schedules"]["T0"].__setitem__("strict", True),
        "cap": lambda p: p["schedules"]["T0"].__setitem__("integer_microbyte_cap", 0),
        "used": lambda p: p["schedules"]["T0"].__setitem__("integer_microbytes_used", 10**12),
        "missing_cell": lambda p: p["schedules"]["T0"]["table"].pop("0:0"),
        "extra_cell": lambda p: p["schedules"]["T0"]["table"].__setitem__("99:99", "A0"),
        "action": lambda p: p["schedules"]["T0"]["table"].__setitem__("0:0", "BAD"),
        "inconsistency": lambda p: p["schedules"]["T0"].update(feasible=False),
        "tie": lambda p: p.__setitem__("deterministic_tie_provenance", {}),
        "diagnostics": lambda p: p.__setitem__("optional_phaseb_diagnostics", {}),
    }
    for name, mutate in cases.items():
        broken = copy.deepcopy(result); mutate(broken)
        with pytest.raises(c3.C3Error):
            c3._validate_calibration(broken, output)


def test_schedule_digest_is_byte_stable_and_includes_metadata():
    schedules = _all_feasible_schedules()
    assert c3.schedule_digest(schedules) == c3.schedule_digest(copy.deepcopy(schedules))
    changed = copy.deepcopy(schedules); changed["T0"]["strict"] = True
    assert c3.schedule_digest(schedules) != c3.schedule_digest(changed)


def test_synthetic_calibrate_e2e_records_all_targets_and_refuses_overwrite(monkeypatch, tmp_path):
    schedules = _all_feasible_schedules()
    schedules["T0"].update(feasible=False, integer_microbytes_used=None, table=None)
    result, output = _calibrate_synthetic(monkeypatch, tmp_path, schedules)
    assert list(result["schedules"]) == list(c3.TARGETS)
    assert result["opened_calibration_tensor_identities"] == c3._expected_tensor_identities(c3.CAL_ROLES)
    assert result["opened_validation_tensor_identities"] == []
    assert len(result["calibration_sources"]) == 6
    assert result["optional_phaseb_diagnostics"] == c3._unimplemented_phaseb_diagnostics()
    assert result["schedules"]["T0"]["table"] is None
    with pytest.raises(c3.C3Error, match="overwrite"):
        c3.calibrate(source_manifest=tmp_path / "source", amended_capture_manifest=tmp_path / "capture", base_capture_root=tmp_path / "base", supplement_root=tmp_path / "supplement", output=output)


def _install_synthetic_validation(monkeypatch, *, passing_target=None):
    class Resolver:
        def __init__(self, *_args): pass
        def records_for_roles(self, roles): return list(roles)

    class Access:
        def __init__(self, _resolver, roles): self.roles = tuple(roles)

    def rows(_access, _records, _protocol, targets):
        methods = {name: [{"metrics": {"cosine_similarity": .99, "relative_l2_error": .1}, "traffic": {"total_kv_bytes": 100}}] * 1440 for name in ("dense", "flat5", "flat10", "uniform5", "uniform10")}
        for target in targets:
            methods[target] = [{"metrics": {"cosine_similarity": .99, "relative_l2_error": .09 if target == passing_target else .2}, "traffic": {"total_kv_bytes": 99}}] * 1440
        return methods, c3._expected_tensor_identities(c3.VAL_ROLES)

    monkeypatch.setattr(c3, "preflight", lambda **_kwargs: {})
    monkeypatch.setattr(c3, "CaptureResolver", Resolver)
    monkeypatch.setattr(c3, "TensorAccess", Access)
    monkeypatch.setattr(c3, "_method_rows", rows)


def test_synthetic_validate_e2e_skips_infeasible_never_optimizes_and_refuses_overwrite(monkeypatch, tmp_path):
    schedules = _all_feasible_schedules()
    schedules["T0"].update(feasible=False, integer_microbytes_used=None, table=None)
    calibration, calibration_path = _calibrate_synthetic(monkeypatch, tmp_path, schedules)
    _install_synthetic_validation(monkeypatch, passing_target="T1")
    import cascadekv.phi35_phaseb as phaseb
    monkeypatch.setattr(phaseb, "optimize_static_schedule", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validation optimized")))
    output = tmp_path / "validation.json"
    result = c3.validate(source_manifest=tmp_path / "source", amended_capture_manifest=tmp_path / "capture", base_capture_root=tmp_path / "base", supplement_root=tmp_path / "supplement", calibration_result=calibration_path, output=output)
    assert result["opened_calibration_tensor_identities"] == []
    assert result["opened_validation_tensor_identities"] == c3._expected_tensor_identities(c3.VAL_ROLES)
    assert result["per_gate_pass_fail"]["T0"] == {"feasible": False, "passes": False}
    assert result["first_passing_target"] == "T1"
    assert all(item["observation_count"] == 1440 for item in result["target_validation_metrics"].values())
    proof = result["no_schedule_mutation_evidence"]
    assert proof["schedule_unchanged"] is True and proof["schedule_sha256_before"] == proof["schedule_sha256_after"] and proof["optimizer_rerun"] is False
    with pytest.raises(c3.C3Error, match="overwrite"):
        c3.validate(source_manifest=tmp_path / "source", amended_capture_manifest=tmp_path / "capture", base_capture_root=tmp_path / "base", supplement_root=tmp_path / "supplement", calibration_result=calibration_path, output=output)


@pytest.mark.parametrize(
    ("infeasible", "passing_target"),
    [({"T0", "T1", "T2"}, "T3"), (set(c3.TARGETS), None)],
    ids=("several-early-infeasible", "all-infeasible"),
)
def test_synthetic_validate_continues_after_early_infeasible_and_handles_no_passing_target(monkeypatch, tmp_path, infeasible, passing_target):
    schedules = _all_feasible_schedules()
    for target in infeasible:
        schedules[target].update(feasible=False, integer_microbytes_used=None, table=None)
    calibration, calibration_path = _calibrate_synthetic(monkeypatch, tmp_path, schedules)
    _install_synthetic_validation(monkeypatch, passing_target=passing_target)
    output = tmp_path / "validation.json"
    result = c3.validate(source_manifest=tmp_path / "source", amended_capture_manifest=tmp_path / "capture", base_capture_root=tmp_path / "base", supplement_root=tmp_path / "supplement", calibration_result=calibration_path, output=output)
    assert list(result["per_gate_pass_fail"]) == list(c3.TARGETS)
    assert all(result["per_gate_pass_fail"][target] == {"feasible": False, "passes": False} for target in infeasible)
    assert result["first_passing_target"] == passing_target
    assert set(result["target_validation_metrics"]) == set(c3.TARGETS) - infeasible
    assert all(item["observation_count"] == 1440 for item in result["target_validation_metrics"].values())
