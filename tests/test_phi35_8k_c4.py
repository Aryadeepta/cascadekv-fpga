"""Synthetic-only C4 contract tests: no HF, model, dataset, or payload access."""
from __future__ import annotations

import builtins
import copy
import hashlib
import io
import json
from itertools import product
from contextlib import redirect_stderr

import pytest

from cascadekv import phi35_8k_c4 as c4


def h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_exact_protocol_action_menu_c3_no_pass_and_local_preparation_state():
    protocol = c4.validate_protocol()
    assert list(protocol["methods"]["actions"]) == list(c4.ACTIONS)
    assert protocol["methods"]["actions"]["A5"]["candidate_fraction"] == .175
    assert protocol["methods"]["actions"]["A6"]["candidate_fraction"] == .20
    assert protocol["c3_observed_no_pass"]["first_passing_target"] is None
    assert protocol["c3_observed_no_pass"]["absolute_gates_passed"] is False
    assert all(value is False or value is True for value in protocol["scientific_state"].values())
    assert protocol["scientific_state"]["fresh_holdout_identities_selected"] is False


def test_protocol_rejects_new_tuning_or_a_c3_pass_claim():
    for path, value in ((["methods", "actions", "A5", "candidate_fraction"], .17), (["c3_observed_no_pass", "absolute_gates_passed"], True), (["holdout_selection", "start_indices", "qa"], 16), (["geometry", "development_observations_per_method"], 2880)):
        broken = copy.deepcopy(c4._json(c4.PROTOCOL)); cursor = broken
        for key in path[:-1]: cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(c4.C4Error): c4.validate_protocol(broken)


def _stat(traffic: float, error: float, cosine: float) -> dict[str, float]:
    return {"total_kv_bytes": traffic, "relative_l2": error, "cosine": cosine}


def test_seven_action_dp_tie_breaking_and_integer_strict_cap():
    # A6 has better error, but at the T5 strict cap it cannot be selected.
    groups = [{action: _stat(1.0, .1, .9) for action in c4.ACTIONS} for _ in range(160)]
    for group in groups:
        group["A0"] = _stat(1.0, .2, .9)
        group["A1"] = _stat(1.0, .2, .95)  # cosine tiebreak
        group["A2"] = _stat(1.0, .2, .95)  # lexical action table tiebreak
        group["A6"] = _stat(1.001, .01, 1.0)
    result = c4.optimize_static_schedule(groups, 1.0, strict=True)
    assert result is None  # 160*1000 cannot meet strict < 160000 microbytes
    result = c4.optimize_static_schedule(groups, 1.001, strict=True)
    assert result is not None and result[0] < 160160
    # With equal traffic/error, descending cosine picks A1 over A0/A2;
    # A1 is lexically before A2.
    result = c4.optimize_static_schedule([{a: _stat(1, 1, .5) for a in c4.ACTIONS} for _ in range(160)], 1.0)
    assert result is not None and result[3] == ("A0",) * 160


def _brute(groups, cap):
    feasible = []
    for table in product(c4.ACTIONS, repeat=len(groups)):
        used = sum(round(groups[cell][action]["total_kv_bytes"] * 1000) for cell, action in enumerate(table))
        if used <= cap:
            feasible.append((sum(groups[cell][action]["relative_l2"] for cell, action in enumerate(table)), -sum(groups[cell][action]["cosine"] for cell, action in enumerate(table)), used, tuple(c4.ACTIONS.index(action) for action in table), table))
    if not feasible:
        return None
    error, neg_cosine, used, _lexical, table = min(feasible)
    return used, error, -neg_cosine, table


@pytest.mark.parametrize("cells,cap", [(2, 2001), (3, 3000), (4, 4001)])
def test_generic_seven_action_dp_matches_exhaustive_small_fixtures(cells, cap):
    groups = []
    for cell in range(cells):
        groups.append({action: _stat(1.0004 + .0003 * index, .1 * (index + 1) + cell * .001, .90 + .001 * (6 - index)) for index, action in enumerate(c4.ACTIONS)})
    # Exercise a cosine tie, lexical table tie, and rounded microbyte traffic.
    groups[0]["A0"] = _stat(1.0004, .4, .9)
    groups[0]["A1"] = _stat(1.0004, .4, .95)
    groups[0]["A2"] = _stat(1.0004, .4, .95)
    actual, expected = c4._optimize_action_cells(groups, cap), _brute(groups, cap)
    assert actual is not None and expected is not None
    assert (actual[0], actual[3]) == (expected[0], expected[3])
    assert actual[1] == pytest.approx(expected[1])
    assert actual[2] == pytest.approx(expected[2])


def test_generic_dp_infeasible_and_a5_a6_dominated_matches_five_action_behavior():
    impossible = [{action: _stat(1, .1, .9) for action in c4.ACTIONS} for _ in range(2)]
    assert c4._optimize_action_cells(impossible, 1999) is None  # strict cap equivalent
    groups = [{action: _stat(1 + index, index + cell, .9 - index * .01) for index, action in enumerate(c4.ACTIONS)} for cell in range(3)]
    for group in groups:
        group["A5"] = _stat(99, 99, 0)
        group["A6"] = _stat(99, 99, 0)
    result = c4._optimize_action_cells(groups, 9000)
    five = min((sum(group[action]["relative_l2"] for group, action in zip(groups, table)), -sum(group[action]["cosine"] for group, action in zip(groups, table)), sum(round(group[action]["total_kv_bytes"] * 1000) for group, action in zip(groups, table)), table) for table in product(c4.ACTIONS[:5], repeat=3))
    assert result is not None and (result[1], -result[2], result[0], result[3]) == five


def test_target_cap_uses_development_flat5_and_strict_microbyte_rule():
    protocol = c4.validate_protocol()
    mean, strict, cap = c4._target_cap(protocol, "T5", 1000., 123.456)
    assert (mean, strict, cap) == (123.456, True, 19_752_959)
    assert c4._target_cap(protocol, "T0", 1000., 123.456) == (115., False, 18_400_000)


def _valid_development_result():
    protocol = c4.validate_protocol(); table = {key: "A0" for key in c4._action_table_keys()}
    traffic = {name: (1.0 if name in c4.ACTIONS else 10.0) for name in (*c4.BASELINES, *c4.ACTIONS)}
    metrics = {name: {"observation_count": 4320, "mean_cosine": .99, "mean_relative_l2": .1, "mean_total_kv_bytes": value} for name, value in traffic.items()}
    stats = {f"{layer}:{head}:{action}": _stat(1.0, .1, .99) for layer in c4.LAYERS for head in c4.HEADS for action in c4.ACTIONS}
    schedules = {}
    for target in c4.TARGETS:
        mean, strict, cap = c4._target_cap(protocol, target, 10.0, 10.0)
        schedules[target] = {"feasible": True, "target_mean_kv_bytes": mean, "strict": strict, "integer_microbyte_cap": cap, "integer_microbytes_used": 160_000, "table": table}
    return {"schema_version": "cascadekv-phi35-8k-c4-development-v1", "protocol_sha256": c4.sha256_path(c4.PROTOCOL), "runtime_manifest_sha256": c4.sha256_path(c4.RUNTIME), "development_source_manifest_sha256": c4.SOURCE_SHA, "capture_manifest_sha256": h("capture"), "opened_development_tensor_identities": c4._expected_tensor_ids("development"), "opened_holdout_tensor_identities": [], "methods": protocol["methods"], "development_metrics": metrics, "development_traffic": traffic, "action_cell_statistics": stats, "schedules": schedules, "schedule_sha256": c4.schedule_digest(schedules), "optimizer": protocol["optimizer"], "classification": None}


def test_development_result_validator_checks_finite_stats_and_rounded_table_traffic(monkeypatch):
    # Runtime closure is intentionally irrelevant to this persisted-result unit test.
    result = _valid_development_result()
    assert c4._validate_development_result(result)["schedule_sha256"] == result["schedule_sha256"]
    broken = copy.deepcopy(result); broken["action_cell_statistics"]["0:0:A0"]["total_kv_bytes"] = float("nan")
    with pytest.raises(c4.C4Error): c4._validate_development_result(broken)
    broken = copy.deepcopy(result); broken["schedules"]["T0"]["integer_microbytes_used"] += 1; broken["schedule_sha256"] = c4.schedule_digest(broken["schedules"])
    with pytest.raises(c4.C4Error, match="traffic"):
        c4._validate_development_result(broken)


def _inspection(family: str, index: int, digest: str, *, eligible: bool = True) -> dict:
    proof = {"field_exists": eligible, "is_python_string": eligible, "at_least_8192": eligible}
    return {"dataset_index": index, "proof": proof, "input_ids_sha256": digest, "source": {"family": family, "dataset_index": index, "dataset": "d", "config": "c", "split": "s", "revision": "r", "field": "f", "identity_kind": "dataset_index"}}


def test_holdout_selector_starts_dedups_globally_stops_and_persists_no_text():
    dev = [h(f"dev{i}") for i in range(9)]
    rows = {
        "narrative": [_inspection("narrative", 16, dev[0]), _inspection("narrative", 17, h("n")), _inspection("narrative", 18, h("must-not-be-inspected"))],
        "report": [_inspection("report", 19, h("n")), _inspection("report", 20, h("r"))],
        "qa": [_inspection("qa", 17, h("q"))],
    }
    selected = c4.select_holdout_from_inspections(rows, dev)
    assert [(x["family"], x["dataset_index"]) for x in selected["selected_sources"]] == [("narrative", 17), ("report", 20), ("qa", 17)]
    assert selected["next_untouched_frontier"] == {"narrative": 18, "report": 21, "qa": 18}
    assert selected["raw_text_persisted"] is False
    assert "must-not-be-inspected" not in json.dumps(selected)
    assert selected["rejected_sources"]["narrative"][0]["reason"] == "duplicate_exact_input_ids_sha256"
    assert selected["rejected_sources"]["report"][0]["reason"] == "duplicate_exact_input_ids_sha256"


def test_select_holdout_requires_frozen_development_result_before_any_external_access(monkeypatch, tmp_path):
    development, source, output = tmp_path / "development.json", tmp_path / "sources.json", tmp_path / "holdout.json"
    development.write_text("{}")
    monkeypatch.setattr(c4, "preflight", lambda: {})
    monkeypatch.setattr(c4, "_validate_development_result", lambda _payload: (_ for _ in ()).throw(c4.C4Error("invalid frozen development result")))
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("torch") or name.startswith("datasets") or name.startswith("transformers"):
            raise AssertionError("external access")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(c4.C4Error, match="invalid frozen"):
        c4.select_holdout(development_source_manifest=source, development_result=development, output=output)
    assert "development_input_hashes" not in c4.select_holdout.__annotations__


def test_qa14_forbidden_and_development_inventory_is_nine_sources(tmp_path):
    payload = {"selected_sources": [{"family": family, "dataset_index": index} for family in c4.FAMILIES for index in c4.DEVELOPMENT[family]], "rejected_sources": {"qa": [{"dataset_index": 14}]}}
    with pytest.raises(c4.C4Error): c4._validate_development_manifest(payload)


def _canonical_corrected_manifest():
    sources = []
    for family in c4.FAMILIES:
        for position, index in enumerate(c4.DEVELOPMENT[family]):
            digest = c4.QA13_SHA if (family, index) == ("qa", 13) else c4.QA15_SHA if (family, index) == ("qa", 15) else h(f"{family}-{index}")
            eligibility = {"field_exists": True, "is_python_string": True, "character_count": 8192 + index, "bounded_frozen_tokenizer_length": 8192, "at_least_8192": True}
            proof = {"mechanical_eligibility": eligibility, "input_ids_length": 8192, "input_ids_sha256": digest}
            sources.append({"family": family, **{key: c4.SOURCE_SPECS[family][key] for key in ("dataset", "config", "split", "revision", "field")}, "identity_kind": "dataset_index", "dataset_index": index, "role": c4.ROLE_ORDER[position], "input_ids_sha256": digest, "proof": proof, "reproof": copy.deepcopy(proof)})
    qa13 = next(item for item in sources if item["family"] == "qa" and item["dataset_index"] == 13)
    return {
        "schema_version": "cascadekv-phi35-8k-dev-sources-v2",
        "purpose": "C1 source-identity amendment: exact model-input deduplication only; no quality evaluation",
        "amendment_reason": "C1 unique dataset indices did not ensure unique exact 8192-token model inputs",
        "base_c1_freeze": {"tag": "cascadekv-phi35-8k-sources-freeze", "commit": "667900a5307fe231565033a774d7788942fb869d", "manifest_sha256": "af8e805bad5c85a1f9d6ae5571b378e4df46fdb1de778a5ec7b5b68b13dc4a8e"},
        "phase_b_freeze": {"tag": "cascadekv-phi35-8k-phaseb-protocol-freeze", "commit": "d94cff8e1e11048ff3d8edc9fb1c605dfe422d6e", "protocol_sha256": "0e3a2e7b51e01c64dca292c7dd667ed9899c5bacc08e470e2b19e12433363c8b", "runtime_manifest_sha256": "e4e10f7659180471eb791eeafa97e8f525b6c14a3ff311798349c07438ec7c92"},
        "c2_v2_freeze": {"tag": "cascadekv-phi35-8k-kaggle-c2-prep-v2", "commit": "fa54b581ee5ee581dd51b8085db8b5233467450a", "protocol_sha256": "315453e8446e8f41611fc23544a4a5093da13a7dba344dfc66a840eebf17857d", "runtime_manifest_sha256": "a5a96915f8cab588e5d1dd6af15855d4d090366f94282022bff7bc1b856957fd"},
        "amendment_freeze": {"tag": c4.SOURCE_AMENDMENT["tag"], "commit": c4.SOURCE_AMENDMENT["commit"], "amendment_protocol_sha256": c4.SOURCE_AMENDMENT["protocol_sha256"], "amendment_runtime_manifest_sha256": c4.SOURCE_AMENDMENT["runtime_manifest_sha256"]},
        "target": {"model": c4.MODEL, "model_revision": c4.REVISION, "tokenizer_revision": c4.REVISION},
        "tokenization_contract": {"add_special_tokens": True, "truncation": True, "max_length": 8192, "required_input_ids_length": 8192, "canonicalization": "CPU contiguous int64 bytes"},
        "identity_kind": "dataset_index", "uniqueness_kind": "input_ids_sha256", "source_text_stored": False, "source_specs": copy.deepcopy(c4.SOURCE_SPECS), "selected_sources": sources,
        "rejected_sources": {"narrative": [], "report": [], "qa": [{"dataset_index": 14, "rejection_reason": "duplicate_input", "duplicate_of_dataset_index": 13, "input_ids_sha256": c4.QA13_SHA, "proof": copy.deepcopy(qa13["proof"])}]},
        "last_inspected_index": {family: c4.DEVELOPMENT[family][-1] for family in c4.FAMILIES},
        "unavailable_inventory": {family: {"base_c1_unavailable_indices": c4.BASE_UNAVAILABLE[family], "amendment_inspected_indices": list(c4.DEVELOPMENT[family]) if family != "qa" else [13, 14, 15, 16], "indices": list(range(16)) if family == "narrative" else list(range(19)) if family == "report" else list(range(17)), "next_untouched_index": c4.DEVELOPMENT[family][-1] + 1} for family in c4.FAMILIES},
        "reproof_status": {"all_selected_sources_equal": True, "selected_source_count": 9, "global_input_ids_sha256_unique": True},
    }


def test_c4_local_corrected_manifest_validator_is_cross_tag_safe_and_fail_closed(monkeypatch, tmp_path):
    payload = _canonical_corrected_manifest()
    commits = {c4.C3["tag"]: c4.C3["commit"], c4.SOURCE_AMENDMENT["tag"]: c4.SOURCE_AMENDMENT["commit"], c4.C4_TAG: "c4-v2-commit", "HEAD": "c4-v2-commit"}
    monkeypatch.setattr(c4, "_git_commit", lambda ref: commits[ref])
    monkeypatch.setattr(c4, "verify_runtime_closure", lambda **_kwargs: ())
    assert c4.preflight()["local_only"] is True
    assert commits[c4.SOURCE_AMENDMENT["tag"]] != commits["HEAD"]
    validated = c4._validate_development_manifest(payload)
    assert [item["role"] for item in validated] == ["development"] * 9
    assert all(item["canonical_role"] in c4.ROLE_ORDER for item in validated)
    for mutation in (
        lambda value: value["amendment_freeze"].update(tag="wrong-amendment-tag"),
        lambda value: value["amendment_freeze"].update(commit="0" * 40),
        lambda value: value["amendment_freeze"].update(amendment_protocol_sha256="0" * 64),
        lambda value: value["amendment_freeze"].update(amendment_runtime_manifest_sha256="0" * 64),
        lambda value: value["selected_sources"].__setitem__(6, {**value["selected_sources"][6], "dataset_index": 14}),
        lambda value: value["rejected_sources"].update(qa=[]),
        lambda value: value["rejected_sources"]["qa"][0].update(duplicate_of_dataset_index=15),
        lambda value: value["selected_sources"][1].update(input_ids_sha256=value["selected_sources"][0]["input_ids_sha256"]),
        lambda value: value["selected_sources"][0].update(reproof={}),
        lambda value: value["unavailable_inventory"]["qa"].update(next_untouched_index=18),
        lambda value: value["last_inspected_index"].update(qa=17),
    ):
        broken = copy.deepcopy(payload); mutation(broken)
        with pytest.raises(c4.C4Error): c4._validate_development_manifest(broken)
    source = tmp_path / "corrected.json"; source.write_text(json.dumps(payload))
    real_sha = c4.sha256_path
    monkeypatch.setattr(c4, "sha256_path", lambda path: "0" * 64 if path == source else real_sha(path))
    with pytest.raises(c4.C4Error, match="historical SHA"):
        c4._validate_development_manifest(payload, source)


def test_capture_development_reaches_capture_layer_under_c4_v2_head(monkeypatch, tmp_path):
    payload, source = _canonical_corrected_manifest(), tmp_path / "corrected.json"
    source.write_text(json.dumps(payload))
    real_sha = c4.sha256_path
    monkeypatch.setattr(c4, "preflight", lambda: {"local_only": True})
    monkeypatch.setattr(c4, "sha256_path", lambda path: c4.SOURCE_SHA if path == source else real_sha(path))
    reached = {}
    monkeypatch.setattr(c4, "_capture_sources", lambda **kwargs: reached.update(kwargs) or {"capture": "reached"})
    result = c4.capture_development(development_source_manifest=source, output=tmp_path / "capture", qualification=tmp_path / "qualification.json")
    assert result == {"capture": "reached"}
    assert len(reached["sources"]) == 9
    assert reached["kind"] == "development"


def _capture_manifest(kind: str, holdout=None, *, holdout_sha=None, development_sha=None, schedule_sha=None):
    ids = c4._expected_tensor_ids(kind, holdout)
    records = []
    for identity_layer in ids:
        identity, layer_token = identity_layer.rsplit(":L", 1)
        layer = int(layer_token)
        source = {"family": identity.split(":")[0], "dataset_index": int(identity.split(":")[1]), "role": identity.split(":")[2]}
        digest = next((row["input_ids_sha256"] for row in holdout["selected_sources"] if c4._source_identity(row, "holdout") == identity), h(f"i{identity}")) if holdout else h(f"i{identity}")
        records.append({"identity": identity, "layer": layer, "artifact_relative_path": f"artifacts/{len(records)}.safe", "artifact_sha256": h(f"a{len(records)}"), "provenance_relative_path": f"artifacts/{len(records)}.json", "provenance_sha256": h(f"p{len(records)}"), "input_ids_sha256": digest, "source": source})
    return {"schema_version": "cascadekv-phi35-8k-c4-capture-v1", "kind": kind, "protocol_sha256": c4.sha256_path(c4.PROTOCOL), "qualification_sha256": c4.QUALIFICATION_SHA, "backend_id": c4.BACKEND_ID, "source_manifest_sha256": c4.SOURCE_SHA if kind == "development" else None, "holdout_manifest_sha256": holdout_sha if kind == "holdout" else None, "development_result_sha256": development_sha if kind == "holdout" else None, "development_schedule_sha256": schedule_sha if kind == "holdout" else None, "artifact_count": len(records), "forwards": 9 if kind == "development" else 3, "artifacts": records}


def test_capture_inventory_firewall_and_path_traversal_precede_payload_open(tmp_path, monkeypatch):
    root, manifest = tmp_path / "capture", tmp_path / "manifest.json"; root.mkdir()
    manifest.write_text(json.dumps(_capture_manifest("development")))
    resolver = c4.CaptureResolver(root, manifest, "development")
    assert len(resolver.records) == 45
    with pytest.raises(c4.C4Error, match="capture kind"):
        c4.TensorAccess(resolver, allowed_kind="holdout")
    access = c4.TensorAccess(resolver, allowed_kind="development")
    record = dict(next(iter(resolver.records.values()))); record["artifact_relative_path"] = "../escape"
    with pytest.raises(c4.C4Error, match="path"):
        access.open(record)
    outside = root / "artifacts" / "other.safe"; outside.parent.mkdir(); outside.write_text("not a tensor")
    linked = root / "artifacts" / "linked.safe"; linked.parent.mkdir(exist_ok=True); linked.symlink_to(outside)
    record = dict(next(iter(resolver.records.values()))); record["artifact_relative_path"] = "artifacts/linked.safe"
    with pytest.raises(c4.C4Error, match="symlink"):
        access.open(record)
    malformed = _capture_manifest("development"); malformed["forwards"] = 8; manifest.write_text(json.dumps(malformed))
    with pytest.raises(c4.C4Error, match="capture manifest"):
        c4.CaptureResolver(root, manifest, "development")


def test_artifact_provenance_contract_rejects_backend_target_and_adapter_before_payload(tmp_path):
    root, manifest = tmp_path / "capture", tmp_path / "manifest.json"; root.mkdir(); (root / "artifacts").mkdir()
    capture = _capture_manifest("development"); record = capture["artifacts"][0]
    artifact, provenance = root / record["artifact_relative_path"], root / record["provenance_relative_path"]
    artifact.write_bytes(b"not-opened")
    record["artifact_sha256"] = c4.sha256_path(artifact)
    source = {"family": "narrative", "dataset": "d", "config": "c", "split": "s", "revision": "r", "field": "f", "identity_kind": "dataset_index", "dataset_index": 13, "role": "development"}
    detail = {"schema_version": "cascadekv-phi35-8k-c4-artifact-v1", "protocol_sha256": c4.sha256_path(c4.PROTOCOL), "qualification_sha256": c4.QUALIFICATION_SHA, "backend_id": "wrong", "identity": record["identity"], "layer": record["layer"], "input_ids_sha256": record["input_ids_sha256"], "target": {"model": c4.MODEL, "revision": c4.REVISION, "tokenizer_revision": c4.REVISION}, "source": source, "source_proof": {}, "q_shape": list(c4.SHAPE), "k_shape": list(c4.SHAPE), "v_shape": list(c4.SHAPE), "storage_dtype": "float16", "model_compute_dtype": "float16", "attention_implementation": "sdpa", "use_cache": False, "quantization": "none", "capture_adapter": "phi3-post-rope-qkv-v1", "artifact_sha256": record["artifact_sha256"]}
    provenance.write_text(json.dumps(detail)); record["provenance_sha256"] = c4.sha256_path(provenance)
    manifest.write_text(json.dumps(capture)); access = c4.TensorAccess(c4.CaptureResolver(root, manifest, "development"), allowed_kind="development")
    with pytest.raises(c4.C4Error, match="provenance contract"):
        access.open(record)
    assert access.opened == []


def test_holdout_capture_requires_actual_manifest_hash_and_per_source_hash(tmp_path):
    selected = [{"family": family, "dataset_index": c4.HOLDOUT_STARTS[family], "role": "holdout", "proof": {"field_exists": True, "is_python_string": True, "at_least_8192": True}, "input_ids_sha256": h(f"h-{family}")} for family in c4.FAMILIES]
    holdout = {"schema_version": "cascadekv-phi35-8k-c4-holdout-selection-v1", "protocol_sha256": c4.sha256_path(c4.PROTOCOL), "development_source_manifest_sha256": c4.SOURCE_SHA, "development_result_sha256": h("development"), "development_schedule_sha256": h("schedule"), "selection_rule": "first mechanically eligible globally-new exact input SHA in ascending order", "selected_sources": selected, "rejected_sources": {family: [] for family in c4.FAMILIES}, "next_untouched_frontier": {item["family"]: item["dataset_index"] + 1 for item in selected}, "raw_text_persisted": False}
    holdout_path, capture_path, root = tmp_path / "holdout.json", tmp_path / "capture.json", tmp_path / "root"
    holdout_path.write_text(json.dumps(holdout)); root.mkdir()
    capture = _capture_manifest("holdout", holdout, holdout_sha="0" * 64, development_sha=h("development"), schedule_sha=h("schedule")); capture_path.write_text(json.dumps(capture))
    with pytest.raises(c4.C4Error, match="holdout capture provenance"):
        c4.CaptureResolver(root, capture_path, "holdout", holdout_manifest=holdout_path)
    capture["holdout_manifest_sha256"] = c4.sha256_path(holdout_path)
    capture["artifacts"][0]["input_ids_sha256"] = h("wrong")
    capture_path.write_text(json.dumps(capture))
    with pytest.raises(c4.C4Error, match="source input"):
        c4.CaptureResolver(root, capture_path, "holdout", holdout_manifest=holdout_path)


def test_preflight_remains_local_and_progress_is_stderr(monkeypatch):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "safetensors", "datasets", "transformers", "huggingface_hub"}:
            raise AssertionError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(c4, "verify_runtime_closure", lambda **_kwargs: ())
    assert c4.preparation_preflight()["local_only"] is True
    stderr = io.StringIO()
    with redirect_stderr(stderr): c4._progress("one layer")
    assert stderr.getvalue() == "C4: one layer\n"


def test_runtime_closure_binds_source_amendment_provenance_files(monkeypatch, tmp_path):
    root, protocol, runtime = tmp_path / "repo", tmp_path / "repo/configs/protocol.json", tmp_path / "repo/configs/runtime.json"
    validator = root / "configs/cascadekv_phi35_8k_source_amendment_protocol.json"
    validator.parent.mkdir(parents=True); protocol.parent.mkdir(exist_ok=True)
    protocol.write_text("{}")
    validator.write_text("validator-v1\n")
    runtime.write_text(json.dumps({"schema_version": 1, "purpose": "test", "protocol": {"path": "configs/protocol.json", "sha256": c4.sha256_path(protocol)}, "bound_files": [{"path": "configs/cascadekv_phi35_8k_source_amendment_protocol.json", "sha256": c4.sha256_path(validator)}]}))
    monkeypatch.setattr(c4, "ROOT", root)
    monkeypatch.setattr(c4, "PROTOCOL", protocol)
    monkeypatch.setattr(c4, "RUNTIME", runtime)
    monkeypatch.setattr(c4, "_tracked", lambda _relative: True)
    assert c4.verify_runtime_closure() == ({"path": "configs/cascadekv_phi35_8k_source_amendment_protocol.json", "sha256": c4.sha256_path(validator)},)
    validator.write_text("validator-v2\n")
    with pytest.raises(c4.C4Error, match="runtime closure mismatch: configs/cascadekv_phi35_8k_source_amendment_protocol.json"):
        c4.verify_runtime_closure()


def test_strict_preflight_requires_the_final_c4_tag():
    with pytest.raises(c4.C4Error, match="required C4 freeze tag"):
        c4.preflight()


def test_final_preflight_defaults_to_the_c4_tag_and_tracking_gates(monkeypatch):
    calls = []
    def git_commit(ref):
        if ref == c4.C3["tag"]: return c4.C3["commit"]
        if ref == c4.SOURCE_AMENDMENT["tag"]: return c4.SOURCE_AMENDMENT["commit"]
        return "c4-head"
    monkeypatch.setattr(c4, "_git_commit", git_commit)
    monkeypatch.setattr(c4, "verify_runtime_closure", lambda *, require_tracked: calls.append(require_tracked) or ())
    assert c4.preflight()["local_only"] is True
    assert calls == [True]


def test_prospective_test_never_optimizes_or_opens_development(monkeypatch, tmp_path):
    selected = [{"family": family, "dataset_index": c4.HOLDOUT_STARTS[family], "role": "holdout", "proof": {"field_exists": True, "is_python_string": True, "at_least_8192": True}, "input_ids_sha256": h(f"holdout-{family}")} for family in c4.FAMILIES]
    holdout_path, capture_path, development_path, output = tmp_path / "holdout.json", tmp_path / "capture.json", tmp_path / "development.json", tmp_path / "test.json"
    development_path.write_text("{}")
    table = {key: "A0" for key in c4._action_table_keys()}
    schedules = {target: {"feasible": True, "target_mean_kv_bytes": 1., "strict": target == "T5", "integer_microbyte_cap": 999999, "integer_microbytes_used": 1, "table": table} for target in c4.TARGETS}
    development = {"schedules": schedules, "schedule_sha256": c4.schedule_digest(schedules)}
    holdout = {"schema_version": "cascadekv-phi35-8k-c4-holdout-selection-v1", "protocol_sha256": c4.sha256_path(c4.PROTOCOL), "development_source_manifest_sha256": c4.SOURCE_SHA, "development_result_sha256": c4.sha256_path(development_path), "development_schedule_sha256": development["schedule_sha256"], "selection_rule": "first mechanically eligible globally-new exact input SHA in ascending order", "selected_sources": selected, "rejected_sources": {family: [] for family in c4.FAMILIES}, "next_untouched_frontier": {x["family"]: x["dataset_index"] + 1 for x in selected}, "raw_text_persisted": False}
    holdout_path.write_text(json.dumps(holdout)); capture_path.write_text(json.dumps(_capture_manifest("holdout", holdout, holdout_sha=c4.sha256_path(holdout_path), development_sha=c4.sha256_path(development_path), schedule_sha=development["schedule_sha256"])))
    class Access:
        def __init__(self, resolver, **_kwargs): self.opened = c4._expected_tensor_ids("holdout", holdout)
    def rows(_access, _protocol, schedules):
        result = {name: [{"metrics": {"cosine_similarity": .99, "relative_l2_error": .1}, "traffic": {"total_kv_bytes": 100}}] * 1440 for name in c4.BASELINES}
        result.update({name: [{"metrics": {"cosine_similarity": .99, "relative_l2_error": .09}, "traffic": {"total_kv_bytes": 99}}] * 1440 for name in schedules})
        return result
    monkeypatch.setattr(c4, "preflight", lambda: {})
    monkeypatch.setattr(c4, "_validate_development_result", lambda _value: development)
    monkeypatch.setattr(c4, "TensorAccess", Access)
    monkeypatch.setattr(c4, "_method_rows", rows)
    monkeypatch.setattr(c4, "optimize_static_schedule", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("optimizer invoked")))
    result = c4.test(capture_root=tmp_path, capture_manifest=capture_path, holdout_manifest=holdout_path, development_result=development_path, output=output)
    assert result["opened_development_tensor_identities"] == []
    assert result["opened_holdout_tensor_identities"] == c4._expected_tensor_ids("holdout", holdout)
    assert result["no_schedule_mutation_evidence"]["optimizer_rerun"] is False
    assert result["no_schedule_mutation_evidence"]["schedule_sha256_before"] == result["no_schedule_mutation_evidence"]["schedule_sha256_after"]


def test_no_frozen_c3_file_is_modified():
    frozen = ["cascadekv/phi35_8k_c3.py", "configs/cascadekv_phi35_8k_c3_protocol.json", "configs/cascadekv_phi35_8k_c3_runtime_manifest.json", "cascadekv/phi35_phaseb.py", "cascadekv/phi35_kaggle_c2.py", "cascadekv/phi35_8k_source_amendment.py", "cascadekv/phi35_8k_capture_amendment.py"]
    import subprocess
    changed = subprocess.check_output(["git", "diff", "--name-only", "--", *frozen], text=True).splitlines()
    assert changed == []
