"""Additive, fail-closed Phi-3.5 8K C4 development and prospective test.

Imports are deliberately data-free.  The only functions which import the
model, tokenizer, dataset, torch, or safetensors stacks are capture/select or
payload open paths, after their respective local provenance gates succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cascadekv_phi35_8k_c4_protocol.json"
RUNTIME = ROOT / "configs/cascadekv_phi35_8k_c4_runtime_manifest.json"
C4_TAG = "cascadekv-phi35-8k-c4-prep-v2"
C3 = {"tag": "cascadekv-phi35-8k-c3-prep-v1", "commit": "80e4a3a09dbdd3f506a617c0a0607077e20926c8", "protocol_sha256": "ec16166a0adb88dff4e8739a78ed3a42896dba2ded2f94868ec5d7f5a1a4089e", "runtime_sha256": "2e579784bcd29d7e393b236acb7958596d7d9bc86f43737da4bf61f1d290c7ee"}
C3_RESULTS = {"calibration_result_sha256": "a2a48a677f6da7bf7c887a51b12ab9f73588fbca70314c485854ce05495c8f34", "validation_result_sha256": "175d7156a5969f68b35fa4d69919a85a08de2c389a91f21ff5a8ce255a3fefb2"}
SOURCE_SHA = "31f1d14d4b33b6e5b6a48d34e294f0effbab2eeadf6c4d4eabfabe16ae879478"
SOURCE_AMENDMENT = {"tag": "cascadekv-phi35-8k-source-dedup-amendment-prep-v1", "commit": "a26c76de596d6f0074556ae630679ffca0021e69", "protocol_sha256": "f43b2311d0decf2a579bf886848572bf53d58ab98e058b1da1be5a2912b85e8e", "runtime_manifest_sha256": "d35817df8a7c8a7a48539cac252344dbe4f94cf97b9a3a50aa307e88a3640edf"}
QUALIFICATION_SHA = "f4703bdcd477ee91f108dde4415e14831ec20a8b156fe59876d8e3a7699bc07e"
BACKEND_ID = "184109f4e8edec3c34455bd86a787e7838bef2eb75424cf0a4f57a183f313a95"
MODEL, REVISION = "microsoft/Phi-3.5-mini-instruct", "2fe192450127e6a83f7441aef6e3ca586c338b77"
LAYERS, POSITIONS, HEADS, SHAPE = (0, 8, 16, 24, 31), (4095, 6143, 8191), tuple(range(32)), (1, 32, 8192, 96)
FAMILIES = ("narrative", "report", "qa")
DEVELOPMENT = {"narrative": (13, 14, 15), "report": (16, 17, 18), "qa": (13, 15, 16)}
HOLDOUT_STARTS = {"narrative": 16, "report": 19, "qa": 17}
SOURCE_SPECS = {
    "narrative": {"dataset": "emozilla/pg19", "config": None, "split": "train", "revision": "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8", "field": "text", "first_permitted_index": 13},
    "report": {"dataset": "ccdv/govreport-summarization", "config": None, "split": "train", "revision": "4e21184e01ae8017e2c036e180fe5e541fef60a0", "field": "report", "first_permitted_index": 14},
    "qa": {"dataset": "zai-org/LongBench", "config": "narrativeqa", "split": "test", "revision": "75b6d5bffbcaa2cf4da85a9fa99939b13ee5b00b", "field": "context", "first_permitted_index": 13},
}
BASE_UNAVAILABLE = {"narrative": list(range(16)), "report": list(range(19)), "qa": list(range(16))}
ROLE_ORDER = ("calibration_1", "calibration_2", "validation")
QA13_SHA = "6efe85ed32e8f8e28ba946fa301f65bb7559f7ed596da3f62a519833ce3a80a4"
QA15_SHA = "29e61e009d96dc7e70261a07b197aaee190073884aaf8a8e1a53d4c1265b62b7"
ACTIONS, TARGETS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6"), ("T0", "T1", "T2", "T3", "T4", "T5")
BASELINES = ("dense", "flat5", "flat10", "uniform5", "uniform10")
SCHEDULE_FIELDS = ("feasible", "target_mean_kv_bytes", "strict", "integer_microbyte_cap", "integer_microbytes_used", "table")


class C4Error(RuntimeError):
    """A C4 provenance, firewall, capture, or scientific contract failed."""


def _progress(message: str) -> None:
    print(f"C4: {message}", file=sys.stderr, flush=True)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise C4Error(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise C4Error(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise C4Error(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(canonical_json(value) + "\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def _git_commit(ref: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{ref}^{{commit}}"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise C4Error(f"required git reference unavailable: {ref}") from exc


def _tracked(relative: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative], capture_output=True, check=False).returncode == 0


def _safe(relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise C4Error("path must be a nonempty relative string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise C4Error("unsafe relative path")
    return path


def _safe_under(root: Path, relative: str) -> Path:
    path = root / _safe(relative)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise C4Error("artifact path escapes declared root") from exc
    if path.is_symlink():
        raise C4Error("symlink artifacts are rejected")
    return path


def _expected_sources(kind: str) -> list[str]:
    if kind == "development":
        return [f"{family}:{index}:development" for family in FAMILIES for index in DEVELOPMENT[family]]
    if kind == "holdout":
        return [f"{family}:*:holdout" for family in FAMILIES]
    raise C4Error("unknown source kind")


def _expected_tensor_ids(kind: str, holdout: Mapping[str, Any] | None = None) -> list[str]:
    if kind == "development":
        return sorted(f"{family}:{index}:development:L{layer}" for family in FAMILIES for index in DEVELOPMENT[family] for layer in LAYERS)
    if not holdout:
        raise C4Error("holdout manifest required")
    selected = holdout.get("selected_sources")
    if not isinstance(selected, list):
        raise C4Error("holdout selection malformed")
    return sorted(f"{x['family']}:{x['dataset_index']}:holdout:L{layer}" for x in selected for layer in LAYERS)


def _no_raw_text(value: Any) -> None:
    forbidden = {"text", "raw_text", "content", "document", "prompt", "value"}
    if isinstance(value, Mapping):
        if forbidden & set(value):
            raise C4Error("raw source text is forbidden in C4 provenance")
        for child in value.values(): _no_raw_text(child)
    elif isinstance(value, list):
        for child in value: _no_raw_text(child)


def _protocol_actions() -> dict[str, dict[str, Any]]:
    return {"A0": {"kind": "hierarchy", "candidate_fraction": .05, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}, "A1": {"kind": "hierarchy", "candidate_fraction": .075, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}, "A2": {"kind": "hierarchy", "candidate_fraction": .10, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}, "A3": {"kind": "hierarchy", "candidate_fraction": .15, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}, "A4": {"kind": "flat_q8k4", "candidate_fraction": .05, "routing_profile": None}, "A5": {"kind": "hierarchy", "candidate_fraction": .175, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}, "A6": {"kind": "hierarchy", "candidate_fraction": .20, "routing_profile": "phi35-8k-depth-transfer-qwen5-v1"}}


def validate_protocol(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = dict(_json(PROTOCOL) if payload is None else payload)
    required = {"schema_version", "c4_freeze", "c3_freeze", "c3_observed_no_pass", "corrected_development_sources", "source_amendment_freeze", "c2_qualification", "geometry", "methods", "targets", "quality_gates", "optimizer", "holdout_selection", "capture", "firewall", "scientific_state"}
    if set(p) != required or p["schema_version"] != "cascadekv-phi35-8k-c4-protocol-v1": raise C4Error("C4 protocol schema differs")
    if p["c4_freeze"] != {"tag": C4_TAG}: raise C4Error("C4 freeze tag binding differs")
    if p["c3_freeze"] != C3 or p["c3_observed_no_pass"].get("calibration_result_sha256") != C3_RESULTS["calibration_result_sha256"] or p["c3_observed_no_pass"].get("validation_result_sha256") != C3_RESULTS["validation_result_sha256"]: raise C4Error("C3 frozen no-pass provenance differs")
    no_pass = p["c3_observed_no_pass"]
    if any(no_pass.get(key) != value for key, value in {"first_passing_target": None, "optimizer_rerun": False, "schedule_unchanged": True, "opened_calibration_tensors_during_validation": 0, "opened_validation_tensors": 15, "all_targets_feasible": True, "all_targets_lower_relative_l2_than_uniform10": True, "all_targets_lower_modeled_kv_than_flat5": True, "absolute_gates_passed": False}.items()): raise C4Error("C3 must be recorded as a no-pass")
    if p["corrected_development_sources"] != {"manifest_sha256": SOURCE_SHA, "identities": {family: list(DEVELOPMENT[family]) for family in FAMILIES}, "forbidden": ["qa:14"]}: raise C4Error("corrected nine-source C4 development inventory differs")
    if p["source_amendment_freeze"] != SOURCE_AMENDMENT: raise C4Error("source-amendment freeze binding differs")
    if p["c2_qualification"] != {"qualification_sha256": QUALIFICATION_SHA, "backend_id": BACKEND_ID, "target_model": MODEL, "target_revision": REVISION}: raise C4Error("C2 qualification/backend binding differs")
    if p["geometry"] != {"context_length": 8192, "layers": list(LAYERS), "query_positions": list(POSITIONS), "heads": 32, "head_dim": 96, "schedule_cells": 160, "observations_per_source_per_method": 480, "development_observations_per_method": 4320, "holdout_observations_per_method": 1440}: raise C4Error("C4 geometry differs")
    methods = p["methods"]
    if methods.get("actions") != _protocol_actions() or list(methods["actions"]) != list(ACTIONS): raise C4Error("C4 seven-action menu/order differs")
    if methods.get("hierarchy") != {"atom_size": 8, "forest": "causal_binary_counter_lifting", "grouped_variance": "G4", "routing": "K4_group16", "rerank": "exact_fp16_K", "selected_values": "FP16"}: raise C4Error("C4 hierarchy differs")
    if methods.get("baselines") != [*BASELINES, "target"]: raise C4Error("C4 baselines differ")
    # Bind profiles directly to the frozen Phase-B source rather than copying a new tuning surface.
    from cascadekv import phi35_phaseb
    frozen = phi35_phaseb.validate_protocol()["routing_profiles"]
    if methods.get("routing_profiles") != frozen: raise C4Error("C4 routing profile mutation is forbidden")
    if p["targets"] != {"order": list(TARGETS), "fractions": {"T0": .115, "T1": .130, "T2": .145, "T3": .160, "T4": .180, "T5": "highest_traffic_strictly_below_development_flat5"}}: raise C4Error("C4 targets differ")
    if p["quality_gates"] != {"mean_cosine_gte": .985, "mean_relative_l2_lte": .120, "winner_relative_l2_lt": "uniform10", "winner_modeled_kv_traffic_lt": "flat5"}: raise C4Error("quality gates differ")
    if p["holdout_selection"].get("start_indices") != HOLDOUT_STARTS or p["holdout_selection"].get("persist_raw_text") is not False or p["holdout_selection"].get("one_per_family") is not True or p["holdout_selection"].get("stop_after_first_globally_new_eligible") is not True or p["holdout_selection"].get("requires_frozen_development_result") is not True or p["holdout_selection"].get("development_hash_source") != "canonical_corrected_development_source_manifest_only" or p["holdout_selection"].get("persist_bindings") != ["development_result_sha256", "development_schedule_sha256"]: raise C4Error("fresh holdout contract differs")
    if p["capture"] != {"semantics": "canonical C2 FP16 SDPA use_cache=false no-quantization deterministic T4x2 phi3-post-rope-qkv-v1", "development_artifacts": 45, "holdout_artifacts": 15, "layers_per_source_forward": 5, "never_overwrite": True, "storage": "CPU-contiguous FP16 safetensors", "holdout_requires_exact_manifest_and_development_bindings": True}: raise C4Error("C4 capture contract differs")
    if p["firewall"] != {"develop_opens": "exactly 45 development artifacts and zero holdout artifacts", "test_opens": "exactly 15 holdout artifacts and zero development artifacts", "test_forbidden": ["optimizer_rerun", "schedule_mutation", "action_change", "target_change", "target_reorder"]}: raise C4Error("C4 firewall differs")
    if p["scientific_state"] != {"local_preparation_only": True, "fresh_holdout_identities_selected": False, "fresh_holdout_source_content_inspected": False, "fresh_holdout_qkv_captured": False, "c4_development_metrics_observed": False, "c4_schedules_optimized": False, "c4_test_metrics_observed": False}: raise C4Error("C4 local-only scientific state differs")
    return p


def verify_runtime_closure(*, require_tracked: bool = True) -> tuple[dict[str, str], ...]:
    manifest = _json(RUNTIME)
    if set(manifest) != {"schema_version", "purpose", "protocol", "bound_files"} or manifest["schema_version"] != 1: raise C4Error("C4 runtime manifest schema differs")
    if manifest["protocol"] != {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256_path(PROTOCOL)}: raise C4Error("C4 protocol runtime binding differs")
    seen, verified = set(), []
    for row in manifest["bound_files"]:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}: raise C4Error("malformed runtime entry")
        relative, digest = row["path"], row["sha256"]
        if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or relative in seen: raise C4Error("unsafe runtime entry")
        seen.add(relative); path = ROOT / _safe(relative)
        if not path.is_file() or (require_tracked and not _tracked(relative)) or sha256_path(path) != digest: raise C4Error(f"runtime closure mismatch: {relative}")
        verified.append({"path": relative, "sha256": digest})
    return tuple(verified)


def preflight(*, require_c4_tag: bool = True, require_c4_tracked: bool = True) -> dict[str, Any]:
    """Fully local: no model, tokenizer, datasets, tensor payloads, or network."""
    if _git_commit(C3["tag"]) != C3["commit"]: raise C4Error("C3 freeze tag differs")
    if sha256_path(ROOT / "configs/cascadekv_phi35_8k_c3_protocol.json") != C3["protocol_sha256"] or sha256_path(ROOT / "configs/cascadekv_phi35_8k_c3_runtime_manifest.json") != C3["runtime_sha256"]: raise C4Error("C3 protocol/runtime freeze differs")
    if _git_commit(SOURCE_AMENDMENT["tag"]) != SOURCE_AMENDMENT["commit"]: raise C4Error("source-amendment freeze tag differs")
    if sha256_path(ROOT / "configs/cascadekv_phi35_8k_source_amendment_protocol.json") != SOURCE_AMENDMENT["protocol_sha256"] or sha256_path(ROOT / "configs/cascadekv_phi35_8k_source_amendment_runtime_manifest.json") != SOURCE_AMENDMENT["runtime_manifest_sha256"]: raise C4Error("source-amendment protocol/runtime freeze differs")
    if require_c4_tag:
        try:
            c4_tag_commit = _git_commit(C4_TAG)
        except C4Error as exc:
            raise C4Error("required C4 freeze tag is unavailable") from exc
        if c4_tag_commit != _git_commit("HEAD"): raise C4Error("HEAD is not the required C4 freeze tag commit")
    validate_protocol(); bound = verify_runtime_closure(require_tracked=require_c4_tracked)
    return {"protocol_path": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha256_path(PROTOCOL), "runtime_manifest_path": str(RUNTIME.relative_to(ROOT)), "runtime_manifest_sha256": sha256_path(RUNTIME), "bound_files": bound, "local_only": True}


def preparation_preflight() -> dict[str, Any]:
    """Local preparation-only validation before the final C4 commit and tag."""
    return preflight(require_c4_tag=False, require_c4_tracked=False)


def _input_hash(ids: Any) -> str:
    """Canonical exact input identity; deliberately rejects noncanonical IDs."""
    if str(ids.dtype) != "torch.int64" or ids.device.type != "cpu" or not ids.is_contiguous() or tuple(ids.shape) != (1, 8192): raise C4Error("input ids must be contiguous CPU int64 [1,8192]")
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def _source_identity(source: Mapping[str, Any], role: str) -> str:
    return f"{source['family']}:{source['dataset_index']}:{role}"


def _validate_development_manifest(payload: Mapping[str, Any], path: Path | None = None) -> list[dict[str, Any]]:
    """Validate the frozen corrected-source content without another phase's HEAD gate.

    This is intentionally a C4-local, text-free validation surface.  It binds
    the amendment as immutable historical provenance, but never imports its
    validator or asks the amendment tag to equal the C4 checkout's HEAD.
    """
    if path is not None and sha256_path(path) != SOURCE_SHA: raise C4Error("development source manifest historical SHA differs")
    _no_raw_text(payload)
    required = {"schema_version", "purpose", "amendment_reason", "base_c1_freeze", "phase_b_freeze", "c2_v2_freeze", "amendment_freeze", "target", "tokenization_contract", "identity_kind", "uniqueness_kind", "source_text_stored", "source_specs", "selected_sources", "rejected_sources", "last_inspected_index", "unavailable_inventory", "reproof_status"}
    if not isinstance(payload, Mapping) or set(payload) != required or payload.get("schema_version") != "cascadekv-phi35-8k-dev-sources-v2": raise C4Error("canonical corrected development manifest schema differs")
    if payload.get("purpose") != "C1 source-identity amendment: exact model-input deduplication only; no quality evaluation" or payload.get("amendment_reason") != "C1 unique dataset indices did not ensure unique exact 8192-token model inputs": raise C4Error("canonical corrected development manifest purpose differs")
    if payload.get("base_c1_freeze") != {"tag": "cascadekv-phi35-8k-sources-freeze", "commit": "667900a5307fe231565033a774d7788942fb869d", "manifest_sha256": "af8e805bad5c85a1f9d6ae5571b378e4df46fdb1de778a5ec7b5b68b13dc4a8e"}: raise C4Error("canonical corrected development C1 binding differs")
    if payload.get("phase_b_freeze") != {"tag": "cascadekv-phi35-8k-phaseb-protocol-freeze", "commit": "d94cff8e1e11048ff3d8edc9fb1c605dfe422d6e", "protocol_sha256": "0e3a2e7b51e01c64dca292c7dd667ed9899c5bacc08e470e2b19e12433363c8b", "runtime_manifest_sha256": "e4e10f7659180471eb791eeafa97e8f525b6c14a3ff311798349c07438ec7c92"}: raise C4Error("canonical corrected development Phase-B binding differs")
    if payload.get("c2_v2_freeze") != {"tag": "cascadekv-phi35-8k-kaggle-c2-prep-v2", "commit": "fa54b581ee5ee581dd51b8085db8b5233467450a", "protocol_sha256": "315453e8446e8f41611fc23544a4a5093da13a7dba344dfc66a840eebf17857d", "runtime_manifest_sha256": "a5a96915f8cab588e5d1dd6af15855d4d090366f94282022bff7bc1b856957fd"}: raise C4Error("canonical corrected development C2-v2 binding differs")
    if payload.get("amendment_freeze") != {"tag": SOURCE_AMENDMENT["tag"], "commit": SOURCE_AMENDMENT["commit"], "amendment_protocol_sha256": SOURCE_AMENDMENT["protocol_sha256"], "amendment_runtime_manifest_sha256": SOURCE_AMENDMENT["runtime_manifest_sha256"]}: raise C4Error("canonical corrected development source-amendment binding differs")
    if payload.get("target") != {"model": MODEL, "model_revision": REVISION, "tokenizer_revision": REVISION}: raise C4Error("canonical corrected development target binding differs")
    if payload.get("tokenization_contract") != {"add_special_tokens": True, "truncation": True, "max_length": 8192, "required_input_ids_length": 8192, "canonicalization": "CPU contiguous int64 bytes"}: raise C4Error("canonical corrected development tokenization contract differs")
    if payload.get("identity_kind") != "dataset_index" or payload.get("uniqueness_kind") != "input_ids_sha256" or payload.get("source_text_stored") is not False or payload.get("source_specs") != SOURCE_SPECS: raise C4Error("canonical corrected development identity/specification differs")
    sources = payload.get("selected_sources")
    if not isinstance(sources, list) or len(sources) != 9: raise C4Error("C4 development requires exactly nine sources")
    source_keys = {"family", "dataset", "config", "split", "revision", "field", "identity_kind", "dataset_index", "role", "input_ids_sha256", "proof", "reproof"}
    proof_keys = {"mechanical_eligibility", "input_ids_length", "input_ids_sha256"}
    eligibility_keys = {"field_exists", "is_python_string", "character_count", "bounded_frozen_tokenizer_length", "at_least_8192"}
    out, hashes = [], set()
    for family in FAMILIES:
        rows = [x for x in sources if isinstance(x, Mapping) and x.get("family") == family]
        if len(rows) != 3 or [x.get("dataset_index") for x in rows] != list(DEVELOPMENT[family]) or [x.get("role") for x in rows] != list(ROLE_ORDER): raise C4Error("C4 corrected development identities differ")
        for source in rows:
            if set(source) != source_keys or any(source.get(key) != SOURCE_SPECS[family][key] for key in ("dataset", "config", "split", "revision", "field")) or source.get("identity_kind") != "dataset_index": raise C4Error("canonical corrected development source record differs")
            digest, proof = source.get("input_ids_sha256"), source.get("proof")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in hashes: raise C4Error("canonical corrected development selected input hashes differ")
            if not isinstance(proof, Mapping) or set(proof) != proof_keys or source.get("reproof") != proof or proof.get("input_ids_length") != 8192 or proof.get("input_ids_sha256") != digest: raise C4Error("canonical corrected development proof/reproof differs")
            eligibility = proof.get("mechanical_eligibility")
            if not isinstance(eligibility, Mapping) or set(eligibility) != eligibility_keys or eligibility.get("field_exists") is not True or eligibility.get("is_python_string") is not True or eligibility.get("at_least_8192") is not True: raise C4Error("canonical corrected development mechanical eligibility differs")
            if not isinstance(eligibility.get("character_count"), int) or isinstance(eligibility["character_count"], bool) or eligibility["character_count"] < 0 or eligibility.get("bounded_frozen_tokenizer_length") != 8192: raise C4Error("canonical corrected development mechanical proof differs")
            hashes.add(digest)
            out.append(dict(source, canonical_role=source["role"], role="development"))
    if len(hashes) != 9: raise C4Error("canonical corrected development hash inventory differs")
    qa = [source for source in sources if source["family"] == "qa"]
    if qa[0]["input_ids_sha256"] != QA13_SHA or qa[1]["input_ids_sha256"] != QA15_SHA or qa[2]["dataset_index"] != 16: raise C4Error("QA 13/15/16 frozen input transition differs")
    rejected = payload.get("rejected_sources")
    if not isinstance(rejected, Mapping) or set(rejected) != set(FAMILIES) or rejected.get("narrative") != [] or rejected.get("report") != [] or not isinstance(rejected.get("qa"), list) or len(rejected["qa"]) != 1: raise C4Error("canonical corrected development rejection inventory differs")
    qa14 = rejected["qa"][0]
    if not isinstance(qa14, Mapping) or set(qa14) != {"dataset_index", "rejection_reason", "duplicate_of_dataset_index", "input_ids_sha256", "proof"} or qa14.get("dataset_index") != 14 or qa14.get("rejection_reason") != "duplicate_input" or qa14.get("duplicate_of_dataset_index") != 13 or qa14.get("input_ids_sha256") != QA13_SHA or qa14.get("proof") != qa[0]["proof"]: raise C4Error("QA14 rejection proof differs")
    last = payload.get("last_inspected_index")
    expected_last = {family: DEVELOPMENT[family][-1] for family in FAMILIES}
    if last != expected_last: raise C4Error("canonical corrected development stop inventory differs")
    inventories = payload.get("unavailable_inventory")
    if not isinstance(inventories, Mapping) or set(inventories) != set(FAMILIES): raise C4Error("canonical corrected development unavailable inventory differs")
    for family in FAMILIES:
        inspected = list(DEVELOPMENT[family]) if family != "qa" else [13, 14, 15, 16]
        expected_inventory = {"base_c1_unavailable_indices": BASE_UNAVAILABLE[family], "amendment_inspected_indices": inspected, "indices": sorted(set(BASE_UNAVAILABLE[family] + inspected)), "next_untouched_index": expected_last[family] + 1}
        if inventories.get(family) != expected_inventory: raise C4Error("canonical corrected development unavailable inventory differs")
    if payload.get("reproof_status") != {"all_selected_sources_equal": True, "selected_source_count": 9, "global_input_ids_sha256_unique": True}: raise C4Error("canonical corrected development reproof status differs")
    return out


def _development_hashes(sources: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    hashes = tuple(str(source.get("input_ids_sha256")) for source in sources)
    if len(hashes) != 9 or len(set(hashes)) != 9 or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise C4Error("canonical development manifest does not provide nine globally unique input hashes")
    return hashes


def _proof_eligible(proof: Mapping[str, Any]) -> bool:
    return proof.get("field_exists") is True and proof.get("is_python_string") is True and proof.get("at_least_8192") is True


def select_holdout_from_inspections(inspections: Mapping[str, Iterable[Mapping[str, Any]]], development_hashes: Iterable[str], *, development_result_sha256: str = "0" * 64, development_schedule_sha256: str = "0" * 64) -> dict[str, Any]:
    """Pure selector for tests/adapters; inspections contain no source text.

    Each inspection must already have the exact canonical input hash produced
    from the pinned tokenizer.  The function deliberately stops immediately
    after first globally-new eligible acceptance per family.
    """
    development_hashes = tuple(development_hashes)
    known = set(development_hashes)
    if len(development_hashes) != 9 or len(known) != 9 or any(not re.fullmatch(r"[0-9a-f]{64}", x) for x in known): raise C4Error("nine development input hashes are required")
    selected, rejected, next_frontier = [], {family: [] for family in FAMILIES}, {}
    for family in FAMILIES:
        rows = sorted((dict(x) for x in inspections.get(family, [])), key=lambda x: x.get("dataset_index", -1))
        accepted = None; last = HOLDOUT_STARTS[family] - 1
        for row in rows:
            index = row.get("dataset_index")
            if not isinstance(index, int) or index < HOLDOUT_STARTS[family]: continue
            last = index
            required = {"dataset_index", "proof", "input_ids_sha256", "source"}
            if set(row) != required or not isinstance(row["proof"], Mapping) or not isinstance(row["source"], Mapping): raise C4Error("holdout inspection schema differs")
            digest = row["input_ids_sha256"]
            if not _proof_eligible(row["proof"]): rejected[family].append({"dataset_index": index, "reason": "mechanically_ineligible", "proof": dict(row["proof"])}); continue
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest): raise C4Error("exact input hash is malformed")
            if digest in known: rejected[family].append({"dataset_index": index, "reason": "duplicate_exact_input_ids_sha256", "input_ids_sha256": digest, "proof": dict(row["proof"])}); continue
            source = dict(row["source"]); _no_raw_text(source)
            if source.get("family") != family or source.get("dataset_index") != index: raise C4Error("inspection source identity differs")
            accepted = {**source, "role": "holdout", "proof": dict(row["proof"]), "input_ids_sha256": digest}; known.add(digest); break
        if accepted is None: raise C4Error(f"no eligible globally-new holdout source for {family}")
        selected.append(accepted); next_frontier[family] = last + 1
    if not re.fullmatch(r"[0-9a-f]{64}", development_result_sha256) or not re.fullmatch(r"[0-9a-f]{64}", development_schedule_sha256): raise C4Error("frozen development binding malformed")
    result = {"schema_version": "cascadekv-phi35-8k-c4-holdout-selection-v1", "protocol_sha256": sha256_path(PROTOCOL), "development_source_manifest_sha256": SOURCE_SHA, "development_result_sha256": development_result_sha256, "development_schedule_sha256": development_schedule_sha256, "selection_rule": "first mechanically eligible globally-new exact input SHA in ascending order", "selected_sources": selected, "rejected_sources": rejected, "next_untouched_frontier": next_frontier, "raw_text_persisted": False}
    _validate_holdout_manifest(result); return result


def _validate_holdout_manifest(payload: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    _no_raw_text(payload)
    required = {"schema_version", "protocol_sha256", "development_source_manifest_sha256", "development_result_sha256", "development_schedule_sha256", "selection_rule", "selected_sources", "rejected_sources", "next_untouched_frontier", "raw_text_persisted"}
    if set(payload) != required or payload.get("schema_version") != "cascadekv-phi35-8k-c4-holdout-selection-v1" or payload.get("protocol_sha256") != sha256_path(PROTOCOL) or payload.get("development_source_manifest_sha256") != SOURCE_SHA or payload.get("raw_text_persisted") is not False: raise C4Error("holdout manifest provenance differs")
    if any(not isinstance(payload[key], str) or not re.fullmatch(r"[0-9a-f]{64}", payload[key]) for key in ("development_result_sha256", "development_schedule_sha256")): raise C4Error("holdout frozen development binding differs")
    selected = payload["selected_sources"]
    if not isinstance(selected, list) or len(selected) != 3 or [x.get("family") for x in selected] != list(FAMILIES): raise C4Error("holdout must select one source per family")
    hashes = set()
    for source in selected:
        if source.get("role") != "holdout" or not isinstance(source.get("dataset_index"), int) or source["dataset_index"] < HOLDOUT_STARTS[source["family"]] or not _proof_eligible(source.get("proof", {})) or not isinstance(source.get("input_ids_sha256"), str): raise C4Error("holdout source is not mechanically eligible")
        if source["input_ids_sha256"] in hashes: raise C4Error("fresh holdouts duplicate each other")
        hashes.add(source["input_ids_sha256"])
    if payload["next_untouched_frontier"] != {x["family"]: x["dataset_index"] + 1 for x in selected}: raise C4Error("next untouched holdout frontier differs")
    return dict(payload)


def select_holdout(*, development_source_manifest: Path, development_result: Path, output: Path) -> dict[str, Any]:
    """Kaggle-only selector.  It loads source/tokenizer, never model/metrics."""
    preflight()
    # This must be complete before tokenizer/dataset access.  The path digest
    # is the prospective freeze that the holdout manifest carries forward.
    development = _validate_development_result(_json(development_result))
    development_sha = sha256_path(development_result)
    sources = _validate_development_manifest(_json(development_source_manifest), development_source_manifest)
    dev_hashes = _development_hashes(sources)
    # Delayed external imports preserve preflight's local-only property.
    from cascadekv.phi35_8k_source_selection import frozen_specs, load_dataset_for_spec, load_frozen_tokenizer, mechanical_proof
    import torch
    tokenizer, specs = load_frozen_tokenizer(), frozen_specs()
    inspections: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    known = set(dev_hashes)
    for family in FAMILIES:
        dataset, index = load_dataset_for_spec(specs[family]), HOLDOUT_STARTS[family]
        while index < len(dataset):
            proof = mechanical_proof(dataset, tokenizer, specs[family], index)
            row = dataset[index]; value = row.get(specs[family]["field"])
            digest = "0" * 64
            if _proof_eligible(proof):
                encoded = tokenizer(value, add_special_tokens=True, truncation=True, max_length=8192)
                digest = _input_hash(torch.tensor([encoded.input_ids], dtype=torch.int64).contiguous())
            source = {"family": family, "dataset": specs[family]["dataset"], "config": specs[family]["config"], "split": specs[family]["split"], "revision": specs[family]["revision"], "field": specs[family]["field"], "identity_kind": "dataset_index", "dataset_index": index}
            inspection = {"dataset_index": index, "proof": proof, "input_ids_sha256": digest, "source": source}
            inspections[family].append(inspection)
            # This is the prescribed mechanical stop rule.  Keep any prior
            # rejections for provenance, but never inspect a later identity
            # once this family has a globally-new eligible exact input.
            if _proof_eligible(proof) and digest not in known:
                known.add(digest); break
            index += 1
    # The pure multi-family pass applies cross-family dedup and asserts all rules.
    result = select_holdout_from_inspections(inspections, dev_hashes, development_result_sha256=development_sha, development_schedule_sha256=development["schedule_sha256"])
    atomic_json(output, result); return result


def _action_table_keys() -> set[str]: return {f"{layer}:{head}" for layer in LAYERS for head in HEADS}


def _target_cap(protocol: Mapping[str, Any], target: str, dense: float, flat5: float) -> tuple[float, bool, int]:
    target_mean, strict = (flat5, True) if target == "T5" else (float(protocol["targets"]["fractions"][target]) * dense, False)
    return target_mean, strict, math.ceil(target_mean * 160 * 1000) - (1 if strict else 0)


def _optimize_action_cells(groups: Iterable[Mapping[str, Mapping[str, float]]], cap: int) -> tuple[int, float, float, tuple[str, ...]] | None:
    """Exact generic seven-action integer-microbyte DP for synthetic testing.

    State reductions retain the lexicographically best representative for an
    exact budget and eliminate only states Pareto-dominated in (error, cosine).
    """
    rows = list(groups)
    if not rows or type(cap) is not int or cap < 0: raise C4Error("optimizer requires cells and a nonnegative integer cap")
    states: dict[int, tuple[float, float, int]] = {0: (0.0, 0.0, 0)}
    for row in rows:
        candidate: dict[int, tuple[float, float, int]] = {}
        for budget, (error, cosine, code) in states.items():
            for index, action in enumerate(ACTIONS):
                stat = row.get(action, {})
                traffic, rel, cos = stat.get("total_kv_bytes"), stat.get("relative_l2"), stat.get("cosine")
                if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in (traffic, rel, cos)): raise C4Error("incomplete action statistics")
                next_budget = budget + round(float(traffic) * 1000)
                if next_budget > cap: continue
                item = (error + float(rel), cosine + float(cos), code * 7 + index)
                prior = candidate.get(next_budget)
                if prior is None or (item[0], -item[1], item[2]) < (prior[0], -prior[1], prior[2]): candidate[next_budget] = item
        # Any earlier/lower-budget state with lower error and higher cosine can
        # replace a later state for every remaining cell, so this is exact.
        states, best_error, best_cos = {}, math.inf, -math.inf
        for budget in sorted(candidate):
            error, cosine, code = candidate[budget]
            if error < best_error or (error == best_error and cosine > best_cos):
                states[budget] = (error, cosine, code)
                if error < best_error: best_error, best_cos = error, cosine
                elif cosine > best_cos: best_cos = cosine
    if not states: return None
    budget, (error, cosine, code) = min(states.items(), key=lambda item: (item[1][0], -item[1][1], item[0], item[1][2]))
    table = []
    for _ in rows:
        code, digit = divmod(code, 7); table.append(ACTIONS[digit])
    return budget, error, cosine, tuple(reversed(table))


def optimize_static_schedule(groups: Iterable[Mapping[str, Mapping[str, float]]], target_mean_kv: float, *, strict: bool = False) -> tuple[int, float, float, tuple[str, ...]] | None:
    """Production C4 wrapper: exactly 160 layer/head schedule cells."""
    rows = list(groups)
    if len(rows) != 160 or not math.isfinite(target_mean_kv): raise C4Error("optimizer requires 160 cells and finite target")
    return _optimize_action_cells(rows, math.ceil(target_mean_kv * len(rows) * 1000) - (1 if strict else 0))


def schedule_digest(schedules: Mapping[str, Any]) -> str:
    if list(schedules) != list(TARGETS): raise C4Error("schedule target order differs")
    rows = []
    for target in TARGETS:
        row = schedules[target]
        if not isinstance(row, Mapping) or set(row) != set(SCHEDULE_FIELDS): raise C4Error("schedule schema differs")
        rows.append({key: row[key] for key in SCHEDULE_FIELDS})
    return hashlib.sha256(canonical_json({"target_order": list(TARGETS), "schedules": rows}).encode("ascii")).hexdigest()


def _profile(protocol: Mapping[str, Any], profile: str) -> tuple[dict[int, float], dict[str, Any]]:
    row = protocol["methods"]["routing_profiles"][profile]
    return {int(k): float(v) for k, v in row["lambdas"].items()}, dict(row["reserve"])


def _source_provenance_identity(source: Mapping[str, Any], role: str) -> dict[str, Any]:
    fields = ("family", "dataset", "config", "split", "revision", "field", "identity_kind", "dataset_index")
    return {**{field: source[field] for field in fields}, "role": role}


ARTIFACT_PROVENANCE_FIELDS = {
    "schema_version", "protocol_sha256", "qualification_sha256", "backend_id", "identity", "layer", "input_ids_sha256", "target", "source", "source_proof",
    "q_shape", "k_shape", "v_shape", "storage_dtype", "model_compute_dtype", "attention_implementation", "use_cache", "quantization", "capture_adapter", "artifact_sha256",
}


class CaptureResolver:
    """Manifest-first resolver; never opens tensors while validating identity."""
    def __init__(self, root: Path, manifest: Path, kind: str, *, holdout_manifest: Path | None = None, development_result_sha256: str | None = None, development_schedule_sha256: str | None = None):
        self.root, self.kind = root.resolve(), kind
        payload = _json(manifest); _no_raw_text(payload)
        expected_count = 45 if kind == "development" else 15
        expected_forwards = 9 if kind == "development" else 3
        required = {"schema_version", "kind", "protocol_sha256", "qualification_sha256", "backend_id", "source_manifest_sha256", "holdout_manifest_sha256", "development_result_sha256", "development_schedule_sha256", "artifact_count", "forwards", "artifacts"}
        if set(payload) != required or payload.get("schema_version") != "cascadekv-phi35-8k-c4-capture-v1" or payload.get("kind") != kind or payload.get("artifact_count") != expected_count or payload.get("forwards") != expected_forwards or payload.get("protocol_sha256") != sha256_path(PROTOCOL) or payload.get("qualification_sha256") != QUALIFICATION_SHA or payload.get("backend_id") != BACKEND_ID: raise C4Error("C4 capture manifest differs")
        if kind == "development" and payload.get("source_manifest_sha256") != SOURCE_SHA: raise C4Error("development capture source provenance differs")
        holdout: dict[str, Any] | None = None
        if kind == "development":
            if any(payload.get(field) is not None for field in ("holdout_manifest_sha256", "development_result_sha256", "development_schedule_sha256")): raise C4Error("development capture contains holdout bindings")
        else:
            if holdout_manifest is None: raise C4Error("actual holdout manifest path is required")
            holdout = _validate_holdout_manifest(_json(holdout_manifest), holdout_manifest)
            actual_holdout_sha = sha256_path(holdout_manifest)
            if payload.get("source_manifest_sha256") is not None or payload.get("holdout_manifest_sha256") != actual_holdout_sha or payload.get("development_result_sha256") != holdout["development_result_sha256"] or payload.get("development_schedule_sha256") != holdout["development_schedule_sha256"]: raise C4Error("holdout capture provenance differs")
            if development_result_sha256 is not None and payload["development_result_sha256"] != development_result_sha256: raise C4Error("holdout capture development-result binding differs")
            if development_schedule_sha256 is not None and payload["development_schedule_sha256"] != development_schedule_sha256: raise C4Error("holdout capture schedule binding differs")
        rows = payload.get("artifacts")
        if not isinstance(rows, list) or len(rows) != expected_count: raise C4Error("C4 capture inventory differs")
        expected = _expected_tensor_ids(kind, holdout)
        records: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"identity", "layer", "artifact_relative_path", "artifact_sha256", "provenance_relative_path", "provenance_sha256", "input_ids_sha256", "source"}: raise C4Error("capture record schema differs")
            identity = row["identity"]; key = f"{identity}:L{row['layer']}"
            if key not in expected or key in records or row["layer"] not in LAYERS: raise C4Error("capture record identity differs")
            for k in ("artifact_sha256", "provenance_sha256", "input_ids_sha256"):
                if not isinstance(row[k], str) or not re.fullmatch(r"[0-9a-f]{64}", row[k]): raise C4Error("capture digest malformed")
            family, index, role = identity.split(":")
            expected_source = {"family": family, "dataset_index": int(index), "role": role}
            if row["source"] != expected_source: raise C4Error("capture source identity differs")
            if holdout is not None:
                selected = next((source for source in holdout["selected_sources"] if _source_identity(source, "holdout") == identity), None)
                if selected is None or row["input_ids_sha256"] != selected["input_ids_sha256"]: raise C4Error("holdout capture source input binding differs")
            records[key] = dict(row)
        if sorted(records) != expected: raise C4Error("capture inventory incomplete")
        self.records, self.holdout = records, holdout

    def ordered(self) -> list[dict[str, Any]]: return [self.records[key] for key in sorted(self.records)]


class TensorAccess:
    def __init__(self, resolver: CaptureResolver, *, allowed_kind: str):
        if resolver.kind != allowed_kind: raise C4Error("tensor firewall denied capture kind")
        self.resolver, self.allowed_kind, self.opened = resolver, allowed_kind, []

    def open(self, record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        key = f"{record['identity']}:L{record['layer']}"
        if key not in self.resolver.records: raise C4Error("tensor firewall denied unknown identity")
        artifact = _safe_under(self.resolver.root, str(record["artifact_relative_path"]))
        provenance = _safe_under(self.resolver.root, str(record["provenance_relative_path"]))
        if not artifact.is_file() or not provenance.is_file() or sha256_path(provenance) != record["provenance_sha256"]: raise C4Error("artifact identity/provenance rejected before payload")
        detail = _json(provenance); _no_raw_text(detail)
        family, index, role = str(record["identity"]).split(":")
        expected_source = {"family": family, "dataset_index": int(index), "role": role}
        fixed = {"schema_version": "cascadekv-phi35-8k-c4-artifact-v1", "protocol_sha256": sha256_path(PROTOCOL), "qualification_sha256": QUALIFICATION_SHA, "backend_id": BACKEND_ID, "identity": record["identity"], "layer": record["layer"], "input_ids_sha256": record["input_ids_sha256"], "target": {"model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION}, "q_shape": list(SHAPE), "k_shape": list(SHAPE), "v_shape": list(SHAPE), "storage_dtype": "float16", "model_compute_dtype": "float16", "attention_implementation": "sdpa", "use_cache": False, "quantization": "none", "capture_adapter": "phi3-post-rope-qkv-v1", "artifact_sha256": record["artifact_sha256"]}
        source_fields = {"family", "dataset", "config", "split", "revision", "field", "identity_kind", "dataset_index", "role"}
        if set(detail) != ARTIFACT_PROVENANCE_FIELDS or any(detail.get(key) != value for key, value in fixed.items()) or not isinstance(detail.get("source_proof"), Mapping) or not isinstance(detail.get("source"), Mapping) or set(detail["source"]) != source_fields or detail["source"].get("family") != expected_source["family"] or detail["source"].get("dataset_index") != expected_source["dataset_index"] or detail["source"].get("role") != expected_source["role"]: raise C4Error("artifact provenance contract mismatch")
        if sha256_path(artifact) != record["artifact_sha256"]: raise C4Error("artifact hash mismatch")
        from safetensors import safe_open
        import torch
        with safe_open(str(artifact), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"q", "k", "v"}: raise C4Error("artifact tensor keys differ")
            q, k, v = (handle.get_tensor(name) for name in ("q", "k", "v"))
        if any(item.dtype != torch.float16 or tuple(item.shape) != SHAPE for item in (q, k, v)): raise C4Error("artifact shape/dtype differs")
        self.opened.append(key); return q, k, v


def _method_rows(access: TensorAccess, protocol: Mapping[str, Any], schedules: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, list[dict[str, Any]]]:
    from cascadekv.evaluation_core import advance_until_budget, build_forests, candidate_budget, exact_attention_reference, exact_rerank_and_output, initialize_route, reserve_ids, select_flat_k4, traffic
    from cascadekv.model_agnostic import PHI35_GEOMETRY
    methods = {name: [] for name in (*BASELINES, *ACTIONS)}
    if schedules: methods.update({target: [] for target in schedules})
    five_lambda, five_reserve = _profile(protocol, "phi35-8k-depth-transfer-qwen5-v1")
    ten_lambda, ten_reserve = _profile(protocol, "phi35-8k-depth-transfer-qwen10-v1")
    for artifact_no, record in enumerate(access.resolver.ordered(), 1):
        q_all, k_all, v_all = access.open(record); layer = int(record["layer"]); _progress(f"evaluated {access.allowed_kind} artifact {artifact_no}/{len(access.resolver.records)}: {record['identity']} L{layer}")
        forests = build_forests(k_all[0], PHI35_GEOMETRY, POSITIONS, atom_size=8, groups=(4,))
        for position in POSITIONS:
            n = position + 1
            for head in HEADS:
                query, keys, values = q_all[0, head, position], k_all[0, head, :n], v_all[0, head, :n]
                reference, forest = exact_attention_reference(query, keys, values), forests[position, head]
                def flat(frac: float):
                    budget = candidate_budget(n, frac); return set(select_flat_k4(query, keys, budget)), traffic({"active_root_reads": 0, "detail_reads": 0, "expanded_internal_nodes": 0, "variance_scalar_reads": 0}, n, budget, 96, variance=False, flat=True)
                def hierarchy(frac: float, lambdas: Mapping[int, float], reserve: Mapping[str, Any]):
                    budget = candidate_budget(n, frac); route = advance_until_budget(initialize_route(forest, query, 4, lambdas[layer], reserve_ids(n, int(reserve["sink"]), int(reserve["local"]), budget)), budget); return set(route["ids"]), traffic(route, n, budget, 96, variance=True)
                choices = {"dense": (set(range(n)), {"total_k_bytes": float(n * 192), "selected_v_fp16_bytes": float(n * 192)}), "flat5": flat(.05), "flat10": flat(.10), "uniform5": hierarchy(.05, five_lambda, five_reserve), "uniform10": hierarchy(.10, ten_lambda, ten_reserve)}
                for action, descriptor in protocol["methods"]["actions"].items(): choices[action] = flat(.05) if action == "A4" else hierarchy(float(descriptor["candidate_fraction"]), five_lambda, five_reserve)
                common = {"source": record["identity"], "layer": layer, "position": position, "head": head}
                for name, (ids, item_traffic) in choices.items():
                    metrics = exact_rerank_and_output(query, keys, values, ids, reference=reference); item_traffic = dict(item_traffic); item_traffic["total_kv_bytes"] = item_traffic["total_k_bytes"] + item_traffic["selected_v_fp16_bytes"]
                    methods[name].append({**common, "metrics": metrics, "traffic": item_traffic})
                if schedules:
                    for target, table in schedules.items(): methods[target].append(dict(methods[table[f"{layer}:{head}"]][-1]))
    return methods


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows: raise C4Error("cannot summarize zero observations")
    return {"observation_count": len(rows), "mean_cosine": mean(float(x["metrics"]["cosine_similarity"]) for x in rows), "mean_relative_l2": mean(float(x["metrics"]["relative_l2_error"]) for x in rows), "mean_total_kv_bytes": mean(float(x["traffic"]["total_kv_bytes"]) for x in rows)}


def _development_schedules(rows: Mapping[str, list[Mapping[str, Any]]], protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stats, groups = {}, []
    for layer in LAYERS:
        for head in HEADS:
            group = {}
            for action in ACTIONS:
                values = [r for r in rows[action] if r["layer"] == layer and r["head"] == head]
                if len(values) != 27: raise C4Error("each development action cell requires 27 observations")
                group[action] = {"relative_l2": mean(x["metrics"]["relative_l2_error"] for x in values), "cosine": mean(x["metrics"]["cosine_similarity"] for x in values), "total_kv_bytes": mean(x["traffic"]["total_kv_bytes"] for x in values)}; stats[f"{layer}:{head}:{action}"] = group[action]
            groups.append(group)
    dense, flat5 = _summary(rows["dense"])["mean_total_kv_bytes"], _summary(rows["flat5"])["mean_total_kv_bytes"]
    schedules = {}
    for target in TARGETS:
        target_mean, strict, cap = _target_cap(protocol, target, dense, flat5); found = optimize_static_schedule(groups, target_mean, strict=strict)
        if found is None: schedules[target] = {"feasible": False, "target_mean_kv_bytes": target_mean, "strict": strict, "integer_microbyte_cap": cap, "integer_microbytes_used": None, "table": None}; continue
        used, _err, _cos, actions = found; table = {f"{layer}:{head}": action for (layer, head), action in zip(((l, h) for l in LAYERS for h in HEADS), actions, strict=True)}
        if used > cap or set(table) != _action_table_keys(): raise C4Error("C4 DP violates cap/table contract")
        schedules[target] = {"feasible": True, "target_mean_kv_bytes": target_mean, "strict": strict, "integer_microbyte_cap": cap, "integer_microbytes_used": used, "table": table}
    return stats, schedules


def develop(*, capture_root: Path, capture_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise C4Error(f"refusing to overwrite result: {output}")
    preflight(); protocol = validate_protocol(); resolver = CaptureResolver(capture_root, capture_manifest, "development")
    access = TensorAccess(resolver, allowed_kind="development")
    rows = _method_rows(access, protocol)
    expected = _expected_tensor_ids("development")
    if access.opened != expected:
        raise C4Error("development tensor firewall proof differs")
    summaries = {name: _summary(value) for name, value in rows.items()}
    if any(item["observation_count"] != 4320 for item in summaries.values()): raise C4Error("development observation count differs")
    stats, schedules = _development_schedules(rows, protocol)
    result = {"schema_version": "cascadekv-phi35-8k-c4-development-v1", "protocol_sha256": sha256_path(PROTOCOL), "runtime_manifest_sha256": sha256_path(RUNTIME), "development_source_manifest_sha256": SOURCE_SHA, "capture_manifest_sha256": sha256_path(capture_manifest), "opened_development_tensor_identities": expected, "opened_holdout_tensor_identities": [], "methods": protocol["methods"], "development_metrics": summaries, "development_traffic": {name: item["mean_total_kv_bytes"] for name, item in summaries.items()}, "action_cell_statistics": stats, "schedules": schedules, "schedule_sha256": schedule_digest(schedules), "optimizer": protocol["optimizer"], "classification": None}
    atomic_json(output, result); return result


def _validate_development_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "protocol_sha256", "runtime_manifest_sha256", "development_source_manifest_sha256", "capture_manifest_sha256", "opened_development_tensor_identities", "opened_holdout_tensor_identities", "methods", "development_metrics", "development_traffic", "action_cell_statistics", "schedules", "schedule_sha256", "optimizer", "classification"}
    if set(payload) != required or payload.get("schema_version") != "cascadekv-phi35-8k-c4-development-v1" or payload.get("protocol_sha256") != sha256_path(PROTOCOL) or payload.get("runtime_manifest_sha256") != sha256_path(RUNTIME) or payload.get("development_source_manifest_sha256") != SOURCE_SHA or not isinstance(payload.get("capture_manifest_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", payload["capture_manifest_sha256"]): raise C4Error("development result provenance differs")
    protocol = validate_protocol()
    if payload.get("opened_development_tensor_identities") != _expected_tensor_ids("development") or payload.get("opened_holdout_tensor_identities") != [] or payload.get("methods") != protocol["methods"] or payload.get("optimizer") != protocol["optimizer"] or payload.get("classification") is not None: raise C4Error("development firewall/protocol result differs")
    metrics, traffic = payload["development_metrics"], payload["development_traffic"]
    if set(metrics) != set((*BASELINES, *ACTIONS)) or set(traffic) != set(metrics): raise C4Error("development method set differs")
    for name, row in metrics.items():
        if set(row) != {"observation_count", "mean_cosine", "mean_relative_l2", "mean_total_kv_bytes"} or row["observation_count"] != 4320 or traffic[name] != row["mean_total_kv_bytes"] or any(not isinstance(row[key], (int, float)) or isinstance(row[key], bool) or not math.isfinite(float(row[key])) for key in ("mean_cosine", "mean_relative_l2", "mean_total_kv_bytes")): raise C4Error("development metric differs")
    stats = payload["action_cell_statistics"]
    if set(stats) != {f"{l}:{h}:{a}" for l in LAYERS for h in HEADS for a in ACTIONS}: raise C4Error("action-cell statistics differ")
    for key, row in stats.items():
        if set(row) != {"relative_l2", "cosine", "total_kv_bytes"} or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in row.values()): raise C4Error(f"action-cell statistic differs: {key}")
    schedules = payload["schedules"]
    if list(schedules) != list(TARGETS) or payload["schedule_sha256"] != schedule_digest(schedules): raise C4Error("development schedule digest differs")
    for target in TARGETS:
        row = schedules[target]
        if set(row) != set(SCHEDULE_FIELDS): raise C4Error("schedule schema differs")
        target_mean, strict, cap = _target_cap(protocol, target, traffic["dense"], traffic["flat5"])
        if row["target_mean_kv_bytes"] != target_mean or row["strict"] is not strict or row["integer_microbyte_cap"] != cap or type(row["feasible"]) is not bool: raise C4Error("schedule target differs")
        if row["feasible"]:
            if not isinstance(row["table"], Mapping) or set(row["table"]) != _action_table_keys() or set(row["table"].values()) - set(ACTIONS) or type(row["integer_microbytes_used"]) is not int: raise C4Error("feasible schedule differs")
            used = sum(round(float(stats[f"{layer}:{head}:{row['table'][f'{layer}:{head}']}"]["total_kv_bytes"]) * 1000) for layer in LAYERS for head in HEADS)
            if row["integer_microbytes_used"] != used or used > cap or (strict and used >= math.ceil(target_mean * 160 * 1000)): raise C4Error("schedule traffic/cap differs")
        elif row["table"] is not None or row["integer_microbytes_used"] is not None: raise C4Error("infeasible schedule differs")
    return dict(payload)


def test(*, capture_root: Path, capture_manifest: Path, holdout_manifest: Path, development_result: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise C4Error(f"refusing to overwrite result: {output}")
    preflight(); development_payload = _validate_development_result(_json(development_result)); development_sha = sha256_path(development_result); holdout = _validate_holdout_manifest(_json(holdout_manifest), holdout_manifest)
    if holdout["development_result_sha256"] != development_sha or holdout["development_schedule_sha256"] != development_payload["schedule_sha256"]: raise C4Error("holdout was not selected from this exact frozen development result")
    before = development_payload["schedule_sha256"]
    resolver = CaptureResolver(capture_root, capture_manifest, "holdout", holdout_manifest=holdout_manifest, development_result_sha256=development_sha, development_schedule_sha256=development_payload["schedule_sha256"]); access = TensorAccess(resolver, allowed_kind="holdout")
    schedules = {target: development_payload["schedules"][target]["table"] for target in TARGETS if development_payload["schedules"][target]["feasible"]}
    rows = _method_rows(access, validate_protocol(), schedules)
    expected = _expected_tensor_ids("holdout", holdout)
    if access.opened != expected: raise C4Error("test tensor firewall proof differs")
    summaries = {name: _summary(values) for name, values in rows.items()}
    if any(item["observation_count"] != 1440 for item in summaries.values()): raise C4Error("holdout observation count differs")
    facts, winner = {}, None
    for target in TARGETS:
        if target not in schedules: facts[target] = {"feasible": False, "passes": False}; continue
        item = summaries[target]; row = {"feasible": True, "mean_cosine_gte": item["mean_cosine"] >= .985, "mean_relative_l2_lte": item["mean_relative_l2"] <= .120, "relative_l2_lt_uniform10": item["mean_relative_l2"] < summaries["uniform10"]["mean_relative_l2"], "modeled_kv_traffic_lt_flat5": item["mean_total_kv_bytes"] < summaries["flat5"]["mean_total_kv_bytes"]}; row["passes"] = all(value for key, value in row.items() if key not in {"feasible", "passes"}); facts[target] = row
        if winner is None and row["passes"]: winner = target
    after = schedule_digest(development_payload["schedules"])
    if before != after: raise C4Error("test mutated frozen schedules")
    result = {"schema_version": "cascadekv-phi35-8k-c4-test-v1", "development_result_sha256": development_sha, "development_schedule_sha256": development_payload["schedule_sha256"], "holdout_manifest_sha256": sha256_path(holdout_manifest), "capture_manifest_sha256": sha256_path(capture_manifest), "protocol_sha256": sha256_path(PROTOCOL), "runtime_manifest_sha256": sha256_path(RUNTIME), "opened_development_tensor_identities": [], "opened_holdout_tensor_identities": expected, "target_order": list(TARGETS), "baseline_test_metrics": {name: summaries[name] for name in BASELINES}, "target_test_metrics": {name: summaries[name] for name in TARGETS if name in summaries}, "gate_values": validate_protocol()["quality_gates"], "per_gate_pass_fail": facts, "first_passing_target": winner, "no_schedule_mutation_evidence": {"schedule_sha256_before": before, "schedule_sha256_after": after, "schedule_unchanged": True, "optimizer_rerun": False}, "classification": None}
    atomic_json(output, result); return result


def _validate_c2_qualification(path: Path) -> dict[str, Any]:
    """Bind exact observed C2 qualification without invoking C2 HEAD preflight."""
    if sha256_path(path) != QUALIFICATION_SHA: raise C4Error("C2 qualification SHA differs")
    record = _json(path)
    if record.get("backend_id") != BACKEND_ID or record.get("target") != {"model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION}: raise C4Error("C2 qualification backend/target differs")
    return record


def _capture_sources(*, kind: str, sources: list[dict[str, Any]], output: Path, qualification: Path, holdout_manifest: Path | None = None, development_result_sha256: str | None = None, development_schedule_sha256: str | None = None) -> dict[str, Any]:
    """Direct C4 provenance capture; one C2-equivalent forward per source."""
    if output.exists() and (output / "final_capture_manifest.json").exists(): raise C4Error("refusing to overwrite completed capture")
    qrecord = _validate_c2_qualification(qualification)
    # All heavyweight helpers are intentionally delayed until C4 capture mode.
    from cascadekv import phi35_kaggle_c2 as c2
    from cascadekv.phi35_8k_source_selection import load_dataset_for_spec, load_frozen_tokenizer, mechanical_proof
    import torch
    # This is C2's qualification-record validator only.  It binds the exact
    # frozen C2 tag/commit/protocol/backend provenance and deliberately does
    # not invoke C2's HEAD==C2-prep preflight.
    c2.validate_qualification(qrecord)
    tokenizer = load_frozen_tokenizer(); model = c2._load_pinned_model()
    try:
        c2.assert_capture_backend(model, qrecord)
        records = []
        for source_no, source in enumerate(sources, 1):
            spec = {key: source[key] for key in ("dataset", "config", "split", "revision", "field")}; dataset = load_dataset_for_spec(spec); proof = mechanical_proof(dataset, tokenizer, spec, int(source["dataset_index"])); value = dataset[int(source["dataset_index"])].get(source["field"])
            if not _proof_eligible(proof) or not isinstance(value, str): raise C4Error("captured source no longer mechanically eligible")
            ids = torch.tensor([tokenizer(value, add_special_tokens=True, truncation=True, max_length=8192).input_ids], dtype=torch.int64).contiguous(); input_sha = _input_hash(ids)
            expected_proof = source["proof"].get("mechanical_eligibility") if kind == "development" else source["proof"]
            if proof != expected_proof or (kind == "development" and source.get("reproof") != source.get("proof")):
                raise C4Error("captured source mechanical proof/reproof differs")
            if input_sha != source["input_ids_sha256"]:
                raise C4Error("captured source exact canonical input hash reproof differs")
            identity = _source_identity(source, kind); captured = c2.capture_five_layers_post_rope_qkv(model, ids.to(c2._first_device(model)))
            for layer in LAYERS:
                stem = f"{source['family']}_{kind}_index{source['dataset_index']}_layer{layer}"; artifact_rel, prov_rel = f"artifacts/{stem}.safetensors", f"artifacts/{stem}.provenance.json"; artifact, provenance = _safe_under(output, artifact_rel), _safe_under(output, prov_rel)
                if artifact.exists() or provenance.exists(): raise C4Error("never overwrite C4 artifact")
                c2.write_artifact(artifact, provenance, captured[layer], {"schema_version": "cascadekv-phi35-8k-c4-artifact-v1", "protocol_sha256": sha256_path(PROTOCOL), "qualification_sha256": QUALIFICATION_SHA, "backend_id": BACKEND_ID, "identity": identity, "layer": layer, "input_ids_sha256": input_sha, "target": {"model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION}, "source": _source_provenance_identity(source, kind), "source_proof": proof, "q_shape": list(SHAPE), "k_shape": list(SHAPE), "v_shape": list(SHAPE), "storage_dtype": "float16", "model_compute_dtype": "float16", "attention_implementation": "sdpa", "use_cache": False, "quantization": "none", "capture_adapter": "phi3-post-rope-qkv-v1"})
                records.append({"identity": identity, "layer": layer, "artifact_relative_path": artifact_rel, "artifact_sha256": sha256_path(artifact), "provenance_relative_path": prov_rel, "provenance_sha256": sha256_path(provenance), "input_ids_sha256": input_sha, "source": {"family": source["family"], "dataset_index": source["dataset_index"], "role": kind}}); _progress(f"captured {kind} source {source_no}/{len(sources)} layer {layer}/5")
            del captured, ids, dataset
        final = {"schema_version": "cascadekv-phi35-8k-c4-capture-v1", "kind": kind, "protocol_sha256": sha256_path(PROTOCOL), "qualification_sha256": QUALIFICATION_SHA, "backend_id": BACKEND_ID, "source_manifest_sha256": SOURCE_SHA if kind == "development" else None, "holdout_manifest_sha256": sha256_path(holdout_manifest) if holdout_manifest else None, "development_result_sha256": development_result_sha256, "development_schedule_sha256": development_schedule_sha256, "artifact_count": len(records), "forwards": len(sources), "artifacts": records}
        atomic_json(output / "final_capture_manifest.json", final); return final
    finally:
        del model


def capture_development(*, development_source_manifest: Path, output: Path, qualification: Path) -> dict[str, Any]:
    preflight(); return _capture_sources(kind="development", sources=_validate_development_manifest(_json(development_source_manifest), development_source_manifest), output=output, qualification=qualification)


def capture_holdout(*, holdout_manifest: Path, development_result: Path, output: Path, qualification: Path) -> dict[str, Any]:
    preflight(); manifest = _validate_holdout_manifest(_json(holdout_manifest), holdout_manifest); development = _validate_development_result(_json(development_result)); development_sha = sha256_path(development_result)
    if manifest["development_result_sha256"] != development_sha or manifest["development_schedule_sha256"] != development["schedule_sha256"]: raise C4Error("holdout capture development freeze binding differs")
    return _capture_sources(kind="holdout", sources=manifest["selected_sources"], output=output, qualification=qualification, holdout_manifest=holdout_manifest, development_result_sha256=development_sha, development_schedule_sha256=development["schedule_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("preflight")
    selector = sub.add_parser("select-holdout"); selector.add_argument("--development-source-manifest", type=Path, required=True); selector.add_argument("--development-result", type=Path, required=True); selector.add_argument("--output", type=Path, required=True)
    for mode in ("capture-development", "capture-holdout"):
        item = sub.add_parser(mode); item.add_argument("--output", type=Path, required=True); item.add_argument("--qualification", type=Path, required=True); item.add_argument("--development-source-manifest" if mode == "capture-development" else "--holdout-manifest", type=Path, required=True)
        if mode == "capture-holdout": item.add_argument("--development-result", type=Path, required=True)
    item = sub.add_parser("develop"); item.add_argument("--capture-root", type=Path, required=True); item.add_argument("--capture-manifest", type=Path, required=True); item.add_argument("--output", type=Path, required=True)
    item = sub.add_parser("test"); item.add_argument("--capture-root", type=Path, required=True); item.add_argument("--capture-manifest", type=Path, required=True); item.add_argument("--holdout-manifest", type=Path, required=True); item.add_argument("--development-result", type=Path, required=True); item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "preflight": result = preflight()
        elif args.mode == "select-holdout": result = select_holdout(development_source_manifest=args.development_source_manifest, development_result=args.development_result, output=args.output)
        elif args.mode == "capture-development": result = capture_development(development_source_manifest=args.development_source_manifest, output=args.output, qualification=args.qualification)
        elif args.mode == "capture-holdout": result = capture_holdout(holdout_manifest=args.holdout_manifest, development_result=args.development_result, output=args.output, qualification=args.qualification)
        elif args.mode == "develop": result = develop(capture_root=args.capture_root, capture_manifest=args.capture_manifest, output=args.output)
        else: result = test(capture_root=args.capture_root, capture_manifest=args.capture_manifest, holdout_manifest=args.holdout_manifest, development_result=args.development_result, output=args.output)
        print(canonical_json(result)); return 0
    except (C4Error, OSError, ValueError, RuntimeError) as exc:
        print(f"PHI35-8K-C4-FAILED: {exc}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
