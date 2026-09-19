"""Data-free-preparable Phi-3.5 8K C1 source-input dedup amendment.

``preflight`` is deliberately repository-only.  ``select`` is the sole mode
which imports tokenizer/dataset helpers, and is intended to run in Kaggle
against the pinned public shards.  Neither mode imports model or capture code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from cascadekv.phi35_8k_source_selection import (
    FAMILIES, IDENTITY_KIND, MINIMUM, PHASE_B_COMMIT, PHASE_B_TAG,
    PROTOCOL_SHA256, RUNTIME_MANIFEST_SHA256, TARGET_MODEL, TARGET_REVISION,
    SourceSelectionError, frozen_specs, mechanical_proof,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "cascadekv-phi35-8k-dev-sources-v2"
C1_TAG = "cascadekv-phi35-8k-sources-freeze"
C1_COMMIT = "667900a5307fe231565033a774d7788942fb869d"
C1_MANIFEST = ROOT / "results/cascadekv_phi35_8k_dev_sources.json"
C1_MANIFEST_SHA256 = "af8e805bad5c85a1f9d6ae5571b378e4df46fdb1de778a5ec7b5b68b13dc4a8e"
C2_TAG = "cascadekv-phi35-8k-kaggle-c2-prep-v2"
C2_COMMIT = "fa54b581ee5ee581dd51b8085db8b5233467450a"
C2_PROTOCOL = ROOT / "configs/cascadekv_phi35_8k_kaggle_c2_protocol.json"
C2_PROTOCOL_SHA256 = "315453e8446e8f41611fc23544a4a5093da13a7dba344dfc66a840eebf17857d"
C2_RUNTIME_MANIFEST = ROOT / "configs/cascadekv_phi35_8k_kaggle_c2_runtime_manifest.json"
C2_RUNTIME_MANIFEST_SHA256 = "a5a96915f8cab588e5d1dd6af15855d4d090366f94282022bff7bc1b856957fd"
AMENDMENT_PROTOCOL = ROOT / "configs/cascadekv_phi35_8k_source_amendment_protocol.json"
AMENDMENT_RUNTIME_MANIFEST = ROOT / "configs/cascadekv_phi35_8k_source_amendment_runtime_manifest.json"
AMENDMENT_FREEZE_TAG = "cascadekv-phi35-8k-source-dedup-amendment-prep-v1"
QWEN_RESULT = ROOT / "results/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.json"
QWEN_AUDIT = ROOT / "results/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2_audit.json"
IMMUTABLE_HASHES = {
    QWEN_RESULT: "076f8fac9f4d6b16c5acfbd4235cc26215f44af3068a0f20e16d4adfa61a2e12",
    QWEN_AUDIT: "c513e44299f07ae32ac17be92600791d3a89d1823d4e6bcff9fbbd65be51ad64",
    ROOT / "configs/cascadekv_phi35_8k_phaseb_protocol.json": PROTOCOL_SHA256,
    ROOT / "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json": RUNTIME_MANIFEST_SHA256,
    C1_MANIFEST: C1_MANIFEST_SHA256,
    C2_PROTOCOL: C2_PROTOCOL_SHA256,
    C2_RUNTIME_MANIFEST: C2_RUNTIME_MANIFEST_SHA256,
}

KAGGLE_AMENDMENT_ROOT = Path("/kaggle/working/cascadekv_phi35_8k_source_amendment")
DEFAULT_OUTPUT = KAGGLE_AMENDMENT_ROOT / "cascadekv_phi35_8k_dev_sources_v2.json"
FIXED_INDICES = {"narrative": (13, 14, 15), "report": (16, 17, 18)}
BASE_C1_UNAVAILABLE = {"narrative": tuple(range(16)), "report": tuple(range(19)), "qa": tuple(range(16))}
OBSERVED_QA_HASHES = {
    13: "6efe85ed32e8f8e28ba946fa301f65bb7559f7ed596da3f62a519833ce3a80a4",
    14: "6efe85ed32e8f8e28ba946fa301f65bb7559f7ed596da3f62a519833ce3a80a4",
    15: "29e61e009d96dc7e70261a07b197aaee190073884aaf8a8e1a53d4c1265b62b7",
}
ROLES = ("calibration_1", "calibration_2", "validation")
INCIDENT_EVIDENCE = {
    "qa_input_ids_sha256": {
        "13": "6efe85ed32e8f8e28ba946fa301f65bb7559f7ed596da3f62a519833ce3a80a4",
        "14": "6efe85ed32e8f8e28ba946fa301f65bb7559f7ed596da3f62a519833ce3a80a4",
        "15": "29e61e009d96dc7e70261a07b197aaee190073884aaf8a8e1a53d4c1265b62b7",
    },
    "qa13_qa14_identical_qkv_artifact_sha256": {
        "0": "2774937b969deecb80354c07ffdcdd36b5de9ebaf7b5532a8431247fd831dcf7",
        "8": "5d9992290a57a954a9da969cedba7df0b324e82f65eb0517d6ba2deede00e741",
        "16": "4623ff39b375cbc9d32fb04d7e940174da5b38ea2fb8fd509d4067238f38b2ee",
        "24": "975dfa94aa6f456f291c3f8ddb22a9b92faea6c5fabdccdc61d966098bf60685",
        "31": "8681007e512143e9a48b9dd1e951c9fbdb5eedcb6b69a820b1a4e86245408477",
    },
    "observation_kind": "mechanical_identity_only",
}


class SourceAmendmentError(RuntimeError):
    """The amendment's mechanical source contract failed closed."""


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", f"{ref}^{{commit}}"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceAmendmentError(f"required git reference is unavailable: {ref}") from exc


def _tracked(relative: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative],
        capture_output=True, check=False,
    ).returncode == 0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAmendmentError(f"cannot load required amendment provenance file: {path.name}") from exc
    if not isinstance(value, dict):
        raise SourceAmendmentError(f"amendment provenance root is not an object: {path.name}")
    return value


def validate_amendment_protocol(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the prospective, text-free incident and selection contract."""
    protocol = dict(_load_json(AMENDMENT_PROTOCOL) if payload is None else payload)
    required = {
        "schema_version", "purpose", "base_c1_freeze", "phase_b_freeze", "c2_v2_freeze",
        "qualification", "target", "tokenization_contract", "uniqueness_kind",
        "incident_evidence", "selection_rules", "scientific_state", "future_role_provenance",
        "kaggle_entrypoint", "immutable_prior_hashes",
    }
    if set(protocol) != required or protocol["schema_version"] != "cascadekv-phi35-8k-source-dedup-amendment-protocol-v1":
        raise SourceAmendmentError("amendment protocol schema differs")
    if protocol["purpose"] != "source-input dedup amendment only":
        raise SourceAmendmentError("amendment protocol purpose differs")
    if protocol["base_c1_freeze"] != {"tag": C1_TAG, "commit": C1_COMMIT, "manifest_sha256": C1_MANIFEST_SHA256}:
        raise SourceAmendmentError("amendment protocol C1 freeze differs")
    if protocol["phase_b_freeze"] != {"tag": PHASE_B_TAG, "commit": PHASE_B_COMMIT, "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256}:
        raise SourceAmendmentError("amendment protocol Phase-B freeze differs")
    if protocol["c2_v2_freeze"] != {"tag": C2_TAG, "commit": C2_COMMIT, "protocol_sha256": C2_PROTOCOL_SHA256, "runtime_manifest_sha256": C2_RUNTIME_MANIFEST_SHA256}:
        raise SourceAmendmentError("amendment protocol C2-v2 freeze differs")
    if protocol["qualification"] != {"sha256": "f4703bdcd477ee91f108dde4415e14831ec20a8b156fe59876d8e3a7699bc07e", "backend_id": "184109f4e8edec3c34455bd86a787e7838bef2eb75424cf0a4f57a183f313a95"}:
        raise SourceAmendmentError("amendment protocol qualification binding differs")
    if protocol["target"] != {"model": TARGET_MODEL, "model_revision": TARGET_REVISION, "tokenizer_revision": TARGET_REVISION}:
        raise SourceAmendmentError("amendment protocol target differs")
    if protocol["tokenization_contract"] != {"add_special_tokens": True, "truncation": True, "max_length": MINIMUM, "required_input_ids_length": MINIMUM, "canonicalization": "CPU contiguous int64 bytes"} or protocol["uniqueness_kind"] != "input_ids_sha256":
        raise SourceAmendmentError("amendment protocol tokenization/uniqueness differs")
    if protocol["incident_evidence"] != INCIDENT_EVIDENCE:
        raise SourceAmendmentError("amendment protocol incident evidence differs")
    expected_rules = {
        "narrative_fixed": {"indices": [13, 14, 15], "roles": list(ROLES), "duplicate_input": "fail_closed"},
        "report_fixed": {"indices": [16, 17, 18], "roles": list(ROLES), "duplicate_input": "fail_closed"},
        "qa": {"calibration_1_index": 13, "reject_duplicate_index": 14, "duplicate_of_dataset_index": 13, "calibration_2_index": 15, "validation": "first_mechanically_eligible_globally_new_exact_input_ascending_from_16_then_stop"},
        "global_selected_input_uniqueness": "all_nine_input_ids_sha256_are_unique",
    }
    if protocol["selection_rules"] != expected_rules:
        raise SourceAmendmentError("amendment protocol selection rules differ")
    if protocol["scientific_state"] != {"cascadekv_quality_metrics_observed": False, "schedule_optimized": False}:
        raise SourceAmendmentError("amendment protocol scientific state differs")
    if protocol["future_role_provenance"] != {"qa15_historical_role": "validation", "qa15_amended_role": "calibration_2", "historical_tensor_bytes_invalidated": False, "later_capture_amendment_required": "new_role_correct_provenance_wrapper_or_reference_or_recapture"}:
        raise SourceAmendmentError("amendment protocol QA15 role provenance differs")
    if protocol["kaggle_entrypoint"] != {"interpreter": "uv run python3", "module": "cascadekv.phi35_8k_source_amendment"}:
        raise SourceAmendmentError("amendment protocol Kaggle entrypoint differs")
    if protocol["immutable_prior_hashes"] != {"qwen_result_sha256": "076f8fac9f4d6b16c5acfbd4235cc26215f44af3068a0f20e16d4adfa61a2e12", "qwen_audit_sha256": "c513e44299f07ae32ac17be92600791d3a89d1823d4e6bcff9fbbd65be51ad64"}:
        raise SourceAmendmentError("amendment protocol immutable Qwen bindings differ")
    return protocol


def verify_amendment_runtime_closure() -> tuple[dict[str, str], ...]:
    """Verify the complete local selection closure, tracked and byte-exact."""
    manifest = _load_json(AMENDMENT_RUNTIME_MANIFEST)
    required = {"schema_version", "purpose", "protocol", "bound_files"}
    if set(manifest) != required or manifest["schema_version"] != 1 or manifest["purpose"] != "Phi-3.5 8K source-input dedup amendment local selection closure":
        raise SourceAmendmentError("amendment runtime-manifest schema differs")
    protocol = manifest["protocol"]
    if protocol != {"path": str(AMENDMENT_PROTOCOL.relative_to(ROOT)), "sha256": sha256_path(AMENDMENT_PROTOCOL)}:
        raise SourceAmendmentError("amendment runtime-manifest protocol binding differs")
    entries = manifest["bound_files"]
    if not isinstance(entries, list) or not entries:
        raise SourceAmendmentError("amendment runtime-manifest binds no files")
    seen: set[str] = set()
    verified = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"} or not isinstance(entry["path"], str) or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise SourceAmendmentError("amendment runtime-manifest bound-file schema differs")
        relative = entry["path"]
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise SourceAmendmentError("amendment runtime-manifest path is duplicate or unsafe")
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file() or not _tracked(relative):
            raise SourceAmendmentError(f"amendment runtime bound file is absent or untracked: {relative}")
        actual = sha256_path(path)
        if actual != entry["sha256"]:
            raise SourceAmendmentError(f"amendment runtime bound file hash differs: {relative}")
        verified.append({"path": relative, "sha256": actual})
    return tuple(verified)


def amendment_freeze_binding() -> dict[str, str]:
    """Return the required real-run tag/HEAD identity and amendment hashes."""
    try:
        commit = git_commit(AMENDMENT_FREEZE_TAG)
    except SourceAmendmentError as exc:
        raise SourceAmendmentError("amendment freeze tag is unavailable") from exc
    if commit != git_commit("HEAD"):
        raise SourceAmendmentError("amendment freeze tag does not resolve exactly to HEAD")
    return {
        "tag": AMENDMENT_FREEZE_TAG,
        "commit": commit,
        "amendment_protocol_sha256": sha256_path(AMENDMENT_PROTOCOL),
        "amendment_runtime_manifest_sha256": sha256_path(AMENDMENT_RUNTIME_MANIFEST),
    }


def preflight(*, require_amendment_tag: bool = True) -> dict[str, str]:
    """Verify local immutable bindings without tokenizer, network, data, or model access."""
    if git_commit(C1_TAG) != C1_COMMIT:
        raise SourceAmendmentError("C1 freeze tag does not resolve to its immutable commit")
    if git_commit(C2_TAG) != C2_COMMIT:
        raise SourceAmendmentError("C2-v2 freeze tag does not resolve to its immutable commit")
    for path, expected in IMMUTABLE_HASHES.items():
        if sha256_path(path) != expected:
            raise SourceAmendmentError(f"immutable prior SHA256 differs: {path.name}")
    validate_amendment_protocol()
    verify_amendment_runtime_closure()
    if require_amendment_tag:
        amendment_freeze_binding()
    return {
        "base_c1_tag": C1_TAG,
        "base_c1_commit": C1_COMMIT,
        "base_c1_manifest_sha256": C1_MANIFEST_SHA256,
        "phase_b_protocol_sha256": PROTOCOL_SHA256,
        "phase_b_runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "c2_v2_protocol_sha256": C2_PROTOCOL_SHA256,
        "c2_v2_runtime_manifest_sha256": C2_RUNTIME_MANIFEST_SHA256,
        "amendment_freeze_tag_required": str(require_amendment_tag).lower(),
    }


def canonical_input_ids_sha256(input_ids: Any) -> str:
    """Hash precisely the CPU-contiguous native ``int64`` model-input bytes."""
    try:
        import numpy as np
        canonical = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int64))
    except Exception as exc:  # pragma: no cover - dependency/type error boundary
        raise SourceAmendmentError("input ids cannot be canonicalized as contiguous int64") from exc
    if canonical.dtype != np.dtype("int64") or not canonical.flags.c_contiguous:
        raise SourceAmendmentError("input ids are not canonical CPU contiguous int64")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _tokenize_exact(tokenizer: Any, value: str) -> tuple[int, str]:
    encoded = tokenizer(value, add_special_tokens=True, truncation=True, max_length=MINIMUM)
    ids = encoded.input_ids
    length = len(ids)
    if length != MINIMUM:
        raise SourceAmendmentError("eligible candidate did not construct an exact 8192-token input")
    return length, canonical_input_ids_sha256(ids)


def inspect_candidate(dataset: Any, tokenizer: Any, spec: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Perform only the frozen mechanical field/length/input-identity predicate."""
    eligibility = mechanical_proof(dataset, tokenizer, spec, index)
    if not (eligibility["field_exists"] and eligibility["is_python_string"] and eligibility["at_least_8192"]):
        return {"mechanical_eligibility": eligibility}
    value = dataset[index].get(spec["field"])
    if not isinstance(value, str):
        raise SourceAmendmentError("mechanical proof and source field type disagree")
    length, digest = _tokenize_exact(tokenizer, value)
    return {
        "mechanical_eligibility": eligibility,
        "input_ids_length": length,
        "input_ids_sha256": digest,
    }


def _accepted(family: str, index: int, role: str, proof: Mapping[str, Any]) -> dict[str, Any]:
    return {"family": family, "dataset_index": index, "role": role, "proof": dict(proof)}


def _select_fixed_family(
    family: str, spec: Mapping[str, Any], tokenizer: Any, dataset_loader: Callable[[Mapping[str, Any]], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Reprove the original Narrative/Report identities; replacement is forbidden."""
    dataset = dataset_loader(spec)
    selected: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index, role in zip(FIXED_INDICES[family], ROLES, strict=True):
        proof = inspect_candidate(dataset, tokenizer, spec, index)
        if "input_ids_sha256" not in proof:
            raise SourceAmendmentError(f"{family} frozen identity {index} is no longer mechanically eligible")
        digest = proof["input_ids_sha256"]
        if digest in seen:
            raise SourceAmendmentError(f"{family} frozen identities duplicate exact model inputs")
        seen[digest] = index
        selected.append(_accepted(family, index, role, proof))
    return selected, [], FIXED_INDICES[family][-1]


def _select_qa(
    spec: Mapping[str, Any], tokenizer: Any, dataset_loader: Callable[[Mapping[str, Any]], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Select three ascending unique QA inputs, stopping at the third acceptance."""
    dataset = dataset_loader(spec)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    index = int(spec["first_permitted_index"])
    while index < len(dataset):
        proof = inspect_candidate(dataset, tokenizer, spec, index)
        digest = proof.get("input_ids_sha256")
        if digest is None:
            rejected.append({"dataset_index": index, "rejection_reason": "ineligible", "proof": proof})
        else:
            expected = OBSERVED_QA_HASHES.get(index)
            if expected is not None and digest != expected:
                raise SourceAmendmentError(f"QA incident reproof differs at index {index}")
            duplicate_of = seen.get(digest)
            if duplicate_of is not None:
                rejected.append({
                    "dataset_index": index, "rejection_reason": "duplicate_input",
                    "duplicate_of_dataset_index": duplicate_of, "input_ids_sha256": digest,
                    "proof": proof,
                })
            else:
                selected.append(_accepted("qa", index, ROLES[len(selected)], proof))
                seen[digest] = index
                if len(selected) == 3:
                    return selected, rejected, index
        index += 1
    raise SourceAmendmentError("insufficient unique eligible QA identities from the frozen frontier")


def _reproof_selected(
    choices: list[dict[str, Any]], specs: Mapping[str, Mapping[str, Any]], tokenizer: Any,
    dataset_loader: Callable[[Mapping[str, Any]], Any],
) -> list[dict[str, Any]]:
    out = []
    for choice in choices:
        dataset = dataset_loader(specs[choice["family"]])
        proof = inspect_candidate(dataset, tokenizer, specs[choice["family"]], choice["dataset_index"])
        if proof != choice["proof"]:
            raise SourceAmendmentError("selected source reproof differs from original mechanical proof")
        out.append(proof)
    return out


def _source_record(choice: Mapping[str, Any], spec: Mapping[str, Any], reproof: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": choice["family"], "dataset": spec["dataset"], "config": spec["config"],
        "split": spec["split"], "revision": spec["revision"], "field": spec["field"],
        "identity_kind": IDENTITY_KIND, "dataset_index": choice["dataset_index"],
        "role": choice["role"], "input_ids_sha256": choice["proof"]["input_ids_sha256"],
        "proof": choice["proof"], "reproof": dict(reproof),
    }


def build_manifest(
    specs: Mapping[str, Mapping[str, Any]], selections: Mapping[str, list[dict[str, Any]]],
    rejections: Mapping[str, list[dict[str, Any]]], reproofs: Mapping[str, list[dict[str, Any]]],
    last_inspected: Mapping[str, int],
) -> dict[str, Any]:
    amendment_freeze = amendment_freeze_binding()
    sources = [
        _source_record(choice, specs[family], reproof)
        for family in FAMILIES
        for choice, reproof in zip(selections[family], reproofs[family], strict=True)
    ]
    unavailable = {}
    for family in FAMILIES:
        base = list(BASE_C1_UNAVAILABLE[family])
        inspected = sorted({item["dataset_index"] for item in selections[family] + rejections[family]})
        unavailable[family] = {
            "base_c1_unavailable_indices": base,
            "amendment_inspected_indices": inspected,
            "indices": sorted(set(base + inspected)),
            "next_untouched_index": last_inspected[family] + 1,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "C1 source-identity amendment: exact model-input deduplication only; no quality evaluation",
        "amendment_reason": "C1 unique dataset indices did not ensure unique exact 8192-token model inputs",
        "base_c1_freeze": {"tag": C1_TAG, "commit": C1_COMMIT, "manifest_sha256": C1_MANIFEST_SHA256},
        "phase_b_freeze": {"tag": PHASE_B_TAG, "commit": PHASE_B_COMMIT, "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256},
        "c2_v2_freeze": {"tag": C2_TAG, "commit": C2_COMMIT, "protocol_sha256": C2_PROTOCOL_SHA256, "runtime_manifest_sha256": C2_RUNTIME_MANIFEST_SHA256},
        "amendment_freeze": amendment_freeze,
        "target": {"model": TARGET_MODEL, "model_revision": TARGET_REVISION, "tokenizer_revision": TARGET_REVISION},
        "tokenization_contract": {"add_special_tokens": True, "truncation": True, "max_length": MINIMUM, "required_input_ids_length": MINIMUM, "canonicalization": "CPU contiguous int64 bytes"},
        "identity_kind": IDENTITY_KIND,
        "uniqueness_kind": "input_ids_sha256",
        "source_text_stored": False,
        "source_specs": {family: {key: specs[family][key] for key in ("dataset", "config", "split", "revision", "field", "first_permitted_index")} for family in FAMILIES},
        "selected_sources": sources,
        "rejected_sources": {family: rejections[family] for family in FAMILIES},
        "last_inspected_index": dict(last_inspected),
        "unavailable_inventory": unavailable,
        "reproof_status": {"all_selected_sources_equal": True, "selected_source_count": len(sources), "global_input_ids_sha256_unique": True},
    }


def _assert_no_raw_text(value: Any) -> None:
    if isinstance(value, Mapping):
        if {"raw_text", "source_text", "document_text", "context_text"}.intersection(value):
            raise SourceAmendmentError("manifest schema permits raw source text")
        for nested in value.values():
            _assert_no_raw_text(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_raw_text(nested)


def validate_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the text-free v2 schema, dedup invariant, and stop boundaries."""
    _assert_no_raw_text(payload)
    required = {"schema_version", "purpose", "amendment_reason", "base_c1_freeze", "phase_b_freeze", "c2_v2_freeze", "amendment_freeze", "target", "tokenization_contract", "identity_kind", "uniqueness_kind", "source_text_stored", "source_specs", "selected_sources", "rejected_sources", "last_inspected_index", "unavailable_inventory", "reproof_status"}
    if not isinstance(payload, Mapping) or set(payload) != required or payload["schema_version"] != SCHEMA_VERSION:
        raise SourceAmendmentError("v2 manifest schema differs")
    if payload["base_c1_freeze"] != {"tag": C1_TAG, "commit": C1_COMMIT, "manifest_sha256": C1_MANIFEST_SHA256}:
        raise SourceAmendmentError("v2 manifest does not bind immutable C1")
    if payload["phase_b_freeze"] != {"tag": PHASE_B_TAG, "commit": PHASE_B_COMMIT, "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256}:
        raise SourceAmendmentError("v2 manifest Phase-B binding differs")
    if payload["c2_v2_freeze"] != {"tag": C2_TAG, "commit": C2_COMMIT, "protocol_sha256": C2_PROTOCOL_SHA256, "runtime_manifest_sha256": C2_RUNTIME_MANIFEST_SHA256}:
        raise SourceAmendmentError("v2 manifest C2-v2 binding differs")
    if payload["amendment_freeze"] != amendment_freeze_binding():
        raise SourceAmendmentError("v2 manifest amendment freeze binding differs")
    if payload["target"] != {"model": TARGET_MODEL, "model_revision": TARGET_REVISION, "tokenizer_revision": TARGET_REVISION}:
        raise SourceAmendmentError("v2 target tokenizer binding differs")
    if payload["tokenization_contract"] != {"add_special_tokens": True, "truncation": True, "max_length": MINIMUM, "required_input_ids_length": MINIMUM, "canonicalization": "CPU contiguous int64 bytes"}:
        raise SourceAmendmentError("v2 exact tokenization contract differs")
    if payload["identity_kind"] != IDENTITY_KIND or payload["uniqueness_kind"] != "input_ids_sha256" or payload["source_text_stored"] is not False:
        raise SourceAmendmentError("v2 identity, uniqueness, or text-free contract differs")
    specs = frozen_specs()
    expected_specs = {family: {key: specs[family][key] for key in ("dataset", "config", "split", "revision", "field", "first_permitted_index")} for family in FAMILIES}
    if payload["source_specs"] != expected_specs:
        raise SourceAmendmentError("v2 source specs differ from frozen Phase-B specs")
    sources = payload["selected_sources"]
    if not isinstance(sources, list) or len(sources) != 9:
        raise SourceAmendmentError("v2 manifest must select exactly nine sources")
    by_family = {family: [] for family in FAMILIES}
    hashes: set[str] = set()
    source_keys = {"family", "dataset", "config", "split", "revision", "field", "identity_kind", "dataset_index", "role", "input_ids_sha256", "proof", "reproof"}
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != source_keys or source.get("family") not in FAMILIES:
            raise SourceAmendmentError("selected-source schema is malformed")
        family = source["family"]
        if any(source[key] != specs[family][key] for key in ("dataset", "config", "split", "revision", "field")):
            raise SourceAmendmentError("selected source differs from pinned source spec")
        if source["identity_kind"] != IDENTITY_KIND or not isinstance(source["dataset_index"], int) or source["dataset_index"] < specs[family]["first_permitted_index"]:
            raise SourceAmendmentError("selected source identity differs from the frozen frontier")
        proof = source["proof"]
        eligibility = proof.get("mechanical_eligibility") if isinstance(proof, Mapping) else None
        expected_eligibility_keys = {"field_exists", "is_python_string", "character_count", "bounded_frozen_tokenizer_length", "at_least_8192"}
        if source["reproof"] != proof or not isinstance(proof, Mapping) or set(proof) != {"mechanical_eligibility", "input_ids_length", "input_ids_sha256"} or proof.get("input_ids_length") != MINIMUM or proof.get("input_ids_sha256") != source["input_ids_sha256"] or not isinstance(eligibility, Mapping) or set(eligibility) != expected_eligibility_keys or eligibility.get("at_least_8192") is not True:
            raise SourceAmendmentError("selected source input proof/reproof differs")
        digest = source["input_ids_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in hashes:
            raise SourceAmendmentError("selected input hashes are not globally unique")
        hashes.add(digest); by_family[family].append(source)
    for family, items in by_family.items():
        ordered = sorted(items, key=lambda item: item["dataset_index"])
        if len(ordered) != 3 or items != ordered or [item["role"] for item in ordered] != list(ROLES):
            raise SourceAmendmentError("family role ordering is not three ascending unique acceptances")
        if family in FIXED_INDICES and [item["dataset_index"] for item in items] != list(FIXED_INDICES[family]):
            raise SourceAmendmentError("Narrative/Report replacement is forbidden")
    rejected = payload["rejected_sources"]
    if not isinstance(rejected, Mapping) or set(rejected) != set(FAMILIES):
        raise SourceAmendmentError("rejection family set differs")
    for family in FAMILIES:
        if not isinstance(rejected[family], list):
            raise SourceAmendmentError("rejection inventory is malformed")
        selected_indices = {item["dataset_index"] for item in by_family[family]}
        accepted_hashes = {item["input_ids_sha256"]: item["dataset_index"] for item in by_family[family]}
        rejected_indices: set[int] = set()
        for item in rejected[family]:
            if not isinstance(item, Mapping) or not isinstance(item.get("dataset_index"), int) or item["dataset_index"] < specs[family]["first_permitted_index"] or item["dataset_index"] in selected_indices or item["dataset_index"] in rejected_indices:
                raise SourceAmendmentError("rejection identity is malformed")
            rejected_indices.add(item["dataset_index"])
            if item.get("rejection_reason") == "ineligible":
                if set(item) != {"dataset_index", "rejection_reason", "proof"} or set(item["proof"]) != {"mechanical_eligibility"}:
                    raise SourceAmendmentError("ineligible rejection schema permits source text")
            elif item.get("rejection_reason") == "duplicate_input":
                if set(item) != {"dataset_index", "rejection_reason", "duplicate_of_dataset_index", "input_ids_sha256", "proof"} or item.get("duplicate_of_dataset_index") != accepted_hashes.get(item.get("input_ids_sha256")):
                    raise SourceAmendmentError("duplicate-input binding differs")
            else:
                raise SourceAmendmentError("unknown rejection reason")
    qa_duplicates = [item for item in rejected["qa"] if item.get("rejection_reason") == "duplicate_input"]
    if not any(item.get("dataset_index") == 14 and item.get("duplicate_of_dataset_index") == 13 and item.get("input_ids_sha256") == OBSERVED_QA_HASHES[14] for item in qa_duplicates):
        raise SourceAmendmentError("QA index 14 duplicate binding is absent")
    last = payload["last_inspected_index"]
    inventories = payload["unavailable_inventory"]
    qa_items = by_family["qa"]
    if [(item["dataset_index"], item["role"]) for item in qa_items[:2]] != [(13, "calibration_1"), (15, "calibration_2")] or qa_items[0]["input_ids_sha256"] != OBSERVED_QA_HASHES[13] or qa_items[1]["input_ids_sha256"] != OBSERVED_QA_HASHES[15] or qa_items[2]["dataset_index"] <= 15:
        raise SourceAmendmentError("QA 13/14/15 amendment transition differs")
    for family in FAMILIES:
        if last[family] != max(item["dataset_index"] for item in by_family[family]):
            raise SourceAmendmentError("last inspected index must equal validation acceptance")
        all_inspected = {item["dataset_index"] for item in by_family[family] + rejected[family]}
        if any(index > last[family] for index in all_inspected):
            raise SourceAmendmentError("identity inspected after validation stop")
        inventory = inventories.get(family) if isinstance(inventories, Mapping) else None
        if not isinstance(inventory, Mapping):
            raise SourceAmendmentError("complete unavailable inventory is malformed")
        if family == "qa" and 14 not in inventory.get("indices", []):
            raise SourceAmendmentError("QA duplicate identity is absent from unavailable inventory")
        expected_base = list(BASE_C1_UNAVAILABLE[family])
        expected_indices = sorted(set(expected_base + list(all_inspected)))
        if inventory.get("base_c1_unavailable_indices") != expected_base or inventory.get("amendment_inspected_indices") != sorted(all_inspected) or inventory.get("indices") != expected_indices or inventory.get("next_untouched_index") != last[family] + 1:
            raise SourceAmendmentError("unavailable inventory or next frontier differs")
    if payload["reproof_status"] != {"all_selected_sources_equal": True, "selected_source_count": 9, "global_input_ids_sha256_unique": True}:
        raise SourceAmendmentError("v2 reproof status differs")


def require_amendment_output(path: Path) -> Path:
    resolved = path.resolve()
    root = KAGGLE_AMENDMENT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise SourceAmendmentError(f"amendment output must be below {KAGGLE_AMENDMENT_ROOT}")
    return resolved


def select(*, output: Path = DEFAULT_OUTPUT, tokenizer: Any | None = None, dataset_loader: Callable[[Mapping[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Kaggle-only selection.  This is never the CLI default operation."""
    preflight()
    from cascadekv.phi35_8k_source_selection import load_dataset_for_spec, load_frozen_tokenizer
    target = require_amendment_output(output)
    if target.exists():
        raise SourceAmendmentError("refusing to overwrite an amendment manifest")
    specs = frozen_specs()
    frozen_tokenizer = tokenizer if tokenizer is not None else load_frozen_tokenizer()
    loader = dataset_loader if dataset_loader is not None else load_dataset_for_spec
    selections: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, list[dict[str, Any]]] = {}
    last: dict[str, int] = {}
    for family in ("narrative", "report"):
        selections[family], rejected[family], last[family] = _select_fixed_family(family, specs[family], frozen_tokenizer, loader)
    selections["qa"], rejected["qa"], last["qa"] = _select_qa(specs["qa"], frozen_tokenizer, loader)
    reproofs = {family: _reproof_selected(selections[family], specs, frozen_tokenizer, loader) for family in FAMILIES}
    manifest = build_manifest(specs, selections, rejected, reproofs, last)
    validate_manifest(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("preflight", help="repository-only; no tokenizer, network, dataset, or model access")
    select_parser = modes.add_parser("select", help="Kaggle-only pinned tokenizer/dataset selection; no model weights")
    select_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            print(json.dumps(preflight(), sort_keys=True))
        else:
            select(output=args.output)
            print(f"wrote text-free amended source manifest: {args.output}")
    except (SourceAmendmentError, SourceSelectionError, OSError, ValueError, IndexError) as exc:
        print(f"PHI35-8K-SOURCE-DEDUP-AMENDMENT-INVALID: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
