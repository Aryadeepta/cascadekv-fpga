"""Fail-closed, minimal Phi-3.5 8K source-dedup capture supplement.

``preflight`` is repository-only.  ``capture`` is explicitly gated and may
load exactly the amended QA16 source and perform one model forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cascadekv_phi35_8k_capture_amendment_protocol.json"
RUNTIME_MANIFEST = ROOT / "configs/cascadekv_phi35_8k_capture_amendment_runtime_manifest.json"
SUPPLEMENT_TAG = "cascadekv-phi35-8k-capture-amendment-prep-v1"
AMENDMENT_TAG = "cascadekv-phi35-8k-source-dedup-amendment-prep-v1"
AMENDMENT_COMMIT = "a26c76de596d6f0074556ae630679ffca0021e69"
AMENDED_SOURCE_SHA256 = "31f1d14d4b33b6e5b6a48d34e294f0effbab2eeadf6c4d4eabfabe16ae879478"
QUALIFICATION_SHA256 = "f4703bdcd477ee91f108dde4415e14831ec20a8b156fe59876d8e3a7699bc07e"
APPROVAL_SHA256 = "dc9b36abd206bd84cf02d2d9494dd0d955ae61fa915deb32fcf1e7ef1675551f"
BASE_FINAL_SHA256 = "e9208dd6d50eaf3273b8365b1e075d2a25cfa65ddb923715a341630ff5be2250"
BACKEND_ID = "184109f4e8edec3c34455bd86a787e7838bef2eb75424cf0a4f57a183f313a95"
SCHEMA = "cascadekv-phi35-8k-amended-capture-v1"
REFERENCE_SCHEMA = "cascadekv-phi35-8k-amended-artifact-reference-v1"
LAYERS = (0, 8, 16, 24, 31)
SHAPE = [1, 32, 8192, 96]
ARTIFACT_BYTES = 150994944
QA16_HASH = "d9bcac44581785b0d180fdcd46cbcc5dcecd50e0e2c4285d81d949d1b35bc3ac"
DEFAULT_OUTPUT = Path("/kaggle/working/cascadekv_phi35_8k_capture_amendment")
DEFAULT_BASE = Path("/kaggle/working/cascadekv_phi35_8k_capture")
SOURCE_AMENDMENT_ROOT = Path("/kaggle/working/cascadekv_phi35_8k_source_amendment")
QA15_TENSORS = {0: "c496e5a8883ab2b194752e6853ea84be52045cb83c3de7217db3e7e187c52c7b", 8: "b52f1b101726c02a0d9e48b2eda688ef4a34fa13c040d23b594517cca842f76b", 16: "2810ba39836521e74072605941a2ec006ece540c5c09db32309dd22612e28c36", 24: "ad172cef3a0f26072790fa2f337a02dc52624e79b418a575ba173ef002f5611b", 31: "e3c21514c00aa0e625483519b21327f3c9b9a61c77dab94075907046ece44db0"}
QA15_PROVENANCE = {0: "24a59131130c880ad45e81b2bb4093d61d0aa7f174cadbe89ecc4162f5da4ad6", 8: "f4394d86d20750671078f3d0d0d587c6d04f98e440f272fe359f44776a479ec2", 16: "da92d043a4f611a061c213e7075cfdeb1d892dfad6cf35f886edbe47fb242beb", 24: "d25313d3d951dfdbb2b3eb46a789d34cb7b3fd56531a37ad6fc0b89f09d79414", 31: "03b3b9fcdddea6e0bad92bb57ea9ae988decc574bf3b39722e7eea4f4c78c092"}

CAPTURE_CONTRACT = {
    "adapter": "phi3-post-rope-qkv-v1",
    "attention": "sdpa",
    "compute_dtype": "float16",
    "layers": list(LAYERS),
    "one_forward": True,
    "qa16": {"family": "qa", "dataset_index": 16, "role": "validation", "input_ids_sha256": QA16_HASH},
    "quantization": "none",
    "shape": SHAPE,
    "storage_dtype": "float16",
    "target": {"model": "microsoft/Phi-3.5-mini-instruct", "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77"},
    "use_cache": False,
}
QA15_HISTORICAL_ARTIFACTS = {
    str(layer): {"tensor_sha256": QA15_TENSORS[layer], "provenance_sha256": QA15_PROVENANCE[layer]}
    for layer in LAYERS
}
FREEZE_BINDINGS = {
    "c1": {"tag": "cascadekv-phi35-8k-sources-freeze", "commit": "667900a5307fe231565033a774d7788942fb869d", "manifest_sha256": "af8e805bad5c85a1f9d6ae5571b378e4df46fdb1de778a5ec7b5b68b13dc4a8e"},
    "c2_v2": {"tag": "cascadekv-phi35-8k-kaggle-c2-prep-v2", "commit": "fa54b581ee5ee581dd51b8085db8b5233467450a", "protocol_sha256": "315453e8446e8f41611fc23544a4a5093da13a7dba344dfc66a840eebf17857d", "runtime_manifest_sha256": "a5a96915f8cab588e5d1dd6af15855d4d090366f94282022bff7bc1b856957fd"},
    "phase_b": {"tag": "cascadekv-phi35-8k-phaseb-protocol-freeze", "commit": "d94cff8e1e11048ff3d8edc9fb1c605dfe422d6e", "protocol_sha256": "0e3a2e7b51e01c64dca292c7dd667ed9899c5bacc08e470e2b19e12433363c8b", "runtime_manifest_sha256": "e4e10f7659180471eb791eeafa97e8f525b6c14a3ff311798349c07438ec7c92"},
    "source_amendment": {"tag": AMENDMENT_TAG, "commit": AMENDMENT_COMMIT, "protocol_sha256": "f43b2311d0decf2a579bf886848572bf53d58ab98e058b1da1be5a2912b85e8e", "runtime_manifest_sha256": "d35817df8a7c8a7a48539cac252344dbe4f94cf97b9a3a50aa307e88a3640edf"},
    "supplement": {"tag": SUPPLEMENT_TAG},
}


class SupplementError(RuntimeError):
    """A supplement provenance, isolation, or execution gate failed closed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(canonical_json(value) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def git_commit(ref: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{ref}^{{commit}}"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SupplementError(f"required git reference is unavailable: {ref}") from exc


def _tracked(relative: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative], capture_output=True, check=False).returncode == 0


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplementError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SupplementError(f"JSON root must be an object: {path}")
    return value


def validate_protocol(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    protocol = dict(_json(PROTOCOL) if payload is None else payload)
    required = {"schema_version", "amended_source_manifest", "base_capture", "capture", "freeze_bindings", "logical_capture", "qa15_historical_artifacts", "scientific_state"}
    if set(protocol) != required or protocol["schema_version"] != "cascadekv-phi35-8k-capture-amendment-protocol-v1":
        raise SupplementError("supplement protocol schema differs")
    if protocol["amended_source_manifest"] != {"schema": "cascadekv-phi35-8k-dev-sources-v2", "sha256": AMENDED_SOURCE_SHA256}:
        raise SupplementError("amended source binding differs")
    if protocol["base_capture"] != {"approval_sha256": APPROVAL_SHA256, "artifact_payload_bytes": ARTIFACT_BYTES, "backend_id": BACKEND_ID, "final_capture_manifest_sha256": BASE_FINAL_SHA256, "qualification_sha256": QUALIFICATION_SHA256, "schema": "cascadekv-phi35-8k-kaggle-capture-v2"}:
        raise SupplementError("base capture bindings differ")
    if protocol["capture"] != CAPTURE_CONTRACT:
        raise SupplementError("frozen QA16 capture contract differs")
    if protocol["qa15_historical_artifacts"] != QA15_HISTORICAL_ARTIFACTS:
        raise SupplementError("QA15 historical artifact bindings differ")
    if protocol["logical_capture"] != {"logical_artifacts": 45, "logical_tensor_payload_bytes": 6794772480, "new_tensor_payload_bytes": 754974720, "new_tensors": 5, "qa14_logical_artifacts": 0, "reused_tensors": 40}:
        raise SupplementError("logical capture accounting differs")
    if protocol["freeze_bindings"] != FREEZE_BINDINGS:
        raise SupplementError("freeze bindings differ")
    if protocol["scientific_state"] != {"cascadekv_quality_metrics_observed": False, "schedule_optimized": False}:
        raise SupplementError("scientific state differs")
    return protocol


def verify_runtime_closure() -> tuple[dict[str, str], ...]:
    manifest = _json(RUNTIME_MANIFEST)
    if set(manifest) != {"schema_version", "purpose", "protocol", "bound_files"} or manifest["schema_version"] != 1:
        raise SupplementError("runtime manifest schema differs")
    if manifest["protocol"] != {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256_path(PROTOCOL)}:
        raise SupplementError("runtime protocol binding differs")
    entries = manifest["bound_files"]
    if not isinstance(entries, list) or not entries:
        raise SupplementError("runtime closure is empty")
    seen: set[str] = set(); verified = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise SupplementError("runtime bound-file schema differs")
        relative, expected = entry["path"], entry["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise SupplementError("runtime bound-file path differs")
        seen.add(relative); path = ROOT / relative
        if not path.is_file() or not _tracked(relative) or sha256_path(path) != expected:
            raise SupplementError(f"runtime bound file fails closed: {relative}")
        verified.append({"path": relative, "sha256": expected})
    return tuple(verified)


def preflight(*, require_supplement_tag: bool = True) -> dict[str, Any]:
    """Repository-only closure validation; imports no model/data/tokenizer stack."""
    if git_commit(AMENDMENT_TAG) != AMENDMENT_COMMIT:
        raise SupplementError("source-amendment freeze tag differs")
    if require_supplement_tag and git_commit(SUPPLEMENT_TAG) != git_commit("HEAD"):
        raise SupplementError("supplement freeze tag must resolve exactly to HEAD")
    # This is still data-free and brings the frozen amendment's transitive C1,
    # Phase-B, C2-v2, and immutable historical-hash checks into this gate.
    from cascadekv import phi35_8k_source_amendment as amendment
    amendment.preflight(require_amendment_tag=False)
    validate_protocol(); closure = verify_runtime_closure()
    return {"protocol_path": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha256_path(PROTOCOL), "runtime_manifest_path": str(RUNTIME_MANIFEST.relative_to(ROOT)), "runtime_manifest_sha256": sha256_path(RUNTIME_MANIFEST), "bound_files": closure, "supplement_tag_required": require_supplement_tag}


def _amended_sources(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_path(path) != AMENDED_SOURCE_SHA256:
        raise SupplementError("source manifest SHA256 differs")
    payload = _json(path)
    # The frozen validator has a historical HEAD gate; this caller separately
    # proves its immutable tag/commit above and supplies that frozen binding.
    from cascadekv import phi35_8k_source_amendment as amendment
    original = amendment.amendment_freeze_binding
    try:
        amendment.amendment_freeze_binding = lambda: {"tag": AMENDMENT_TAG, "commit": AMENDMENT_COMMIT, "amendment_protocol_sha256": "f43b2311d0decf2a579bf886848572bf53d58ab98e058b1da1be5a2912b85e8e", "amendment_runtime_manifest_sha256": "d35817df8a7c8a7a48539cac252344dbe4f94cf97b9a3a50aa307e88a3640edf"}
        amendment.validate_manifest(payload)
    finally:
        amendment.amendment_freeze_binding = original
    sources = payload["selected_sources"]
    qa = [item for item in sources if item["family"] == "qa"]
    identities = [(item["dataset_index"], item["role"], item["input_ids_sha256"]) for item in qa]
    if identities != [(13, "calibration_1", "6efe85ed32e8f8e28ba946fa301f65bb7559f7ed596da3f62a519833ce3a80a4"), (15, "calibration_2", "29e61e009d96dc7e70261a07b197aaee190073884aaf8a8e1a53d4c1265b62b7"), (16, "validation", QA16_HASH)]:
        raise SupplementError("amended QA identities differ")
    rejected = payload["rejected_sources"]["qa"]
    if not any(item.get("dataset_index") == 14 and item.get("rejection_reason") == "duplicate_input" and item.get("duplicate_of_dataset_index") == 13 for item in rejected):
        raise SupplementError("QA14 rejection binding differs")
    if payload["unavailable_inventory"]["qa"].get("next_untouched_index") != 17:
        raise SupplementError("QA next untouched frontier differs")
    return payload, qa[-1]


def _safe_relative(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SupplementError("unsafe artifact path")
    return candidate


def verify_base_capture(base: Path, qualification: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Verify history without tensor interpretation.

    Historical tensor files are read only as raw bytes for bytewise SHA-256
    integrity verification.  They are never deserialized/interpreted as
    tensors, linked, copied, or rewritten by this supplement.
    """
    base = base.resolve()
    if sha256_path(qualification) != QUALIFICATION_SHA256 or qualification.resolve().parent != base:
        raise SupplementError("qualification path/hash differs from immutable base capture")
    approval = base / "approval.json"; final_path = base / "final_capture_manifest.json"
    if sha256_path(approval) != APPROVAL_SHA256 or sha256_path(final_path) != BASE_FINAL_SHA256:
        raise SupplementError("base approval or final manifest SHA differs")
    final = _json(final_path)
    if final.get("schema_version") != "cascadekv-phi35-8k-kaggle-capture-v2" or final.get("qualification_sha256") != QUALIFICATION_SHA256 or final.get("backend_id") != BACKEND_ID:
        raise SupplementError("base final manifest binding differs")
    records = final.get("complete_artifacts")
    if not isinstance(records, list) or len(records) != 45:
        raise SupplementError("base capture must contain exactly 45 artifacts")
    indexed: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"} or not isinstance(record["path"], str) or not isinstance(record["sha256"], str):
            raise SupplementError("base artifact entry differs")
        relative = str(_safe_relative(record["path"]))
        if relative in indexed or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise SupplementError("base artifact entry is duplicate or malformed")
        artifact = base / relative
        if not artifact.is_file() or sha256_path(artifact) != record["sha256"]:
            raise SupplementError("historical artifact missing or hash differs")
        indexed[relative] = record["sha256"]
    qa15: dict[int, dict[str, Any]] = {}
    for layer in LAYERS:
        tensor_rel = f"artifacts/qa_validation_index15_layer{layer}.safetensors"
        prov_rel = f"artifacts/qa_validation_index15_layer{layer}.provenance.json"
        if indexed.get(tensor_rel) != QA15_TENSORS[layer]:
            raise SupplementError("QA15 historical tensor binding differs")
        provenance = base / prov_rel
        if not provenance.is_file() or sha256_path(provenance) != QA15_PROVENANCE[layer]:
            raise SupplementError("QA15 historical provenance binding differs")
        detail = _json(provenance); source = detail.get("source_identity", {})
        if detail.get("artifact_sha256") != QA15_TENSORS[layer] or detail.get("backend_id") != BACKEND_ID or detail.get("frozen_qualification_sha256") != QUALIFICATION_SHA256 or detail.get("input_ids_sha256") != "29e61e009d96dc7e70261a07b197aaee190073884aaf8a8e1a53d4c1265b62b7" or detail.get("layer") != layer or source.get("family") != "qa" or source.get("dataset_index") != 15 or source.get("role") != "validation":
            raise SupplementError("QA15 old provenance must remain historical validation")
        qa15[layer] = {"tensor_path": tensor_rel, "provenance_path": prov_rel, "tensor_sha256": QA15_TENSORS[layer], "provenance_sha256": QA15_PROVENANCE[layer]}
    return final, qa15


def _output_isolated(output: Path, base: Path) -> Path:
    resolved, frozen_base, amendment_root = output.resolve(), base.resolve(), SOURCE_AMENDMENT_ROOT.resolve()
    if any(resolved == root or root in resolved.parents or resolved in root.parents for root in (frozen_base, amendment_root)):
        raise SupplementError("supplement output may not overlap an immutable capture/amendment root")
    return resolved


def _reference(layer: int, item: Mapping[str, Any], protocol_sha: str, runtime_sha: str) -> dict[str, Any]:
    return {"schema_version": REFERENCE_SCHEMA, "amended_source_role": "calibration_2", "family": "qa", "dataset_index": 15, "layer": layer, "input_ids_sha256": "29e61e009d96dc7e70261a07b197aaee190073884aaf8a8e1a53d4c1265b62b7", "artifact_origin": "historical_base_capture", "historical_artifact_relative_path": item["tensor_path"], "historical_artifact_sha256": item["tensor_sha256"], "historical_provenance_relative_path": item["provenance_path"], "historical_provenance_sha256": item["provenance_sha256"], "historical_role": "validation", "amended_role": "calibration_2", "tensor_bytes_reused_unchanged": True, "tensor_bytes_recaptured": False, "amended_source_manifest_sha256": AMENDED_SOURCE_SHA256, "c1_freeze": {"tag": "cascadekv-phi35-8k-sources-freeze", "commit": "667900a5307fe231565033a774d7788942fb869d", "manifest_sha256": "af8e805bad5c85a1f9d6ae5571b378e4df46fdb1de778a5ec7b5b68b13dc4a8e"}, "phase_b_freeze": {"protocol_sha256": "0e3a2e7b51e01c64dca292c7dd667ed9899c5bacc08e470e2b19e12433363c8b", "runtime_manifest_sha256": "e4e10f7659180471eb791eeafa97e8f525b6c14a3ff311798349c07438ec7c92"}, "c2_v2_freeze": {"protocol_sha256": "315453e8446e8f41611fc23544a4a5093da13a7dba344dfc66a840eebf17857d", "runtime_manifest_sha256": "a5a96915f8cab588e5d1dd6af15855d4d090366f94282022bff7bc1b856957fd"}, "source_amendment_freeze": {"tag": AMENDMENT_TAG, "commit": AMENDMENT_COMMIT, "protocol_sha256": "f43b2311d0decf2a579bf886848572bf53d58ab98e058b1da1be5a2912b85e8e", "runtime_manifest_sha256": "d35817df8a7c8a7a48539cac252344dbe4f94cf97b9a3a50aa307e88a3640edf"}, "supplement_protocol_sha256": protocol_sha, "supplement_runtime_manifest_sha256": runtime_sha, "base_final_capture_manifest_sha256": BASE_FINAL_SHA256, "base_qualification_sha256": QUALIFICATION_SHA256, "backend_id": BACKEND_ID, "target": "microsoft/Phi-3.5-mini-instruct", "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77", "capture_adapter": "phi3-post-rope-qkv-v1", "storage_dtype": "float16", "shape": {"q": SHAPE, "k": SHAPE, "v": SHAPE}}


def _reference_path(output: Path, layer: int) -> Path:
    return output / "artifacts" / f"qa_calibration_2_index15_layer{layer}.reference.provenance.json"


def write_references(output: Path, qa15: Mapping[int, Mapping[str, Any]], protocol_sha: str, runtime_sha: str) -> dict[int, dict[str, Any]]:
    result = {}
    for layer in LAYERS:
        path, expected = _reference_path(output, layer), _reference(layer, qa15[layer], protocol_sha, runtime_sha)
        if path.exists():
            if _json(path) != expected: raise SupplementError("QA15 role reference mismatch; refusing overwrite")
        else: atomic_json(path, expected)
        result[layer] = {"path": str(path.relative_to(output)), "sha256": sha256_path(path)}
    return result


def _qa16_input(source: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The sole source operation: one QA16 row, exact 8192 IDs, then stop."""
    from cascadekv.phi35_8k_source_selection import load_dataset_for_spec, load_frozen_tokenizer
    from cascadekv.phi35_8k_source_amendment import inspect_candidate
    import torch
    spec = {key: source[key] for key in ("dataset", "config", "split", "revision", "field")}
    dataset = load_dataset_for_spec(spec); tokenizer = load_frozen_tokenizer()
    proof = inspect_candidate(dataset, tokenizer, spec, 16)
    if proof != source["proof"] or source["reproof"] != source["proof"] or proof.get("input_ids_sha256") != QA16_HASH:
        raise SupplementError("QA16 reproof/hash differs")
    value = dataset[16].get(source["field"])
    encoded = tokenizer(value, add_special_tokens=True, truncation=True, max_length=8192)
    ids = torch.tensor([encoded.input_ids], dtype=torch.int64).contiguous()
    if tuple(ids.shape) != (1, 8192) or hashlib.sha256(ids.numpy().tobytes()).hexdigest() != QA16_HASH:
        raise SupplementError("QA16 exact input construction differs")
    return ids, proof


def _supplement_provenance(source: Mapping[str, Any], proof: Mapping[str, Any], layer: int, qualification: Mapping[str, Any], qualification_sha: str, model: Any, protocol_sha: str, runtime_sha: str) -> dict[str, Any]:
    from cascadekv import phi35_kaggle_c2 as c2
    return {"schema_version": "cascadekv-phi35-8k-capture-amendment-artifact-v1", "supplement_protocol_sha256": protocol_sha, "supplement_runtime_manifest_sha256": runtime_sha, "amended_source_manifest_sha256": AMENDED_SOURCE_SHA256, "source_amendment_freeze": {"tag": AMENDMENT_TAG, "commit": AMENDMENT_COMMIT, "protocol_sha256": "f43b2311d0decf2a579bf886848572bf53d58ab98e058b1da1be5a2912b85e8e", "runtime_manifest_sha256": "d35817df8a7c8a7a48539cac252344dbe4f94cf97b9a3a50aa307e88a3640edf"}, "c1_freeze": {"tag": "cascadekv-phi35-8k-sources-freeze", "commit": "667900a5307fe231565033a774d7788942fb869d", "manifest_sha256": "af8e805bad5c85a1f9d6ae5571b378e4df46fdb1de778a5ec7b5b68b13dc4a8e"}, "phase_b_freeze": {"protocol_sha256": "0e3a2e7b51e01c64dca292c7dd667ed9899c5bacc08e470e2b19e12433363c8b", "runtime_manifest_sha256": "e4e10f7659180471eb791eeafa97e8f525b6c14a3ff311798349c07438ec7c92"}, "c2_v2_freeze": {"protocol_sha256": "315453e8446e8f41611fc23544a4a5093da13a7dba344dfc66a840eebf17857d", "runtime_manifest_sha256": "a5a96915f8cab588e5d1dd6af15855d4d090366f94282022bff7bc1b856957fd"}, "base_final_capture_manifest_sha256": BASE_FINAL_SHA256, "frozen_qualification_sha256": qualification_sha, "backend_id": qualification["backend_id"], "target": {"model": "microsoft/Phi-3.5-mini-instruct", "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77", "model_config_sha256": c2._config_sha(model)}, "source_identity": {key: source[key] for key in ("family", "dataset", "config", "split", "revision", "field", "identity_kind", "dataset_index", "role")}, "source_proof": dict(proof), "source_reproof": dict(proof), "input_ids_sha256": QA16_HASH, "layer": layer, "q_shape": SHAPE, "k_shape": SHAPE, "v_shape": SHAPE, "storage_dtype": "float16", "model_compute_dtype": "float16", "attention_implementation": "sdpa", "use_cache": False, "quantization": "none", "resolved_hf_device_map": c2._resolved_map(model), "capture_adapter": "phi3-post-rope-qkv-v1", "software_versions": c2._versions(), "tensor_payload_bytes": ARTIFACT_BYTES}


def _logical_manifest(base_final: Mapping[str, Any], refs: Mapping[int, Mapping[str, Any]], qa16: Mapping[int, Mapping[str, Any]], protocol_sha: str, runtime_sha: str) -> dict[str, Any]:
    entries = []
    for item in base_final["complete_artifacts"]:
        path = item["path"]
        if path.startswith("artifacts/qa_calibration_2_index14_"): continue
        origin = "historical_role_rebound" if path.startswith("artifacts/qa_validation_index15_") else "historical_unchanged"
        entry = {"origin": origin, "base_artifact_relative_path": path, "base_artifact_sha256": item["sha256"]}
        layer = next((layer for layer in LAYERS if path.endswith(f"layer{layer}.safetensors")), None)
        if origin == "historical_role_rebound": entry.update({"historical_provenance_relative_path": f"artifacts/qa_validation_index15_layer{layer}.provenance.json", "historical_provenance_sha256": QA15_PROVENANCE[layer], "role_reference_relative_path": refs[layer]["path"], "role_reference_sha256": refs[layer]["sha256"]})
        entries.append(entry)
    for layer in LAYERS: entries.append({"origin": "supplement_capture", "supplement_artifact_relative_path": qa16[layer]["path"], "supplement_artifact_sha256": qa16[layer]["sha256"], "supplement_provenance_relative_path": qa16[layer]["provenance_path"], "supplement_provenance_sha256": qa16[layer]["provenance_sha256"]})
    if len(entries) != 45 or sum(item["origin"] == "historical_unchanged" for item in entries) != 35 or sum(item["origin"] == "historical_role_rebound" for item in entries) != 5 or sum(item["origin"] == "supplement_capture" for item in entries) != 5:
        raise SupplementError("logical artifact inventory differs")
    return {"schema_version": SCHEMA, "capture_result": "COMPLETE", "amended_source_manifest_sha256": AMENDED_SOURCE_SHA256, "source_amendment_freeze": {"tag": AMENDMENT_TAG, "commit": AMENDMENT_COMMIT, "protocol_sha256": "f43b2311d0decf2a579bf886848572bf53d58ab98e058b1da1be5a2912b85e8e", "runtime_manifest_sha256": "d35817df8a7c8a7a48539cac252344dbe4f94cf97b9a3a50aa307e88a3640edf"}, "supplement_freeze": {"tag": SUPPLEMENT_TAG, "commit": git_commit(SUPPLEMENT_TAG)}, "supplement_protocol_sha256": protocol_sha, "supplement_runtime_manifest_sha256": runtime_sha, "base_final_capture_manifest_sha256": BASE_FINAL_SHA256, "base_qualification_sha256": QUALIFICATION_SHA256, "backend_id": BACKEND_ID, "target": {"model": "microsoft/Phi-3.5-mini-instruct", "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77"}, "logical_source_roles": {"qa13": "calibration_1", "qa15": "calibration_2", "qa16": "validation", "qa14": "rejected_duplicate_input"}, "logical_artifact_count": 45, "logical_tensor_payload_bytes": 6794772480, "reused_tensor_count": 40, "new_tensor_count": 5, "new_tensor_payload_bytes": 754974720, "artifacts": entries}


def _validated_completed_record(c2: Any, artifact: Path, provenance: Path, expected: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Return a completed QA16 record only after frozen C2 semantic validation."""
    artifact_sha = sha256_path(artifact)
    verified = c2.validate_artifact(artifact, provenance, {**expected, "artifact_sha256": artifact_sha})
    if verified.get("artifact_sha256") != artifact_sha:
        raise SupplementError("C2 artifact validator returned a mismatched artifact SHA256")
    return {
        "path": str(artifact.relative_to(output)),
        "sha256": artifact_sha,
        "provenance_path": str(provenance.relative_to(output)),
        "provenance_sha256": sha256_path(provenance),
    }


def capture(*, source_manifest: Path, base_capture_root: Path, qualification: Path, output: Path) -> dict[str, Any]:
    pre = preflight(); output = _output_isolated(output, base_capture_root)
    source_payload, qa16_source = _amended_sources(source_manifest)
    base_final, qa15 = verify_base_capture(base_capture_root, qualification)
    from cascadekv import phi35_kaggle_c2 as c2
    record, qualification_sha = c2.load_approved_qualification(qualification, base_capture_root / "approval.json")
    if qualification_sha != QUALIFICATION_SHA256 or record["backend_id"] != BACKEND_ID: raise SupplementError("qualification reproof differs")
    output.mkdir(parents=True, exist_ok=True); refs = write_references(output, qa15, pre["protocol_sha256"], pre["runtime_manifest_sha256"])
    ids, proof = _qa16_input(qa16_source); model = c2._load_pinned_model()
    try:
        c2.assert_capture_backend(model, record)
        expected = {layer: _supplement_provenance(qa16_source, proof, layer, record, qualification_sha, model, pre["protocol_sha256"], pre["runtime_manifest_sha256"]) for layer in LAYERS}
        absent, completed = set(), {}
        for layer in LAYERS:
            artifact = output / "artifacts" / f"qa_validation_index16_layer{layer}.safetensors"; provenance = artifact.with_suffix(".provenance.json")
            if artifact.exists() or provenance.exists():
                completed[layer] = _validated_completed_record(c2, artifact, provenance, expected[layer], output)
            else: absent.add(layer)
        if absent:
            captured = c2.capture_five_layers_post_rope_qkv(model, ids.to(c2._first_device(model)))
            for layer in absent:
                artifact = output / "artifacts" / f"qa_validation_index16_layer{layer}.safetensors"; provenance = artifact.with_suffix(".provenance.json")
                c2.write_artifact(artifact, provenance, captured[layer], expected[layer])
                completed[layer] = _validated_completed_record(c2, artifact, provenance, expected[layer], output)
        if set(completed) != set(LAYERS): raise SupplementError("all five QA16 artifacts must pass C2 validation")
        progress = {"schema_version": SCHEMA, "complete_new_artifacts": [completed[layer] for layer in LAYERS], "forwards_this_invocation": 1 if absent else 0}; atomic_json(output / "supplement_progress_manifest.json", progress)
        final = _logical_manifest(base_final, refs, completed, pre["protocol_sha256"], pre["runtime_manifest_sha256"]); atomic_json(output / "final_amended_capture_manifest.json", final); return final
    finally:
        del model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("preflight", help="repository-only; no network, tokenizer, dataset, or model")
    capture_parser = modes.add_parser("capture", help="gated QA16-only Kaggle supplement capture")
    capture_parser.add_argument("--source-manifest", type=Path, required=True); capture_parser.add_argument("--base-capture-root", type=Path, required=True); capture_parser.add_argument("--qualification", type=Path, required=True); capture_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = preflight() if args.mode == "preflight" else capture(source_manifest=args.source_manifest, base_capture_root=args.base_capture_root, qualification=args.qualification, output=args.output)
        print(canonical_json(result)); return 0
    except (SupplementError, OSError, ValueError, RuntimeError) as exc:
        print(f"PHI35-8K-CAPTURE-AMENDMENT-FAILED: {exc}"); return 1


if __name__ == "__main__": raise SystemExit(main())
