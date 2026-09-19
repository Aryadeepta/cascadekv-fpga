"""Fail-closed, Kaggle-only Phase-C2 preparation for Phi-3.5 8K.

This entry point has three explicit modes.  ``preflight`` reads only local
provenance.  ``qualify`` loads the pinned model and uses synthetic valid token
ids only.  ``capture`` is intentionally gated by a separately reviewed,
frozen qualification record; it is never the default action.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from cascadekv.model_agnostic import PHI35_GEOMETRY, Phi3CaptureAdapter, serialize_qkv_for_cache
from cascadekv.phi35_8k_source_selection import (
    FAMILIES, MINIMUM, PHASE_B_COMMIT, PHASE_B_TAG, PROTOCOL_SHA256,
    RUNTIME_MANIFEST_SHA256, TARGET_MODEL, TARGET_REVISION, load_dataset_for_spec,
    load_frozen_tokenizer, mechanical_proof, validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
C1_TAG = "cascadekv-phi35-8k-sources-freeze"
C1_COMMIT = "667900a5307fe231565033a774d7788942fb869d"
C1_MANIFEST = ROOT / "results/cascadekv_phi35_8k_dev_sources.json"
C1_MANIFEST_SHA256 = "af8e805bad5c85a1f9d6ae5571b378e4df46fdb1de778a5ec7b5b68b13dc4a8e"
C2_PREP_TAG = "cascadekv-phi35-8k-kaggle-c2-prep-v2"
C2_PROTOCOL = "cascadekv-phi35-8k-kaggle-c2-protocol-v2"
C2_PROTOCOL_FILE = ROOT / "configs/cascadekv_phi35_8k_kaggle_c2_protocol.json"
C2_PROTOCOL_SHA256 = "315453e8446e8f41611fc23544a4a5093da13a7dba344dfc66a840eebf17857d"
C2_RUNTIME_MANIFEST = ROOT / "configs/cascadekv_phi35_8k_kaggle_c2_runtime_manifest.json"
QUALIFICATION_SCHEMA = "cascadekv-phi35-8k-kaggle-qualification-v2"
CAPTURE_SCHEMA = "cascadekv-phi35-8k-kaggle-capture-v2"
APPROVAL_SCHEMA = "cascadekv-phi35-8k-qualification-approval-v2"
LAYERS = (0, 8, 16, 24, 31)
LENGTHS = (512, 2048, 8192)
SHAPE = (1, 32, 8192, 96)
ARTIFACT_PAYLOAD_BYTES = 3 * 1 * 32 * 8192 * 96 * 2
TOTAL_ARTIFACT_PAYLOAD_BYTES = 45 * ARTIFACT_PAYLOAD_BYTES
DEFAULT_OUTPUT = Path("/kaggle/working/cascadekv_phi35_8k_capture")
# Recorded provenance policy.  Explicit ``device_map`` controls placement; this
# setting does not itself prove activation headroom.
MAX_MEMORY = {0: "13GiB", 1: "13GiB"}
QWEN_RESULT = ROOT / "results/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.json"
QWEN_AUDIT = ROOT / "results/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2_audit.json"
IMMUTABLE_HASHES = {
    QWEN_RESULT: "076f8fac9f4d6b16c5acfbd4235cc26215f44af3068a0f20e16d4adfa61a2e12",
    QWEN_AUDIT: "c513e44299f07ae32ac17be92600791d3a89d1823d4e6bcff9fbbd65be51ad64",
    ROOT / "configs/cascadekv_phi35_8k_phaseb_protocol.json": PROTOCOL_SHA256,
    ROOT / "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json": RUNTIME_MANIFEST_SHA256,
    C1_MANIFEST: C1_MANIFEST_SHA256,
}


class C2Error(RuntimeError):
    """A C2 backend, provenance, or freeze gate failed closed."""


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
        raise C2Error(f"required git reference is unavailable: {ref}") from exc


def c1_manifest() -> dict[str, Any]:
    if sha256_path(C1_MANIFEST) != C1_MANIFEST_SHA256:
        raise C2Error("C1 manifest SHA256 differs")
    try:
        manifest = json.loads(C1_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise C2Error("cannot read frozen C1 manifest") from exc
    validate_manifest(manifest)
    expected = {"narrative": [13, 14, 15], "report": [16, 17, 18], "qa": [13, 14, 15]}
    actual = {family: [] for family in FAMILIES}
    for item in manifest["selected_sources"]:
        actual[item["family"]].append(item["dataset_index"])
    if actual != expected:
        raise C2Error("C1 selected identities differ")
    return manifest


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def validate_c2_protocol() -> dict[str, Any]:
    """Validate the exact frozen C2 package contract without model/data access."""
    try:
        protocol = json.loads(C2_PROTOCOL_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise C2Error("cannot read C2 protocol") from exc
    if not isinstance(protocol, dict):
        raise C2Error("C2 protocol root is not an object")
    expected = {
        "schema_version": C2_PROTOCOL,
        "target": {"model": TARGET_MODEL, "revision": TARGET_REVISION},
        "geometry": {"context": 8192, "q_heads": 32, "kv_heads": 32, "head_dim": 96, "layers": list(LAYERS)},
        "c1_freeze": {"tag": C1_TAG, "commit": C1_COMMIT, "source_manifest_path": _relative(C1_MANIFEST), "source_manifest_sha256": C1_MANIFEST_SHA256},
        "phase_b_freeze": {"tag": PHASE_B_TAG, "commit": PHASE_B_COMMIT, "protocol_path": "configs/cascadekv_phi35_8k_phaseb_protocol.json", "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_path": "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json", "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256},
        "capture_semantics": {"adapter_identity": Phi3CaptureAdapter.identity, "q_k": "post-RoPE", "v": "attention projection boundary"},
        "capture": {"forbidden_until_separate_frozen_qualification_approval": True, "forwards_per_complete_run": 9, "layers_per_forward": list(LAYERS), "persistent_artifacts": 45, "source_manifest_sha256": C1_MANIFEST_SHA256},
        "candidate_backend": {"accelerator": "NVIDIA T4 x2", "cuda_devices": 2, "compute_dtype": "float16", "attention_implementation": "sdpa", "use_cache": False, "quantization": "none", "device_map": {"model.embed_tokens": 0, "model.layers.0-15": 0, "model.layers.16-31": 1, "model.norm": 1, "lm_head": 1}, "max_memory": {"0": "13GiB", "1": "13GiB"}, "max_memory_semantics": "configured memory ceiling/policy recorded for provenance; explicit device_map controls placement and qualification peak-memory measurements establish fit"},
    }
    if protocol != expected:
        raise C2Error("C2 protocol differs from the frozen C1/Phase-B/capture/backend contract")
    return protocol


def verify_c2_runtime_manifest() -> tuple[dict[str, str], ...]:
    """Verify C2's v2 closure without changing the immutable Phase-A/B verifier."""
    try:
        manifest = json.loads(C2_RUNTIME_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise C2Error("cannot read C2 runtime manifest") from exc
    expected_state = {
        "phi_source_identities_selected": True,
        "phi_source_identity_manifest_sha256": C1_MANIFEST_SHA256,
        "phi_source_semantics_inspected": False,
        "phi_model_weights_loaded": False,
        "phi_model_outputs_observed": False,
        "phi_qkv_captured": False,
        "phi_quality_metrics_observed": False,
        "schedule_optimized": False,
    }
    expected_routing = {
        "status": "FROZEN_IN_PHASE_B",
        "protocol_path": "configs/cascadekv_phi35_8k_phaseb_protocol.json",
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime_manifest_path": "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json",
        "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
    }
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2 or manifest.get("target_family") != "phi3" or manifest.get("target_model") != {"name": TARGET_MODEL, "revision": TARGET_REVISION} or manifest.get("scientific_state") != expected_state or manifest.get("routing_protocol") != expected_routing:
        raise C2Error("C2 runtime manifest scientific state or routing binding differs")
    entries = manifest.get("bound_files")
    if not isinstance(entries, list) or not entries:
        raise C2Error("C2 runtime manifest binds no files")
    verified: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise C2Error("C2 runtime manifest has malformed bound-file entry")
        relative, expected = entry["path"], entry["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise C2Error("C2 runtime manifest has unsafe bound-file entry")
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file():
            raise C2Error(f"C2 bound runtime source is absent: {relative}")
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative], text=True, capture_output=True, check=False)
        if tracked.returncode != 0:
            raise C2Error(f"C2 bound runtime source is untracked: {relative}")
        actual = sha256_path(path)
        if actual != expected:
            raise C2Error(f"C2 bound runtime source hash mismatch: {relative}")
        verified.append({"path": relative, "sha256": actual})
    return tuple(verified)


def preflight(*, require_c2_tag: bool = True) -> dict[str, Any]:
    """Validate only repository-local closure; this never imports HF or data code."""
    if git_commit(C1_TAG) != C1_COMMIT:
        raise C2Error("C1 freeze tag does not resolve to the frozen commit")
    if require_c2_tag and git_commit(C2_PREP_TAG) != git_commit("HEAD"):
        raise C2Error("HEAD is not the required C2-prep freeze tag commit")
    if sha256_path(C2_PROTOCOL_FILE) != C2_PROTOCOL_SHA256:
        raise C2Error("C2 protocol SHA256 differs")
    validate_c2_protocol()
    verify_c2_runtime_manifest()
    for path, expected in IMMUTABLE_HASHES.items():
        if sha256_path(path) != expected:
            raise C2Error(f"immutable prior SHA256 differs: {path.name}")
    manifest = c1_manifest()
    return {
        "c2_protocol": C2_PROTOCOL, "c2_protocol_sha256": C2_PROTOCOL_SHA256,
        "c2_runtime_manifest_sha256": sha256_path(C2_RUNTIME_MANIFEST),
        "c1_tag": C1_TAG, "c1_commit": C1_COMMIT,
        "c1_manifest_sha256": C1_MANIFEST_SHA256, "selected_source_count": len(manifest["selected_sources"]),
        "phase_b_protocol_sha256": PROTOCOL_SHA256,
        "phase_b_runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "next_untouched_frontiers": {"narrative": 16, "report": 19, "qa": 16},
        "artifact_payload_bytes": ARTIFACT_PAYLOAD_BYTES,
        "total_artifact_payload_bytes": TOTAL_ARTIFACT_PAYLOAD_BYTES,
        "environment_capabilities": {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        },
    }


def require_durable_output(path: Path) -> Path:
    root = DEFAULT_OUTPUT.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise C2Error(f"durable C2 output must be below {DEFAULT_OUTPUT}")
    return resolved


def require_t4x2(torch_module: Any = torch) -> list[dict[str, Any]]:
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 2:
        raise C2Error("canonical qualification requires exactly two CUDA GPUs")
    devices = []
    for index in range(2):
        name = str(torch_module.cuda.get_device_name(index))
        if "T4" not in name.upper():
            raise C2Error(f"canonical qualification requires NVIDIA T4 x2, got {name!r}")
        properties = torch_module.cuda.get_device_properties(index)
        devices.append({"index": index, "name": name, "total_vram_bytes": int(properties.total_memory)})
    return devices


def deterministic_device_map() -> dict[str, int]:
    mapping = {"model.embed_tokens": 0}
    mapping.update({f"model.layers.{layer}": 0 if layer < 16 else 1 for layer in range(32)})
    mapping.update({"model.norm": 1, "lm_head": 1})
    return mapping


def _resolved_map(model: Any) -> dict[str, int]:
    raw = getattr(model, "hf_device_map", None)
    if not isinstance(raw, Mapping) or not raw:
        raise C2Error("Accelerate did not expose a resolved hf_device_map")
    resolved: dict[str, int] = {}
    for name, device in raw.items():
        if not isinstance(name, str) or isinstance(device, bool) or not isinstance(device, int) or device not in (0, 1):
            raise C2Error("noncanonical CPU/disk or non-T4 device placement is forbidden")
        resolved[name] = device
    if set(resolved.values()) != {0, 1}:
        raise C2Error("model is not sharded across both T4 devices")
    return dict(sorted(resolved.items()))


def _model_layers(model: Any) -> Any:
    core = getattr(model, "model", model)
    layers = getattr(core, "layers", None)
    if layers is None or len(layers) != 32:
        raise C2Error("Phi model layer boundary differs from the frozen 32-layer contract")
    return layers


def _first_device(model: Any) -> torch.device:
    device_map = _resolved_map(model)
    first = device_map.get("model.embed_tokens", 0)
    return torch.device(f"cuda:{first}")


def synthetic_input_ids(length: int, vocab_size: int) -> torch.Tensor:
    if length not in LENGTHS or not isinstance(vocab_size, int) or vocab_size < 2:
        raise C2Error("synthetic sequence length or model vocabulary is invalid")
    # Every id is in [0, vocab_size); no out-of-vocabulary sentinel is used.
    return (torch.arange(length, dtype=torch.long).unsqueeze(0) % vocab_size).contiguous()


def _check_triplet(triplet: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, expected: tuple[int, ...]) -> None:
    if len(triplet) != 3:
        raise C2Error("capture adapter did not produce Q/K/V")
    for tensor in triplet:
        if tuple(tensor.shape) != expected or tensor.dtype != torch.float16 or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise C2Error("capture shape, dtype, or CPU-contiguity contract differs")


def capture_five_layers_post_rope_qkv(model: Any, input_ids: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One model forward with five simultaneous frozen Phase-A adapter hooks."""
    layers = _model_layers(model)
    captured: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    adapter = Phi3CaptureAdapter()
    handles = []
    for layer_index in LAYERS:
        attention = layers[layer_index].self_attn
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], *, index: int = layer_index) -> None:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            positions = kwargs.get("position_embeddings")
            if hidden is None or positions is None or index in captured:
                raise C2Error("Phi attention hook boundary/provenance differs")
            q, k, v = adapter.capture(module, hidden, positions)
            captured[index] = serialize_qkv_for_cache(q, k, v, storage_dtype="float16")
        handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(LAYERS):
        raise C2Error("not all five capture hooks ran during the sole forward")
    return captured


def _versions() -> dict[str, str]:
    import accelerate
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "accelerate": accelerate.__version__}


def _config_sha(model: Any) -> str:
    return hashlib.sha256(model.config.to_json_string(use_diff=False).encode("utf-8")).hexdigest()


def _load_pinned_model() -> Any:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, revision=TARGET_REVISION, token=False, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map=deterministic_device_map(), max_memory=MAX_MEMORY,
        low_cpu_mem_usage=True,
    ).eval()
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise C2Error("SDPA was not selected by the loaded model")
    if any(parameter.dtype != torch.float16 for parameter in model.parameters()):
        raise C2Error("model computation weights are not uniformly FP16")
    _resolved_map(model)
    return model


def qualification_backend_id(record: Mapping[str, Any]) -> str:
    """Identity of execution backend only, not the complete scientific package.

    It covers target/config, FP16/SDPA/no-cache/no-quantization execution,
    CUDA and software versions, T4 identities/VRAM, resolved device map, and
    the recorded max-memory policy.  The exact qualification JSON SHA binds
    this backend to C1, Phase-B, C2 package, and capture-adapter provenance.
    """
    bound = {key: record[key] for key in (
        "target", "model_config_sha256", "model_compute_dtype", "attention_implementation",
        "use_cache", "quantization", "cuda_version", "gpu_count", "gpu_names",
        "vram_per_gpu_bytes", "software_versions", "resolved_hf_device_map", "max_memory",
    )}
    return hashlib.sha256(canonical_json(bound).encode()).hexdigest()


def qualify(output: Path) -> dict[str, Any]:
    output = require_durable_output(output)
    pre = preflight()
    devices = require_t4x2()
    model = _load_pinned_model()
    try:
        config_sha = _config_sha(model)
        shape_proof: dict[str, Any] = {}
        peak_cuda_memory_by_length: dict[str, list[dict[str, int]]] = {}
        for length in LENGTHS:
            ids = synthetic_input_ids(length, int(model.config.vocab_size)).to(_first_device(model))
            # Reset both devices after model placement.  The ensuing peaks include
            # resident weights plus this sequence's execution allocations only.
            for device_index in range(2):
                torch.cuda.reset_peak_memory_stats(device_index)
            if length == 8192:
                captured = capture_five_layers_post_rope_qkv(model, ids)
                for layer, triplet in captured.items():
                    _check_triplet(triplet, expected=SHAPE)
                    del triplet
                shape_proof = {str(layer): {"q_shape": list(SHAPE), "k_shape": list(SHAPE), "v_shape": list(SHAPE)} for layer in LAYERS}
                del captured
            else:
                with torch.inference_mode():
                    model(input_ids=ids, use_cache=False)
            measurements = []
            for device_index in range(2):
                total = int(devices[device_index]["total_vram_bytes"])
                allocated = int(torch.cuda.max_memory_allocated(device_index))
                reserved = int(torch.cuda.max_memory_reserved(device_index))
                measurement = {"index": device_index, "total_vram_bytes": total,
                               "allocated_bytes": allocated, "reserved_bytes": reserved}
                if reserved <= total:
                    measurement["headroom_reserved_bytes"] = total - reserved
                measurements.append(measurement)
            peak_cuda_memory_by_length[str(length)] = measurements
            del ids
            gc.collect()
        result: dict[str, Any] = {
            "schema_version": QUALIFICATION_SCHEMA, "qualification_result": "QUALIFIED",
            "repo_capture_prep": {"tag": C2_PREP_TAG, "commit": git_commit(C2_PREP_TAG)},
            "c2_protocol_binding": {"path": _relative(C2_PROTOCOL_FILE), "sha256": C2_PROTOCOL_SHA256},
            "c2_runtime_manifest_binding": {"path": _relative(C2_RUNTIME_MANIFEST), "sha256": sha256_path(C2_RUNTIME_MANIFEST)},
            "c1_manifest_binding": {"sha256": C1_MANIFEST_SHA256, "tag": C1_TAG, "commit": C1_COMMIT},
            "phase_b_freeze": {"protocol_path": "configs/cascadekv_phi35_8k_phaseb_protocol.json", "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_path": "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json", "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256},
            "target": {"model": TARGET_MODEL, "revision": TARGET_REVISION, "tokenizer_revision": TARGET_REVISION},
            "model_config_sha256": config_sha, "model_compute_dtype": "float16",
            "attention_implementation": "sdpa", "use_cache": False, "quantization": "none",
            "cuda_version": str(torch.version.cuda), "gpu_count": 2,
            "gpu_names": [item["name"] for item in devices],
            "vram_per_gpu_bytes": [item["total_vram_bytes"] for item in devices],
            "software_versions": _versions(), "resolved_hf_device_map": _resolved_map(model),
            "max_memory": {str(key): value for key, value in MAX_MEMORY.items()},
            "synthetic_sequence_lengths": list(LENGTHS), "capture_shape_proof_8192": shape_proof,
            "peak_memory_reset_device_indices": [0, 1],
            "peak_cuda_memory_by_length": peak_cuda_memory_by_length,
            "capture_adapter": Phi3CaptureAdapter.identity,
        }
        result["backend_id"] = qualification_backend_id(result)
        atomic_json(output, result)
        return result
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_qualification(record: Mapping[str, Any], *, expected_c2_commit: str | None = None) -> None:
    required = {"schema_version", "qualification_result", "repo_capture_prep", "c2_protocol_binding", "c2_runtime_manifest_binding", "c1_manifest_binding", "phase_b_freeze", "target", "model_config_sha256", "model_compute_dtype", "attention_implementation", "use_cache", "quantization", "cuda_version", "gpu_count", "gpu_names", "vram_per_gpu_bytes", "software_versions", "resolved_hf_device_map", "max_memory", "synthetic_sequence_lengths", "capture_shape_proof_8192", "peak_memory_reset_device_indices", "peak_cuda_memory_by_length", "capture_adapter", "backend_id"}
    if not isinstance(record, Mapping) or set(record) != required:
        raise C2Error("qualification schema is incomplete or permits unbound data")
    if record["schema_version"] != QUALIFICATION_SCHEMA or record["qualification_result"] != "QUALIFIED":
        raise C2Error("qualification did not pass")
    expected_commit = git_commit(C2_PREP_TAG) if expected_c2_commit is None else expected_c2_commit
    if record["repo_capture_prep"] != {"tag": C2_PREP_TAG, "commit": expected_commit}:
        raise C2Error("qualification C2-prep tag/commit differs")
    if record["c2_protocol_binding"] != {"path": _relative(C2_PROTOCOL_FILE), "sha256": C2_PROTOCOL_SHA256}:
        raise C2Error("qualification C2 protocol binding differs")
    if record["c2_runtime_manifest_binding"] != {"path": _relative(C2_RUNTIME_MANIFEST), "sha256": sha256_path(C2_RUNTIME_MANIFEST)}:
        raise C2Error("qualification C2 runtime-manifest binding differs")
    if record["c1_manifest_binding"] != {"sha256": C1_MANIFEST_SHA256, "tag": C1_TAG, "commit": C1_COMMIT}:
        raise C2Error("qualification C1 manifest binding differs")
    if record["phase_b_freeze"] != {"protocol_path": "configs/cascadekv_phi35_8k_phaseb_protocol.json", "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_path": "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json", "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256}:
        raise C2Error("qualification Phase-B binding differs")
    if record["target"] != {"model": TARGET_MODEL, "revision": TARGET_REVISION, "tokenizer_revision": TARGET_REVISION}:
        raise C2Error("qualification target differs")
    if record["model_compute_dtype"] != "float16" or record["attention_implementation"] != "sdpa" or record["use_cache"] is not False or record["quantization"] != "none":
        raise C2Error("qualification backend differs from canonical FP16/SDPA/no-cache/no-quantization")
    if record["gpu_count"] != 2 or len(record["gpu_names"]) != 2 or any("T4" not in str(name).upper() for name in record["gpu_names"]):
        raise C2Error("qualification does not prove T4 x2")
    if record["synthetic_sequence_lengths"] != list(LENGTHS) or record["capture_adapter"] != Phi3CaptureAdapter.identity:
        raise C2Error("qualification synthetic/capture adapter contract differs")
    if record["peak_memory_reset_device_indices"] != [0, 1]:
        raise C2Error("qualification did not explicitly reset both GPU peak counters")
    peaks = record["peak_cuda_memory_by_length"]
    if not isinstance(peaks, Mapping) or set(peaks) != {str(length) for length in LENGTHS}:
        raise C2Error("qualification peak memory is missing a synthetic sequence length")
    for length in LENGTHS:
        measurements = peaks[str(length)]
        if not isinstance(measurements, list) or len(measurements) != 2:
            raise C2Error("qualification peak memory does not measure both GPUs")
        by_index = {item.get("index"): item for item in measurements if isinstance(item, Mapping)}
        if set(by_index) != {0, 1}:
            raise C2Error("qualification peak memory GPU indices differ")
        for device_index, measurement in by_index.items():
            expected_total = record["vram_per_gpu_bytes"][device_index]
            required_measurement = {"index", "total_vram_bytes", "allocated_bytes", "reserved_bytes"}
            allowed_measurement = required_measurement | {"headroom_reserved_bytes"}
            if not required_measurement.issubset(measurement) or set(measurement) - allowed_measurement or measurement["total_vram_bytes"] != expected_total:
                raise C2Error("qualification peak memory VRAM binding differs")
            numeric = tuple(key for key in ("total_vram_bytes", "allocated_bytes", "reserved_bytes", "headroom_reserved_bytes") if key in measurement)
            if any(isinstance(measurement[key], bool) or not isinstance(measurement[key], int) or measurement[key] < 0 for key in numeric):
                raise C2Error("qualification peak memory values are invalid")
            if measurement["reserved_bytes"] < measurement["allocated_bytes"]:
                raise C2Error("qualification reserved peak is below allocated peak")
            if measurement.get("headroom_reserved_bytes") != (expected_total - measurement["reserved_bytes"] if measurement["reserved_bytes"] <= expected_total else None):
                raise C2Error("qualification reserved-memory headroom differs")
    if qualification_backend_id(record) != record["backend_id"]:
        raise C2Error("qualification backend identity SHA differs")
    if not isinstance(record["model_config_sha256"], str) or len(record["model_config_sha256"]) != 64:
        raise C2Error("qualification model config SHA differs")
    if set(record["capture_shape_proof_8192"]) != {str(item) for item in LAYERS}:
        raise C2Error("qualification omitted a five-layer shape proof")
    for shapes in record["capture_shape_proof_8192"].values():
        if shapes != {"q_shape": list(SHAPE), "k_shape": list(SHAPE), "v_shape": list(SHAPE)}:
            raise C2Error("qualification 8192 Q/K/V shape proof differs")
    if set(record["resolved_hf_device_map"].values()) != {0, 1}:
        raise C2Error("qualification map is not two-device GPU-only")


def load_approved_qualification(qualification: Path, approval: Path) -> tuple[dict[str, Any], str]:
    try:
        record = json.loads(qualification.read_text())
        reviewed = json.loads(approval.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise C2Error("cannot load qualification or separate frozen approval") from exc
    validate_qualification(record)
    digest = sha256_path(qualification)
    if reviewed != {"schema_version": APPROVAL_SCHEMA, "review_status": "FROZEN_APPROVED", "qualification_sha256": digest, "backend_id": record["backend_id"]}:
        raise C2Error("capture requires an exact separate reviewed/frozen qualification approval")
    return dict(record), digest


def assert_capture_backend(model: Any, qualification: Mapping[str, Any]) -> None:
    """The reviewed qualification is a backend freeze, not a mere permission slip."""
    devices = require_t4x2()
    if [item["name"] for item in devices] != qualification["gpu_names"]:
        raise C2Error("capture GPU identities differ from the reviewed qualification")
    if [item["total_vram_bytes"] for item in devices] != qualification["vram_per_gpu_bytes"]:
        raise C2Error("capture GPU VRAM differs from the reviewed qualification")
    if _config_sha(model) != qualification["model_config_sha256"]:
        raise C2Error("capture model config differs from the reviewed qualification")
    if _resolved_map(model) != qualification["resolved_hf_device_map"]:
        raise C2Error("capture device map differs from the reviewed qualification")
    if _versions() != qualification["software_versions"] or str(torch.version.cuda) != qualification["cuda_version"]:
        raise C2Error("capture software/CUDA backend differs from the reviewed qualification")


def _safe_source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    return {key: source[key] for key in ("family", "dataset", "config", "split", "revision", "field", "identity_kind", "dataset_index", "role")}


def _input_hash(input_ids: torch.Tensor) -> str:
    if input_ids.dtype != torch.int64 or input_ids.device.type != "cpu" or not input_ids.is_contiguous():
        raise C2Error("input ids must be canonical contiguous CPU int64")
    return hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()


def _selected_input(source: Mapping[str, Any], tokenizer: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    spec = {"dataset": source["dataset"], "config": source["config"], "split": source["split"], "revision": source["revision"], "field": source["field"]}
    dataset = load_dataset_for_spec(spec)
    proof = mechanical_proof(dataset, tokenizer, spec, int(source["dataset_index"]))
    if proof != source["proof"] or source["reproof"] != source["proof"]:
        raise C2Error("C1 selected source mechanical reproof differs")
    value = dataset[int(source["dataset_index"])].get(source["field"])
    if not isinstance(value, str):
        raise C2Error("C1 selected source field no longer supplies text")
    encoded = tokenizer(value, add_special_tokens=True, truncation=True, max_length=MINIMUM)
    del value, dataset
    ids = torch.tensor([encoded.input_ids], dtype=torch.int64).contiguous()
    if tuple(ids.shape) != (1, MINIMUM):
        raise C2Error("C1 selected source does not tokenize to exactly 8192 ids")
    return ids, proof


def _artifact_paths(output: Path, source: Mapping[str, Any], layer: int) -> tuple[Path, Path]:
    stem = f"{source['family']}_{source['role']}_index{source['dataset_index']}_layer{layer}"
    return output / "artifacts" / f"{stem}.safetensors", output / "artifacts" / f"{stem}.provenance.json"


def _artifact_provenance(source: Mapping[str, Any], proof: Mapping[str, Any], layer: int, input_sha: str, qualification: Mapping[str, Any], qualification_sha: str, model: Any) -> dict[str, Any]:
    return {
        "schema_version": CAPTURE_SCHEMA, "c2_protocol": C2_PROTOCOL,
        "c2_protocol_sha256": C2_PROTOCOL_SHA256,
        "c2_runtime_manifest_sha256": sha256_path(C2_RUNTIME_MANIFEST),
        "frozen_qualification_sha256": qualification_sha, "backend_id": qualification["backend_id"],
        "c1_freeze": {"tag": C1_TAG, "commit": C1_COMMIT, "manifest_sha256": C1_MANIFEST_SHA256},
        "phase_b_freeze": {"tag": PHASE_B_TAG, "commit": PHASE_B_COMMIT, "protocol_sha256": PROTOCOL_SHA256, "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256},
        "target": {"model": TARGET_MODEL, "revision": TARGET_REVISION, "model_config_sha256": _config_sha(model), "tokenizer_revision": TARGET_REVISION},
        "source_identity": _safe_source_identity(source), "source_proof": dict(proof), "source_reproof": dict(proof),
        "input_ids_sha256": input_sha, "layer": layer,
        "q_shape": list(SHAPE), "k_shape": list(SHAPE), "v_shape": list(SHAPE),
        "storage_dtype": "float16", "model_compute_dtype": "float16", "attention_implementation": "sdpa",
        "resolved_hf_device_map": _resolved_map(model), "capture_adapter": Phi3CaptureAdapter.identity,
        "software_versions": _versions(), "tensor_payload_bytes": ARTIFACT_PAYLOAD_BYTES,
    }


def _validate_no_text(value: Any) -> None:
    forbidden = {"text", "raw_text", "content", "document", "prompt"}
    if isinstance(value, Mapping):
        if forbidden.intersection(value):
            raise C2Error("capture metadata must not contain raw text")
        for item in value.values(): _validate_no_text(item)
    elif isinstance(value, list):
        for item in value: _validate_no_text(item)


def validate_artifact(path: Path, provenance_path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or not provenance_path.is_file():
        raise C2Error("artifact or per-artifact provenance is absent")
    try:
        actual = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise C2Error("artifact provenance is unreadable") from exc
    _validate_no_text(actual)
    if actual != expected:
        raise C2Error("artifact provenance mismatch; refusing overwrite or reuse")
    actual_sha = sha256_path(path)
    if actual.get("artifact_sha256") != actual_sha:
        raise C2Error("artifact SHA256 differs")
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"q", "k", "v"}:
                raise C2Error("artifact tensor names differ")
            for key in ("q", "k", "v"):
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != SHAPE or tensor.dtype != torch.float16:
                    raise C2Error("artifact tensor shape/dtype differs")
    except C2Error:
        raise
    except Exception as exc:
        raise C2Error("artifact tensor payload is corrupt") from exc
    return actual


def write_artifact(path: Path, provenance_path: Path, triplet: tuple[torch.Tensor, torch.Tensor, torch.Tensor], base_provenance: Mapping[str, Any]) -> dict[str, Any]:
    _check_triplet(triplet, expected=SHAPE)
    if path.exists() or provenance_path.exists():
        raise C2Error("refusing to overwrite an existing artifact")
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    save_file({"q": triplet[0], "k": triplet[1], "v": triplet[2]}, str(temporary))
    os.replace(temporary, path)
    provenance = dict(base_provenance)
    provenance["artifact_sha256"] = sha256_path(path)
    _validate_no_text(provenance)
    atomic_json(provenance_path, provenance)
    return provenance


def capture(output: Path, qualification_path: Path, approval_path: Path) -> dict[str, Any]:
    """Real-source mode; callable only after an independently frozen approval."""
    output = require_durable_output(output)
    preflight()
    qualification, qualification_sha = load_approved_qualification(qualification_path, approval_path)
    manifest = c1_manifest()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_frozen_tokenizer()
    model = _load_pinned_model()
    progress_path = output / "progress_manifest.json"
    progress: dict[str, Any] = {"schema_version": CAPTURE_SCHEMA, "backend_id": qualification["backend_id"], "complete_artifacts": []}
    try:
        assert_capture_backend(model, qualification)
        for source in manifest["selected_sources"]:
            ids, proof = _selected_input(source, tokenizer)
            input_sha = _input_hash(ids)
            expected = {layer: _artifact_provenance(source, proof, layer, input_sha, qualification, qualification_sha, model) for layer in LAYERS}
            existing: set[int] = set()
            absent: set[int] = set()
            for layer in LAYERS:
                artifact, provenance = _artifact_paths(output, source, layer)
                if artifact.exists() or provenance.exists():
                    existing.add(layer)
                    validate_artifact(artifact, provenance, {**expected[layer], "artifact_sha256": sha256_path(artifact)})
                else:
                    absent.add(layer)
            if absent:
                captured = capture_five_layers_post_rope_qkv(model, ids.to(_first_device(model)))
                for layer in absent:
                    artifact, provenance = _artifact_paths(output, source, layer)
                    write_artifact(artifact, provenance, captured[layer], expected[layer])
                del captured
            for layer in LAYERS:
                artifact, provenance = _artifact_paths(output, source, layer)
                verified = validate_artifact(artifact, provenance, {**expected[layer], "artifact_sha256": sha256_path(artifact)})
                progress["complete_artifacts"].append({"path": str(artifact.relative_to(output)), "sha256": verified["artifact_sha256"]})
            atomic_json(progress_path, progress)
            del ids
            gc.collect()
            torch.cuda.empty_cache()  # after a source forward and CPU serialization only
        if len(progress["complete_artifacts"]) != 45:
            raise C2Error("complete capture manifest must contain exactly 45 artifacts")
        final = {**progress, "qualification_sha256": qualification_sha, "artifact_payload_bytes": ARTIFACT_PAYLOAD_BYTES, "total_artifact_payload_bytes": TOTAL_ARTIFACT_PAYLOAD_BYTES, "capture_result": "COMPLETE"}
        atomic_json(output / "final_capture_manifest.json", final)
        return final
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight", help="local provenance only; no model or dataset")
    qualify_parser = sub.add_parser("qualify", help="pinned model plus synthetic valid IDs only")
    qualify_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "qualification.json")
    capture_parser = sub.add_parser("capture", help="gated real C1 capture; never default")
    capture_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    capture_parser.add_argument("--qualification", type=Path, required=True)
    capture_parser.add_argument("--approved-qualification", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight": result = preflight()
        elif args.command == "qualify": result = qualify(args.output)
        else: result = capture(args.output, args.qualification, args.approved_qualification)
        print(canonical_json(result))
    except (C2Error, OSError, ValueError, RuntimeError) as exc:
        print(f"PHI35-8K-KAGGLE-C2-FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
