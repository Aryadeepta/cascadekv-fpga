"""Fail-closed Phi-3.5 8K C3 calibration and one-shot validation.

Nothing in this module touches a tensor at import or preflight time.  Tensor
payloads are opened only through :class:`TensorAccess`, which makes the C3
calibration/validation boundary both executable and auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cascadekv_phi35_8k_c3_protocol.json"
RUNTIME = ROOT / "configs/cascadekv_phi35_8k_c3_runtime_manifest.json"
C3_TAG = "cascadekv-phi35-8k-c3-prep-v1"
SOURCE_SHA = "31f1d14d4b33b6e5b6a48d34e294f0effbab2eeadf6c4d4eabfabe16ae879478"
CAPTURE_SHA = "9b7736e37f7fa333fdc920978b99b7833a7cf91e5b4e6207139becf4f475b8eb"
SUPPLEMENT_PROGRESS_SHA = "196d111da6c5ccbf810b449c6d486115f6d53b49a7ee211b8628ebecd067a1aa"
PHASE_B = {"tag": "cascadekv-phi35-8k-phaseb-protocol-freeze", "commit": "d94cff8e1e11048ff3d8edc9fb1c605dfe422d6e", "protocol_sha256": "0e3a2e7b51e01c64dca292c7dd667ed9899c5bacc08e470e2b19e12433363c8b", "runtime_sha256": "e4e10f7659180471eb791eeafa97e8f525b6c14a3ff311798349c07438ec7c92"}
SUPPLEMENT = {"tag": "cascadekv-phi35-8k-capture-amendment-prep-v1", "commit": "3f9ca89724871946fe310246151c8475cfee30a5", "protocol_sha256": "007ffb3e197f759c1e84fd58f027a6b140d439db74097455df63c1dc1e41e1b5", "runtime_sha256": "26cad790476bf299df37017b39fbc20998aba1fb98d380e812d46a612cb0da86"}
LAYERS, POSITIONS, HEADS, SHAPE = (0, 8, 16, 24, 31), (4095, 6143, 8191), tuple(range(32)), (1, 32, 8192, 96)
ACTIONS, TARGETS = ("A0", "A1", "A2", "A3", "A4"), ("T0", "T1", "T2", "T3", "T4", "T5")
CAL_ROLES, VAL_ROLES = ("calibration_1", "calibration_2"), ("validation",)
EXPECTED = {"narrative": [(13, "calibration_1"), (14, "calibration_2"), (15, "validation")], "report": [(16, "calibration_1"), (17, "calibration_2"), (18, "validation")], "qa": [(13, "calibration_1"), (15, "calibration_2"), (16, "validation")]}
EVALUATED_METHODS = ("dense", "flat5", "flat10", "uniform5", "uniform10", *ACTIONS)
CLASSIFICATIONS = ("PHI35-8K-DEVELOPMENT-INVALID", "PHI35-8K-STATIC-NOT-PROMISING", "PHI35-8K-OUTPUT-GATE-NOT-PROMISING", "PHI35-8K-BANDWIDTH-NOT-PROMISING", "PHI35-8K-STATIC-PROMISING")


class C3Error(RuntimeError):
    """A C3 provenance, firewall, payload, or scientific contract failed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise C3Error(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(canonical_json(value) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise C3Error(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise C3Error(f"JSON root must be object: {path}")
    return value


def _git_commit(ref: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{ref}^{{commit}}"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise C3Error(f"required git reference unavailable: {ref}") from exc


def _tracked(relative: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative], capture_output=True, check=False).returncode == 0


def validate_protocol(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = dict(_json(PROTOCOL) if payload is None else payload)
    expected = {"schema_version", "phase_b_freeze", "supplement_freeze", "external_inputs", "geometry", "methods", "targets", "optimizer", "quality_gates", "firewall", "result_schemas", "c3_freeze_tag", "scientific_state", "classifications"}
    if set(p) != expected or p["schema_version"] != "cascadekv-phi35-8k-c3-protocol-v1":
        raise C3Error("C3 protocol schema differs")
    if p["phase_b_freeze"] != PHASE_B or p["supplement_freeze"] != SUPPLEMENT:
        raise C3Error("prior freeze binding differs")
    if p["external_inputs"] != {"source_manifest_sha256": SOURCE_SHA, "amended_capture_manifest_sha256": CAPTURE_SHA, "supplement_progress_sha256": SUPPLEMENT_PROGRESS_SHA}:
        raise C3Error("canonical external input binding differs")
    if p["geometry"] != {"context_length": 8192, "layers": list(LAYERS), "query_positions": list(POSITIONS), "heads": 32, "head_dim": 96, "observations_per_source_per_method": 480, "calibration_observations_per_method": 2880, "validation_observations_per_method": 1440, "schedule_cells": 160}:
        raise C3Error("frozen geometry differs")
    if p["targets"] != {"order": list(TARGETS), "fractions": {"T0": .115, "T1": .130, "T2": .145, "T3": .160, "T4": .180, "T5": "highest_traffic_strictly_below_calibration_flat5"}}:
        raise C3Error("frozen targets differ")
    if p["quality_gates"] != {"mean_cosine_gte": .985, "mean_relative_l2_lte": .120, "winner_relative_l2_lt": "uniform10", "winner_modeled_kv_traffic_lt": "flat5"}:
        raise C3Error("quality gates differ")
    # Phase B is authoritative: C3 may only rename reporting keys, never alter
    # the frozen architecture, action menu, or Phase-B DP semantics.
    from cascadekv import phi35_phaseb
    phase_b = phi35_phaseb.validate_protocol()
    if p["methods"]["actions"] != phase_b["actions"]:
        raise C3Error("C3 action menu differs from frozen Phase B")
    if p["methods"].get("hierarchy") != {"atom_size": 8, "forest": "causal_binary_counter_lifting", "grouped_variance": "G4", "routing": "K4_group16", "rerank": "exact_fp16_K", "selected_values": "FP16"}:
        raise C3Error("C3 hierarchy differs from frozen v1 architecture")
    for profile in ("phi35-8k-depth-transfer-qwen5-v1", "phi35-8k-depth-transfer-qwen10-v1"):
        c3_profile, phase_profile = p["methods"]["routing_profiles"][profile], phase_b["routing_profiles"][profile]
        if c3_profile["lambdas"] != phase_profile["lambdas"] or c3_profile["reserve"] != phase_profile["reserve"]:
            raise C3Error("C3 routing profile differs from frozen Phase B")
    if p["methods"]["baselines"] != ["dense", "flat5", "flat10", "uniform5", "uniform10", "target"]:
        raise C3Error("C3 baseline aliases differ")
    if p["optimizer"] != phase_b["optimizer"]:
        raise C3Error("C3 optimizer differs from frozen Phase-B semantics")
    if p["firewall"] != {"calibration_tensor_roles": list(CAL_ROLES), "validation_tensor_roles": list(VAL_ROLES), "validation_forbidden": ["optimizer_rerun", "schedule_mutation", "target_change", "target_reorder"]}:
        raise C3Error("C3 firewall differs")
    if p["result_schemas"] != {"calibration": "cascadekv-phi35-8k-c3-calibration-v1", "validation": "cascadekv-phi35-8k-c3-validation-v1"}:
        raise C3Error("C3 result schemas differ")
    if p["c3_freeze_tag"] != C3_TAG:
        raise C3Error("C3 freeze tag differs")
    if p["classifications"] != list(phase_b["classifications"]) or tuple(p["classifications"]) != CLASSIFICATIONS:
        raise C3Error("C3 classification vocabulary differs from frozen Phase B")
    if p["scientific_state"] != {"local_preparation_only": True, "quality_metrics_observed": False, "schedule_optimized": False}:
        raise C3Error("C3 preparation state differs")
    return p


def verify_runtime_closure() -> tuple[dict[str, str], ...]:
    manifest = _json(RUNTIME)
    if set(manifest) != {"schema_version", "purpose", "protocol", "bound_files"} or manifest["schema_version"] != 1:
        raise C3Error("C3 runtime manifest schema differs")
    if manifest["protocol"] != {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256_path(PROTOCOL)}:
        raise C3Error("C3 runtime protocol binding differs")
    seen, result = set(), []
    for entry in manifest["bound_files"]:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise C3Error("C3 runtime entry malformed")
        relative, digest = entry["path"], entry["sha256"]
        if not isinstance(relative, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise C3Error("C3 runtime entry unsafe")
        seen.add(relative); path = ROOT / relative
        if not path.is_file() or not _tracked(relative) or sha256_path(path) != digest:
            raise C3Error(f"C3 runtime file fails closed: {relative}")
        result.append({"path": relative, "sha256": digest})
    return tuple(result)


def preflight(*, require_c3_tag: bool = True) -> dict[str, Any]:
    """Repository-only; no tensor, model, tokenizer, or dataset imports."""
    if _git_commit(PHASE_B["tag"]) != PHASE_B["commit"] or _git_commit(SUPPLEMENT["tag"]) != SUPPLEMENT["commit"]:
        raise C3Error("immutable prior tag does not resolve to frozen commit")
    if require_c3_tag and _git_commit(C3_TAG) != _git_commit("HEAD"):
        raise C3Error("C3 freeze tag must resolve exactly to HEAD")
    from cascadekv import phi35_phaseb
    phi35_phaseb.validate_protocol()
    validate_protocol(); bound = verify_runtime_closure()
    return {"protocol_path": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha256_path(PROTOCOL), "runtime_manifest_path": str(RUNTIME.relative_to(ROOT)), "runtime_manifest_sha256": sha256_path(RUNTIME), "bound_files": bound, "c3_tag_required": require_c3_tag}


def _safe(path: str) -> Path:
    if not isinstance(path, str) or not path:
        raise C3Error("artifact path must be a non-empty relative string")
    value = Path(path)
    if value.is_absolute() or ".." in value.parts:
        raise C3Error("unsafe relative artifact path")
    return value


def _safe_under(root: Path, relative: str) -> Path:
    """Resolve a manifest path without permitting traversal through symlinks."""
    candidate = root / _safe(relative)
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise C3Error("artifact path escapes its declared root") from exc
    return candidate


def _identity(item: Mapping[str, Any]) -> str:
    return f"{item['family']}:{item['dataset_index']}:{item['role']}"


def _expected_tensor_identities(roles: Iterable[str]) -> list[str]:
    allowed = set(roles)
    return sorted(
        f"{family}:{index}:{role}:L{layer}"
        for family, rows in EXPECTED.items()
        for index, role in rows if role in allowed
        for layer in LAYERS
    )


def _expected_source_identities(roles: Iterable[str]) -> list[str]:
    allowed = set(roles)
    return sorted(f"{family}:{index}:{role}" for family, rows in EXPECTED.items() for index, role in rows if role in allowed)


SCHEDULE_FIELDS = ("feasible", "target_mean_kv_bytes", "strict", "integer_microbyte_cap", "integer_microbytes_used", "table")


def _schedule_cell_keys() -> set[str]:
    return {f"{layer}:{head}" for layer in LAYERS for head in HEADS}


def _target_mean_and_cap(protocol: Mapping[str, Any], target: str, dense_mean: float, flat5_mean: float) -> tuple[float, bool, int]:
    """Apply the frozen Phase-B integer-microbyte budget rule exactly."""
    target_mean, strict = (flat5_mean, True) if target == "T5" else (protocol["targets"]["fractions"][target] * dense_mean, False)
    return target_mean, strict, math.ceil(target_mean * len(LAYERS) * len(HEADS) * 1000) - (1 if strict else 0)


def _unimplemented_phaseb_diagnostics() -> dict[str, dict[str, str]]:
    """Make the lack of a frozen percentile convention explicit, never implicit."""
    reason = "no frozen repository percentile calculation semantics recovered"
    return {name: {"status": "unimplemented", "reason": reason} for name in ("cosine_p5", "relative_l2_p95", "relative_l2_worst")}


def schedule_digest(schedules: Mapping[str, Any]) -> str:
    """Canonical, data-free digest of all frozen target schedule metadata."""
    if not isinstance(schedules, Mapping) or list(schedules) != list(TARGETS):
        raise C3Error("schedule digest requires T0..T5 in frozen order")
    canonical = []
    for target in TARGETS:
        item = schedules[target]
        if not isinstance(item, Mapping) or set(item) != set(SCHEDULE_FIELDS):
            raise C3Error("schedule digest requires exact target mappings")
        canonical.append({key: item[key] for key in SCHEDULE_FIELDS})
    return hashlib.sha256(canonical_json({"target_order": list(TARGETS), "schedules": canonical}).encode("ascii")).hexdigest()


class CaptureResolver:
    """Manifest-only logical resolver.  It never opens tensor payloads."""
    def __init__(self, source_manifest: Path, amended_capture_manifest: Path, base_capture_root: Path, supplement_root: Path):
        if sha256_path(source_manifest) != SOURCE_SHA or sha256_path(amended_capture_manifest) != CAPTURE_SHA:
            raise C3Error("source/capture manifest SHA256 differs")
        self.sources_payload, self.capture = _json(source_manifest), _json(amended_capture_manifest)
        self.base, self.supplement = base_capture_root.resolve(), supplement_root.resolve()
        progress = self.supplement / "supplement_progress_manifest.json"
        if not progress.is_file() or sha256_path(progress) != SUPPLEMENT_PROGRESS_SHA:
            raise C3Error("supplement progress SHA256 differs")
        self.sources = self._validate_sources()
        self.records = self._validate_capture()

    def _validate_sources(self) -> dict[str, dict[str, Any]]:
        selected = self.sources_payload.get("selected_sources")
        if not isinstance(selected, list) or len(selected) != 9:
            raise C3Error("amended source manifest must contain exactly nine selections")
        out: dict[str, dict[str, Any]] = {}
        for family, pairs in EXPECTED.items():
            rows = [x for x in selected if isinstance(x, Mapping) and x.get("family") == family]
            if [(x.get("dataset_index"), x.get("role")) for x in rows] != pairs:
                raise C3Error("corrected source roles differ")
            for row in rows:
                out[_identity(row)] = dict(row)
        rejected = self.sources_payload.get("rejected_sources", {}).get("qa", [])
        if not any(x.get("dataset_index") == 14 and x.get("rejection_reason") == "duplicate_input" for x in rejected if isinstance(x, Mapping)):
            raise C3Error("QA14 duplicate rejection missing")
        return out

    def _validate_capture(self) -> dict[str, dict[str, Any]]:
        if self.capture.get("schema_version") != "cascadekv-phi35-8k-amended-capture-v1" or self.capture.get("amended_source_manifest_sha256") != SOURCE_SHA or self.capture.get("logical_artifact_count") != 45:
            raise C3Error("amended capture manifest differs")
        entries = self.capture.get("artifacts")
        if not isinstance(entries, list) or len(entries) != 45:
            raise C3Error("amended capture logical inventory differs")
        origins = {name: sum(x.get("origin") == name for x in entries if isinstance(x, Mapping)) for name in ("historical_unchanged", "historical_role_rebound", "supplement_capture")}
        if origins != {"historical_unchanged": 35, "historical_role_rebound": 5, "supplement_capture": 5}:
            raise C3Error("logical origin accounting differs")
        out: dict[str, dict[str, Any]] = {}
        for identity, source in self.sources.items():
            family, index, role = source["family"], source["dataset_index"], source["role"]
            for layer in LAYERS:
                key = f"{identity}:L{layer}"
                # QA15 is a byte-identical historical validation artifact with
                # a separately verified calibration_2 role-reference wrapper.
                stored_role = "validation" if (family, index, role) == ("qa", 15, "calibration_2") else role
                suffix = f"{family}_{stored_role}_index{index}_layer{layer}.safetensors"
                found = [x for x in entries if isinstance(x, Mapping) and (x.get("base_artifact_relative_path", "").endswith(suffix) or x.get("supplement_artifact_relative_path", "").endswith(suffix))]
                if len(found) != 1:
                    raise C3Error(f"logical artifact absent/ambiguous: {key}")
                record = dict(found[0]); origin = record["origin"]
                if origin == "historical_unchanged":
                    path, digest, root = record["base_artifact_relative_path"], record["base_artifact_sha256"], self.base
                elif origin == "historical_role_rebound":
                    if set(record) != {"origin", "base_artifact_relative_path", "base_artifact_sha256", "historical_provenance_relative_path", "historical_provenance_sha256", "role_reference_relative_path", "role_reference_sha256"}:
                        raise C3Error("historical role rebound manifest entry differs")
                    path, digest, root = record["base_artifact_relative_path"], record["base_artifact_sha256"], self.base
                    historical = _safe_under(self.base, record["historical_provenance_relative_path"])
                    ref = _safe_under(self.supplement, record["role_reference_relative_path"])
                    if not historical.is_file() or sha256_path(historical) != record["historical_provenance_sha256"]:
                        raise C3Error("historical role rebound provenance SHA differs")
                    if not ref.is_file() or sha256_path(ref) != record["role_reference_sha256"]:
                        raise C3Error("historical role rebound reference SHA differs")
                    reference = _json(ref)
                    if (reference.get("schema_version"), reference.get("family"), reference.get("dataset_index"), reference.get("layer"), reference.get("historical_role"), reference.get("amended_role"), reference.get("tensor_bytes_reused_unchanged"), reference.get("tensor_bytes_recaptured"), reference.get("historical_artifact_sha256"), reference.get("historical_provenance_relative_path"), reference.get("historical_provenance_sha256")) != ("cascadekv-phi35-8k-amended-artifact-reference-v1", "qa", 15, layer, "validation", "calibration_2", True, False, digest, record["historical_provenance_relative_path"], record["historical_provenance_sha256"]):
                        raise C3Error("historical role rebound semantics differ")
                elif origin == "supplement_capture":
                    if set(record) != {"origin", "supplement_artifact_relative_path", "supplement_artifact_sha256", "supplement_provenance_relative_path", "supplement_provenance_sha256"}:
                        raise C3Error("supplement capture manifest entry differs")
                    path, digest, root = record["supplement_artifact_relative_path"], record["supplement_artifact_sha256"], self.supplement
                    provenance = _safe_under(self.supplement, record["supplement_provenance_relative_path"])
                    if not provenance.is_file() or sha256_path(provenance) != record["supplement_provenance_sha256"]:
                        raise C3Error("supplement provenance SHA differs")
                    detail = _json(provenance); source_identity = detail.get("source_identity", {})
                    if (detail.get("schema_version"), source_identity.get("family"), source_identity.get("dataset_index"), source_identity.get("role"), detail.get("layer"), detail.get("artifact_sha256"), detail.get("input_ids_sha256")) != ("cascadekv-phi35-8k-capture-amendment-artifact-v1", "qa", 16, "validation", layer, digest, "d9bcac44581785b0d180fdcd46cbcc5dcecd50e0e2c4285d81d949d1b35bc3ac"):
                        raise C3Error("supplement provenance semantics differ")
                else:
                    raise C3Error("unknown logical artifact origin")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise C3Error("logical artifact SHA malformed")
                out[key] = {"identity": identity, "layer": layer, "origin": origin, "path": _safe_under(root, path), "sha256": digest}
        if any("qa:14:" in key for key in out):
            raise C3Error("QA14 must have zero logical artifacts")
        if set(out) != set(_expected_tensor_identities((*CAL_ROLES, *VAL_ROLES))):
            raise C3Error("logical artifact identities differ from corrected source inventory")
        return out

    def records_for_roles(self, roles: Iterable[str]) -> list[dict[str, Any]]:
        allowed = set(roles)
        return [self.records[key] for key in sorted(self.records) if self.records[key]["identity"].rsplit(":", 1)[1] in allowed]


class TensorAccess:
    """Payload gate recording every logical tensor identity actually opened."""
    def __init__(self, resolver: CaptureResolver, allowed_roles: Iterable[str]):
        self.resolver, self.allowed = resolver, set(allowed_roles)
        self.expected = set(_expected_tensor_identities(self.allowed))
        self.opened: list[str] = []

    def open(self, record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        identity = str(record["identity"])
        if identity.rsplit(":", 1)[1] not in self.allowed:
            raise C3Error(f"tensor firewall denied: {identity}")
        opened_identity = f"{identity}:L{record['layer']}"
        if opened_identity not in self.expected:
            raise C3Error(f"tensor firewall denied unexpected identity: {opened_identity}")
        path = Path(record["path"])
        if not path.is_file() or sha256_path(path) != record["sha256"]:
            raise C3Error(f"tensor SHA verification failed: {identity}")
        # Deferred: preflight does not import safetensors or torch.
        from safetensors import safe_open
        import torch
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"q", "k", "v"}:
                raise C3Error("tensor keys must be exactly q, k, v")
            q, k, v = (handle.get_tensor(name) for name in ("q", "k", "v"))
        if any(x.dtype != torch.float16 or tuple(x.shape) != SHAPE for x in (q, k, v)):
            raise C3Error("tensor dtype/shape differs from frozen geometry")
        self.opened.append(opened_identity)
        return q, k, v


def _profile(protocol: Mapping[str, Any], name: str) -> tuple[dict[int, float], dict[str, Any]]:
    profile = protocol["methods"]["routing_profiles"][name]
    return {int(k): float(v) for k, v in profile["lambdas"].items()}, dict(profile["reserve"])


def _method_rows(access: TensorAccess, records: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any], schedules: Mapping[str, Mapping[str, str]] | None = None) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Evaluate frozen actions/baselines and optionally fixed target schedules."""
    from cascadekv.evaluation_core import (advance_until_budget, build_forests, candidate_budget, exact_attention_reference, exact_rerank_and_output, initialize_route, reserve_ids, select_flat_k4, traffic)
    from cascadekv.model_agnostic import PHI35_GEOMETRY
    methods = {name: [] for name in EVALUATED_METHODS}
    if schedules: methods.update({name: [] for name in schedules})
    five_lambdas, five_reserve = _profile(protocol, "phi35-8k-depth-transfer-qwen5-v1")
    ten_lambdas, ten_reserve = _profile(protocol, "phi35-8k-depth-transfer-qwen10-v1")
    for record in records:
        q_all, k_all, v_all = access.open(record); layer = int(record["layer"])
        forests = build_forests(k_all[0], PHI35_GEOMETRY, POSITIONS, atom_size=8, groups=(4,))
        for position in POSITIONS:
            n = position + 1
            for head in HEADS:
                query, keys, values = q_all[0, head, position], k_all[0, head, :n], v_all[0, head, :n]
                reference = exact_attention_reference(query, keys, values)
                forest = forests[position, head]
                def flat(frac: float):
                    budget = candidate_budget(n, frac); ids = select_flat_k4(query, keys, budget)
                    return ids, traffic({"active_root_reads": 0, "detail_reads": 0, "expanded_internal_nodes": 0, "variance_scalar_reads": 0}, n, budget, 96, variance=False, flat=True)
                def hier(frac: float, lambdas: Mapping[int, float], reserve: Mapping[str, Any]):
                    budget = candidate_budget(n, frac); route = advance_until_budget(initialize_route(forest, query, 4, lambdas[layer], reserve_ids(n, int(reserve["sink"]), int(reserve["local"]), budget)), budget)
                    return set(route["ids"]), traffic(route, n, budget, 96, variance=True)
                choices = {"dense": (set(range(n)), {"total_k_bytes": float(n * 192), "selected_v_fp16_bytes": float(n * 192)}), "flat5": flat(.05), "flat10": flat(.10), "uniform5": hier(.05, five_lambdas, five_reserve), "uniform10": hier(.10, ten_lambdas, ten_reserve)}
                for action in ACTIONS:
                    choices[action] = flat(.05) if action == "A4" else hier(float(protocol["methods"]["actions"][action]["candidate_fraction"]), five_lambdas, five_reserve)
                common = {"source": record["identity"], "layer": layer, "position": position, "head": head}
                for name, (ids, item_traffic) in choices.items():
                    values_out = exact_rerank_and_output(query, keys, values, ids, reference=reference)
                    item_traffic = dict(item_traffic); item_traffic["total_kv_bytes"] = item_traffic["total_k_bytes"] + item_traffic["selected_v_fp16_bytes"]
                    methods[name].append({**common, "metrics": values_out, "traffic": item_traffic})
                if schedules:
                    cell = f"{layer}:{head}"
                    for target in schedules:
                        action = schedules[target][cell]
                        methods[target].append(dict(methods[action][-1]))
    return methods, access.opened


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise C3Error("cannot summarize zero observations")
    return {"observation_count": len(rows), "mean_cosine": mean(float(x["metrics"]["cosine_similarity"]) for x in rows), "mean_relative_l2": mean(float(x["metrics"]["relative_l2_error"]) for x in rows), "mean_total_kv_bytes": mean(float(x["traffic"]["total_kv_bytes"]) for x in rows)}


def _calibration_schedules(rows: Mapping[str, list[Mapping[str, Any]]], protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from cascadekv.phi35_phaseb import optimize_static_schedule
    stats, groups = {}, []
    for layer in LAYERS:
        for head in HEADS:
            group = {}
            for action in ACTIONS:
                subset = [row for row in rows[action] if row["layer"] == layer and row["head"] == head]
                if len(subset) != 18:
                    raise C3Error("calibration action cell must contain 18 observations")
                group[action] = {"relative_l2": mean(x["metrics"]["relative_l2_error"] for x in subset), "cosine": mean(x["metrics"]["cosine_similarity"] for x in subset), "total_kv_bytes": mean(x["traffic"]["total_kv_bytes"] for x in subset)}
                stats[f"{layer}:{head}:{action}"] = group[action]
            groups.append(group)
    flat5 = _summary(rows["flat5"])["mean_total_kv_bytes"]
    dense = _summary(rows["dense"])["mean_total_kv_bytes"]
    schedules: dict[str, Any] = {}
    for target in TARGETS:
        # T0--T4 are the frozen dense K+V fractions, measured over the exact
        # calibration observation distribution; T5 is strictly below flat5.
        target_value, strict, cap = _target_mean_and_cap(protocol, target, dense, flat5)
        solution = optimize_static_schedule(groups, target_value, strict=strict)
        if solution is None:
            schedules[target] = {"feasible": False, "target_mean_kv_bytes": target_value, "strict": strict, "integer_microbyte_cap": cap, "integer_microbytes_used": None, "table": None}
            continue
        used, _error, _cosine, action_table = solution
        table = {f"{layer}:{head}": action for (layer, head), action in zip(((layer, head) for layer in LAYERS for head in HEADS), action_table, strict=True)}
        if type(used) is not int or used < 0 or used > cap or set(table) != _schedule_cell_keys() or set(table.values()) - set(ACTIONS):
            raise C3Error("frozen Phase-B optimizer returned an invalid schedule")
        schedules[target] = {"feasible": True, "target_mean_kv_bytes": target_value, "strict": strict, "integer_microbyte_cap": cap, "integer_microbytes_used": used, "table": table}
    return stats, schedules


def calibrate(*, source_manifest: Path, amended_capture_manifest: Path, base_capture_root: Path, supplement_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise C3Error(f"refusing to overwrite result: {output}")
    preflight(require_c3_tag=True)
    resolver = CaptureResolver(source_manifest, amended_capture_manifest, base_capture_root, supplement_root)
    access = TensorAccess(resolver, CAL_ROLES)
    protocol = validate_protocol(); rows, opened = _method_rows(access, resolver.records_for_roles(CAL_ROLES), protocol)
    if opened != _expected_tensor_identities(CAL_ROLES):
        raise C3Error("calibration tensor access firewall proof failed")
    summaries = {name: _summary(value) for name, value in rows.items()}
    if any(value["observation_count"] != 2880 for value in summaries.values()):
        raise C3Error("calibration observation count differs")
    statistics, schedules = _calibration_schedules(rows, protocol)
    # An infeasible target is a valid observed Phase-B DP outcome, not a
    # reason to discard the calibration artifact or skip later targets.
    result = {"schema_version": "cascadekv-phi35-8k-c3-calibration-v1", "c3_protocol_sha256": sha256_path(PROTOCOL), "c3_runtime_manifest_sha256": sha256_path(RUNTIME), "source_manifest_sha256": SOURCE_SHA, "amended_capture_manifest_sha256": CAPTURE_SHA, "prior_freezes": {"phase_b": PHASE_B, "supplement": SUPPLEMENT}, "calibration_sources": _expected_source_identities(CAL_ROLES), "validation_tensor_access": False, "opened_validation_tensor_identities": [], "opened_calibration_tensor_identities": opened, "methods": protocol["methods"], "calibration_metrics": summaries, "calibration_traffic": {name: value["mean_total_kv_bytes"] for name, value in summaries.items()}, "action_cell_statistics": statistics, "schedules": schedules, "deterministic_tie_provenance": protocol["optimizer"], "classification": None, "optional_phaseb_diagnostics": _unimplemented_phaseb_diagnostics()}
    atomic_json(output, result); return result


def _validate_calibration(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Validate every persisted calibration fact before validation may open data."""
    expected_keys = {"schema_version", "c3_protocol_sha256", "c3_runtime_manifest_sha256", "source_manifest_sha256", "amended_capture_manifest_sha256", "prior_freezes", "calibration_sources", "validation_tensor_access", "opened_validation_tensor_identities", "opened_calibration_tensor_identities", "methods", "calibration_metrics", "calibration_traffic", "action_cell_statistics", "schedules", "deterministic_tie_provenance", "classification", "optional_phaseb_diagnostics"}
    if set(payload) != expected_keys or payload.get("schema_version") != "cascadekv-phi35-8k-c3-calibration-v1":
        raise C3Error("calibration result schema differs")
    protocol = validate_protocol()
    if (payload.get("c3_protocol_sha256"), payload.get("c3_runtime_manifest_sha256"), payload.get("source_manifest_sha256"), payload.get("amended_capture_manifest_sha256"), payload.get("prior_freezes"), payload.get("methods"), payload.get("deterministic_tie_provenance")) != (sha256_path(PROTOCOL), sha256_path(RUNTIME), SOURCE_SHA, CAPTURE_SHA, {"phase_b": PHASE_B, "supplement": SUPPLEMENT}, protocol["methods"], protocol["optimizer"]):
        raise C3Error("calibration provenance differs")
    if payload.get("classification") is not None and payload["classification"] not in CLASSIFICATIONS:
        raise C3Error("calibration classification differs")
    if payload.get("optional_phaseb_diagnostics") != _unimplemented_phaseb_diagnostics():
        raise C3Error("calibration optional Phase-B diagnostics differ")
    if payload.get("validation_tensor_access") is not False or payload.get("opened_validation_tensor_identities") != [] or payload.get("calibration_sources") != _expected_source_identities(CAL_ROLES) or payload.get("opened_calibration_tensor_identities") != _expected_tensor_identities(CAL_ROLES):
        raise C3Error("calibration tensor firewall proof differs")
    metrics, traffic = payload.get("calibration_metrics"), payload.get("calibration_traffic")
    if not isinstance(metrics, Mapping) or not isinstance(traffic, Mapping) or set(metrics) != set(EVALUATED_METHODS) or set(traffic) != set(EVALUATED_METHODS):
        raise C3Error("calibration evaluated method set differs")
    for name in EVALUATED_METHODS:
        item = metrics[name]
        if not isinstance(item, Mapping) or set(item) != {"observation_count", "mean_cosine", "mean_relative_l2", "mean_total_kv_bytes"} or item.get("observation_count") != 2880 or not all(isinstance(item.get(key), (int, float)) and math.isfinite(item[key]) for key in ("mean_cosine", "mean_relative_l2", "mean_total_kv_bytes")) or traffic[name] != item["mean_total_kv_bytes"]:
            raise C3Error(f"calibration metric/traffic differs: {name}")
    action_stats = payload.get("action_cell_statistics")
    expected_stats = {f"{layer}:{head}:{action}" for layer in LAYERS for head in HEADS for action in ACTIONS}
    if not isinstance(action_stats, Mapping) or set(action_stats) != expected_stats:
        raise C3Error("calibration action statistics differ")
    for value in action_stats.values():
        if not isinstance(value, Mapping) or set(value) != {"relative_l2", "cosine", "total_kv_bytes"} or not all(isinstance(value.get(key), (int, float)) and math.isfinite(value[key]) for key in value):
            raise C3Error("calibration action statistic malformed")
    schedules = payload.get("schedules")
    if not isinstance(schedules, Mapping) or list(schedules) != list(TARGETS):
        raise C3Error("calibration target order differs")
    dense, flat5 = traffic["dense"], traffic["flat5"]
    expected_cells = _schedule_cell_keys()
    for target in TARGETS:
        item = schedules[target]
        if not isinstance(item, Mapping) or set(item) != set(SCHEDULE_FIELDS):
            raise C3Error(f"calibration target schema differs: {target}")
        expected_target, strict, cap = _target_mean_and_cap(protocol, target, dense, flat5)
        if item.get("target_mean_kv_bytes") != expected_target or item.get("strict") is not strict or item.get("integer_microbyte_cap") != cap or type(item.get("feasible")) is not bool:
            raise C3Error(f"calibration target metadata differs: {target}")
        if item["feasible"]:
            table, used = item.get("table"), item.get("integer_microbytes_used")
            if not isinstance(table, Mapping) or set(table) != expected_cells or set(table.values()) - set(ACTIONS) or type(used) is not int or used < 0 or used > cap:
                raise C3Error(f"calibration feasible target differs: {target}")
        elif item.get("table") is not None or item.get("integer_microbytes_used") is not None:
            raise C3Error(f"calibration infeasible target differs: {target}")
    schedule_digest(schedules)
    return dict(payload)


def validate(*, source_manifest: Path, amended_capture_manifest: Path, base_capture_root: Path, supplement_root: Path, calibration_result: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise C3Error(f"refusing to overwrite result: {output}")
    preflight(require_c3_tag=True)
    calibration_sha = sha256_path(calibration_result); calibration = _validate_calibration(_json(calibration_result), calibration_result)
    resolver = CaptureResolver(source_manifest, amended_capture_manifest, base_capture_root, supplement_root)
    protocol = validate_protocol()
    schedule_sha_before = schedule_digest(calibration["schedules"])
    schedules = {target: calibration["schedules"][target]["table"] for target in TARGETS if calibration["schedules"][target]["feasible"]}
    access = TensorAccess(resolver, VAL_ROLES)
    rows, opened = _method_rows(access, resolver.records_for_roles(VAL_ROLES), protocol, schedules)
    if opened != _expected_tensor_identities(VAL_ROLES):
        raise C3Error("validation tensor access firewall proof failed")
    summaries = {name: _summary(value) for name, value in rows.items()}
    if any(summaries[name]["observation_count"] != 1440 for name in summaries):
        raise C3Error("validation observation count differs")
    target_facts, winner = {}, None
    for target in TARGETS:
        if target not in schedules:
            target_facts[target] = {"feasible": False, "passes": False}; continue
        item = summaries[target]; facts = {"feasible": True, "mean_cosine_gte": item["mean_cosine"] >= .985, "mean_relative_l2_lte": item["mean_relative_l2"] <= .120, "relative_l2_lt_uniform10": item["mean_relative_l2"] < summaries["uniform10"]["mean_relative_l2"], "modeled_kv_traffic_lt_flat5": item["mean_total_kv_bytes"] < summaries["flat5"]["mean_total_kv_bytes"]}
        facts["passes"] = all(value for key, value in facts.items() if key not in {"feasible", "passes"}); target_facts[target] = facts
        if winner is None and facts["passes"]: winner = target
    schedule_sha_after = schedule_digest(calibration["schedules"])
    if schedule_sha_before != schedule_sha_after:
        raise C3Error("validation mutated the calibration schedules")
    # Phase B declares a vocabulary but no unambiguous classification mapping;
    # preserve the explicit gate facts and deliberately emit no invented label.
    result = {"schema_version": "cascadekv-phi35-8k-c3-validation-v1", "calibration_result_sha256": calibration_sha, "source_manifest_sha256": SOURCE_SHA, "amended_capture_manifest_sha256": CAPTURE_SHA, "c3_protocol_sha256": sha256_path(PROTOCOL), "c3_runtime_manifest_sha256": sha256_path(RUNTIME), "prior_freezes": {"phase_b": PHASE_B, "supplement": SUPPLEMENT}, "validation_sources": _expected_source_identities(VAL_ROLES), "opened_calibration_tensor_identities": [], "opened_validation_tensor_identities": opened, "target_order": list(TARGETS), "baseline_validation_metrics": {name: summaries[name] for name in ("dense", "flat5", "flat10", "uniform5", "uniform10")}, "target_validation_metrics": {name: summaries[name] for name in TARGETS if name in summaries}, "gate_values": protocol["quality_gates"], "per_gate_pass_fail": target_facts, "first_passing_target": winner, "no_schedule_mutation_evidence": {"schedule_sha256_before": schedule_sha_before, "schedule_sha256_after": schedule_sha_after, "schedule_unchanged": True, "optimizer_rerun": False}, "traffic_accounting": {name: summaries[name]["mean_total_kv_bytes"] for name in summaries}, "classification": None, "optional_phaseb_diagnostics": _unimplemented_phaseb_diagnostics()}
    atomic_json(output, result); return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("preflight", help="repository-only; imports no data/model/tensor stack")
    for name in ("calibrate", "validate"):
        item = modes.add_parser(name)
        item.add_argument("--source-manifest", type=Path, required=True); item.add_argument("--amended-capture-manifest", type=Path, required=True)
        item.add_argument("--base-capture-root", type=Path, required=True); item.add_argument("--supplement-root", type=Path, required=True); item.add_argument("--output", type=Path, required=True)
        if name == "validate": item.add_argument("--calibration-result", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "preflight": result = preflight(require_c3_tag=False)
        elif args.mode == "calibrate": result = calibrate(source_manifest=args.source_manifest, amended_capture_manifest=args.amended_capture_manifest, base_capture_root=args.base_capture_root, supplement_root=args.supplement_root, output=args.output)
        else: result = validate(source_manifest=args.source_manifest, amended_capture_manifest=args.amended_capture_manifest, base_capture_root=args.base_capture_root, supplement_root=args.supplement_root, calibration_result=args.calibration_result, output=args.output)
        print(canonical_json(result)); return 0
    except (C3Error, OSError, ValueError, RuntimeError) as exc:
        print(f"PHI35-8K-C3-FAILED: {exc}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
