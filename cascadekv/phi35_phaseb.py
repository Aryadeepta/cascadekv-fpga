"""Data-free Phase-B protocol contract for prospective Phi-3.5 8K work.

This module intentionally has no model, tokenizer, dataset, cache, or capture
imports.  Phase C must provide already-permitted mechanical inspection records;
this module never resolves a remote source or reads candidate text itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from cascadekv.runtime_provenance import RuntimeProvenanceError, verify_runtime_manifest

ROOT = Path(__file__).resolve().parents[1]
PHASE_A_TAG = "cascadekv-phi35-8k-phasea-freeze"
PHASE_A_COMMIT = "08084f65628b7204603f068e68bff074599a116b"
PHASE_A_MANIFEST = ROOT / "configs/cascadekv_phi35_phasea_runtime_manifest.json"
PROTOCOL = ROOT / "configs/cascadekv_phi35_8k_phaseb_protocol.json"
RUNTIME_MANIFEST = ROOT / "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json"
ACTIONS = ("A0", "A1", "A2", "A3", "A4")
TARGETS = ("T0", "T1", "T2", "T3", "T4", "T5")
REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
QWEN_PROTOCOL_TAG = "cascadekv-v3-qwen3-1p7b-wide-vaware-dev-protocol-freeze"
QWEN_PROTOCOL_TAG_OBJECT_SHA = "b2700ad5f5db835bf54b61e01166b5688ed8f394"
QWEN_PROTOCOL_COMMIT = "f0c5b5666ad089531fca68862c5be959088d49ef"


class PhaseBError(RuntimeError):
    """A prospective Phase-B contract is missing, mutable, or malformed."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseBError(f"cannot load required protocol file: {path}") from exc
    if not isinstance(value, dict):
        raise PhaseBError(f"protocol root must be an object: {path}")
    return value


def protocol() -> dict[str, Any]:
    return _load(PROTOCOL)


def _tracked(relative: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative], capture_output=True).returncode == 0


def verify_phase_a() -> tuple[dict[str, str], ...]:
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{PHASE_A_TAG}^{{commit}}"], text=True).strip()
    if commit != PHASE_A_COMMIT:
        raise PhaseBError("Phase-A freeze tag does not resolve to its frozen commit")
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", PHASE_A_TAG, "HEAD"], check=False).returncode:
        raise PhaseBError("current scientific closure is not descended from the Phase-A freeze")
    try:
        bound = verify_runtime_manifest(PHASE_A_MANIFEST, ROOT, require_tracked=True)
    except RuntimeProvenanceError as exc:
        raise PhaseBError(f"Phase-A runtime closure failed: {exc}") from exc
    phase_a = _load(PHASE_A_MANIFEST)
    target = phase_a.get("target_model", {})
    if target != {"name": "microsoft/Phi-3.5-mini-instruct", "revision": REVISION} or phase_a.get("target_family") != "phi3" or phase_a.get("phase_a_scientific_status") != "NO_PHI_DATA_CONSUMED":
        raise PhaseBError("Phase-A target/status binding differs")
    return bound


def _no_selected_identities(value: Any) -> bool:
    if isinstance(value, dict):
        return not any(key in {"selected_sources", "dataset_index", "selected_identity"} or not _no_selected_identities(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_no_selected_identities(item) for item in value)
    return True


def _verify_prior_qwen_provenance(provenance: Any) -> None:
    """Bind the annotated protocol tag object separately from its commit."""
    expected = {
        "protocol_tag": QWEN_PROTOCOL_TAG,
        "protocol_tag_object_sha": QWEN_PROTOCOL_TAG_OBJECT_SHA,
        "protocol_commit": QWEN_PROTOCOL_COMMIT,
    }
    if not isinstance(provenance, Mapping) or any(provenance.get(key) != value for key, value in expected.items()):
        raise PhaseBError("prior Qwen protocol tag provenance differs")
    tag_object = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", QWEN_PROTOCOL_TAG], text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", f"{QWEN_PROTOCOL_TAG}^{{commit}}"], text=True
    ).strip()
    if tag_object != QWEN_PROTOCOL_TAG_OBJECT_SHA or commit != QWEN_PROTOCOL_COMMIT:
        raise PhaseBError("prior Qwen annotated tag does not resolve to its frozen identities")


def validate_protocol(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = dict(protocol() if payload is None else payload)
    if p.get("protocol_state") != "FROZEN-PRE-DATA" or not _no_selected_identities(p):
        raise PhaseBError("Phase-B protocol must be pre-data and contain no selected identities")
    target, geo = p.get("target_model"), p.get("geometry")
    if target != {"name": "microsoft/Phi-3.5-mini-instruct", "revision": REVISION, "target_family": "phi3"}:
        raise PhaseBError("Phi target/revision is not exact")
    _verify_prior_qwen_provenance(p.get("prior_qwen_provenance"))
    expected_geo = {"context_length": 8192, "sampled_layers": [0, 8, 16, 24, 31], "sampled_query_positions": [4095, 6143, 8191], "q_heads": 32, "kv_heads": 32, "head_dimension": 96, "kv_head_mapping": "kv_head = q_head", "static_schedule_cells": 160, "observations_per_source_per_method": 480, "observations_per_method_three_sources": 1440}
    if geo != expected_geo:
        raise PhaseBError("Phi geometry is not the frozen 8K MHA geometry")
    mapping = {"0": 0, "8": 7, "16": 14, "24": 21, "31": 27}
    profiles = p.get("routing_profiles", {})
    expected_profiles = {
        "phi35-8k-depth-transfer-qwen5-v1": ({"0": .25, "8": 1.5, "16": .5, "24": 1.5, "31": .25}, 16, 64, .05),
        "phi35-8k-depth-transfer-qwen10-v1": ({"0": 1., "8": 1., "16": 1., "24": .75, "31": .25}, 32, 128, .1),
    }
    if set(profiles) != set(expected_profiles):
        raise PhaseBError("routing profile identities differ")
    for name, (lambdas, sink, local, fraction) in expected_profiles.items():
        row = profiles[name]
        if row.get("depth_mapping") != mapping or row.get("lambdas") != lambdas or row.get("source_fraction") != fraction or row.get("reserve") != {"schema": "cascadekv-reserve-v1", "context_behavior": "absolute_tokens", "sink": sink, "local": local}:
            raise PhaseBError(f"transferred routing profile differs: {name}")
    actions = p.get("actions", {})
    expected_actions = {"A0": ("hierarchy", .05, "phi35-8k-depth-transfer-qwen5-v1"), "A1": ("hierarchy", .075, "phi35-8k-depth-transfer-qwen5-v1"), "A2": ("hierarchy", .10, "phi35-8k-depth-transfer-qwen5-v1"), "A3": ("hierarchy", .15, "phi35-8k-depth-transfer-qwen5-v1"), "A4": ("flat_q8k4", .05, None)}
    if set(actions) != set(ACTIONS) or any((actions[k].get("kind"), actions[k].get("candidate_fraction"), actions[k].get("routing_profile")) != v for k, v in expected_actions.items()):
        raise PhaseBError("action menu differs, including A2's required 5% profile")
    if p.get("uniform_hierarchy_10") != {"candidate_fraction": .1, "routing_profile": "phi35-8k-depth-transfer-qwen10-v1"}:
        raise PhaseBError("uniform hierarchy10 must use the transferred 10% profile")
    if p.get("quality_gates", {}).get("min_mean_cosine") != .985 or p["quality_gates"].get("max_mean_relative_l2") != .120:
        raise PhaseBError("quality gates differ")
    targets = p.get("bandwidth", {}).get("targets", {})
    if [targets.get(k) for k in TARGETS] != [.115, .130, .145, .160, .180, "highest_traffic_strictly_below_calibration_flat5"]:
        raise PhaseBError("bandwidth envelope differs")
    specs = p.get("source_specs", {})
    if [specs.get(k, {}).get("first_permitted_index") for k in ("narrative", "report", "qa")] != [13, 14, 13] or specs.get("report", {}).get("historically_ineligible") != [4]:
        raise PhaseBError("source frontiers differ")
    if p.get("development_inventory") != {"calibration": {"narrative": 2, "report": 2, "qa": 2}, "validation": {"narrative": 1, "report": 1, "qa": 1}}:
        raise PhaseBError("development inventory differs")
    return p


def verify_runtime_closure() -> tuple[dict[str, str], ...]:
    manifest = _load(RUNTIME_MANIFEST)
    if manifest.get("phase_a") != {"tag": PHASE_A_TAG, "commit": PHASE_A_COMMIT, "runtime_manifest_sha256": sha256(PHASE_A_MANIFEST)}:
        raise PhaseBError("Phase-B runtime manifest does not bind Phase A")
    entries = manifest.get("bound_files")
    if not isinstance(entries, list) or not entries:
        raise PhaseBError("Phase-B runtime manifest binds no files")
    output = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or not isinstance(entry["path"], str):
            raise PhaseBError("invalid Phase-B bound file entry")
        path = ROOT / entry["path"]
        if not path.is_file():
            raise PhaseBError(f"bound Phase-B file absent: {entry['path']}")
        if not _tracked(entry["path"]):
            raise PhaseBError(f"bound Phase-B file untracked: {entry['path']}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise PhaseBError(f"bound Phase-B file hash mismatch: {entry['path']}")
        output.append({"path": entry["path"], "sha256": actual})
    validate_protocol()
    verify_phase_a()
    return tuple(output)


def select_mechanically_inspected(family: str, inspections: Iterable[Mapping[str, Any]], *, unavailable: Iterable[int] = ()) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Choose 2 calibration + 1 validation slots from injected nonsemantic proofs.

    Callers, not this data-free module, perform permitted Phase-C row and
    tokenizer operations.  Records must contain no text and are retained as
    rejection proofs only.
    """
    spec = validate_protocol()["source_specs"].get(family)
    if not spec:
        raise PhaseBError("unknown source family")
    consumed = set(unavailable)
    accepted, rejected = [], []
    last_inspected_index: int | None = None
    for proof in sorted(inspections, key=lambda x: x.get("dataset_index", -1)):
        index = proof.get("dataset_index")
        if not isinstance(index, int) or index < spec["first_permitted_index"] or index in consumed:
            continue
        allowed = {"dataset_index", "field_exists", "is_python_string", "character_count", "bounded_tokenizer_length"}
        if set(proof) != allowed or not isinstance(proof["character_count"], int) or not isinstance(proof["bounded_tokenizer_length"], int):
            raise PhaseBError("mechanical inspection proof is malformed or contains text")
        last_inspected_index = index
        good = proof["field_exists"] and proof["is_python_string"] and proof["bounded_tokenizer_length"] >= 8192
        if good:
            accepted.append({"family": family, "dataset_index": index, "role": "calibration" if len(accepted) < 2 else "validation", "proof": dict(proof)})
            consumed.add(index)
            if len(accepted) == 3:
                break
        else:
            rejected.append({"dataset_index": index, "reason": "mechanically_ineligible", "proof": dict(proof)})
            consumed.add(index)
    if len(accepted) != 3:
        raise PhaseBError("insufficient mechanically eligible sources; do not substitute identities")
    assert last_inspected_index is not None
    return accepted, rejected, last_inspected_index


def optimize_static_schedule(groups: Iterable[Mapping[str, Mapping[str, float]]], target_mean_kv: float, *, strict: bool = False) -> tuple[float, float, float, tuple[str, ...]] | None:
    """Exact Qwen-style integer-microbyte DP, generalized to any cell count."""
    rows = list(groups)
    if len(rows) != 160 or not math.isfinite(target_mean_kv):
        raise PhaseBError("optimizer requires exactly 160 finite-target calibration cells")
    cap = math.ceil(target_mean_kv * len(rows) * 1000) - (1 if strict else 0)
    states: list[tuple[int, float, float, int]] = [(0, 0.0, 0.0, 0)]
    for row in rows:
        candidates = []
        for budget, error, cosine, code in states:
            for index, action in enumerate(ACTIONS):
                values = row.get(action, {})
                traffic, rel_l2, cos = values.get("total_kv_bytes"), values.get("relative_l2"), values.get("cosine")
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (traffic, rel_l2, cos)):
                    raise PhaseBError("optimizer action statistics are incomplete")
                next_budget = budget + round(traffic * 1000)
                if next_budget <= cap:
                    candidates.append((next_budget, error + rel_l2, cosine + cos, code * 5 + index))
        best_by_budget: dict[int, tuple[int, float, float, int]] = {}
        for candidate in sorted(candidates, key=lambda z: (z[0], z[1], -z[2], z[3])):
            best_by_budget.setdefault(candidate[0], candidate)
        states, best = [], None
        for candidate in best_by_budget.values():
            score = (candidate[1], -candidate[2])
            if best is None or score < best:
                states.append(candidate)
                best = score
    if not states:
        return None
    budget, error, cosine, code = min(states, key=lambda z: (z[1], -z[2], z[0], z[3]))
    table = []
    for _ in rows:
        code, action_index = divmod(code, 5)
        table.append(ACTIONS[action_index])
    return budget, error, cosine, tuple(reversed(table))


def runner_preflight(selected_source_manifest: Path | None = None) -> None:
    """Fail closed: Phase-B has no evaluator and cannot create scientific data."""
    verify_runtime_closure()
    if selected_source_manifest is None or not selected_source_manifest.is_file():
        raise PhaseBError("STOP Phase-C runner requires a future selected-source manifest")
    raise PhaseBError("STOP Phase-B runner is a scaffold only; no Phi capture/evaluation is implemented")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the data-free Phi-3.5 Phase-B protocol")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--selected-source-manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight:
            verify_runtime_closure()
            print("Phi35 Phase-B runtime/protocol closure verified")
        else:
            runner_preflight(args.selected_source_manifest)
    except PhaseBError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
