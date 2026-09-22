"""Fail-closed, bundle-only archival validation and C4-v2 postmortem.

This module intentionally imports only the Python standard library.  It never
opens a capture payload, model, tokenizer, dataset, or network connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

REQUIRED = (
    "qualification.json", "cascadekv_phi35_8k_dev_sources_v2.json",
    "development_capture_manifest.json", "development.json", "holdout.json",
    "holdout_capture_manifest.json", "test.json", "c4_protocol.json",
    "c4_runtime_manifest.json",
)
EXPECTED = {
    "qualification.json": "f4703bdcd477ee91f108dde4415e14831ec20a8b156fe59876d8e3a7699bc07e",
    "cascadekv_phi35_8k_dev_sources_v2.json": "31f1d14d4b33b6e5b6a48d34e294f0effbab2eeadf6c4d4eabfabe16ae879478",
    "development.json": "964ca02923b0dc2438cb5041c7027bd00396ff10170e012b451d3a57eb4c4958",
    "holdout.json": "45e6c2d09c53be9a4053c696349011786c9bb934ce49aea15561b7c92eaeeee6",
    "holdout_capture_manifest.json": "781c912c389dce576685d64c8f96a1cc8f134106fd294d82f6b23588ee99fbd3",
    "test.json": "945b54ec8307a4e590262b5945a98d897bddeffb6ea9b341b5097c7f1b75ef3c",
    "c4_protocol.json": "6e2f7c671d04f5dd650d71e62de23e8fff0697c32dfc837e2921535edfc59c1d",
    "c4_runtime_manifest.json": "76cd263efed206334b9de8e02c3355536893c565bdb20b12901567b8215bd589",
}
TARGETS = ("T0", "T1", "T2", "T3", "T4", "T5")
LAYERS, HEADS, ACTIONS = (0, 8, 16, 24, 31), tuple(range(32)), ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
HOLDOUT = {"narrative": 16, "report": 19, "qa": 17}
FRONTIER = {"narrative": 17, "report": 20, "qa": 18}


class BundleError(RuntimeError):
    """A bundle is incomplete, malformed, or does not bind the frozen C4 run."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"JSON root is not an object: {path.name}")
    return value


def _no_raw_text(value: Any) -> None:
    if isinstance(value, Mapping):
        if {"raw_text", "text", "document", "prompt", "content", "value"} & set(value):
            raise BundleError("persisted raw source text field is forbidden")
        for child in value.values():
            _no_raw_text(child)
    elif isinstance(value, list):
        for child in value:
            _no_raw_text(child)


def _sha_sums(root: Path) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    if not sums.is_file() or sums.is_symlink():
        raise BundleError("SHA256SUMS is required")
    rows: dict[str, str] = {}
    try:
        lines = sums.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise BundleError("SHA256SUMS is not ASCII") from exc
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match or match.group(2) in rows:
            raise BundleError("SHA256SUMS has an unsafe or duplicate entry")
        rows[match.group(2)] = match.group(1)
    if set(rows) != set(REQUIRED):
        raise BundleError("SHA256SUMS file set differs from the required bundle")
    return rows


def _schedule_sha(schedules: Any) -> str:
    return hashlib.sha256(_canon(schedules)).hexdigest()


def _identities(kind: str, holdout: Mapping[str, Any] | None = None) -> list[str]:
    if kind == "development":
        indices = {"narrative": (13, 14, 15), "report": (16, 17, 18), "qa": (13, 15, 16)}
    else:
        assert holdout is not None
        indices = {x["family"]: (x["dataset_index"],) for x in holdout["selected_sources"]}
    return sorted(f"{family}:{index}:{kind}:L{layer}" for family, values in indices.items()
                  for index in values for layer in LAYERS)


def _metric(row: Mapping[str, Any]) -> tuple[float, float, float]:
    try:
        result = (float(row["mean_cosine"]), float(row["mean_relative_l2"]),
                  float(row["mean_total_kv_bytes"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError("invalid metric row") from exc
    if not all(math.isfinite(x) for x in result):
        raise BundleError("non-finite metric")
    return result


def validate_bundle(root: Path, *, expected: Mapping[str, str] = EXPECTED) -> dict[str, Any]:
    """Validate an extracted C4 bundle.  No malformed bundle yields analysis."""
    root = root.resolve()
    if not root.is_dir():
        raise BundleError("bundle input must be an extracted directory")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in REQUIRED):
        raise BundleError("required file missing or symlinked")
    sums = _sha_sums(root)
    for name in REQUIRED:
        digest = _sha(root / name)
        if sums[name] != digest:
            raise BundleError(f"SHA256SUMS mismatch: {name}")
        if name in expected and expected[name] != digest:
            raise BundleError(f"frozen observed SHA mismatch: {name}")
    p = {name: _load(root / name) for name in REQUIRED if name.endswith(".json")}
    for value in p.values():
        _no_raw_text(value)
    protocol, runtime = p["c4_protocol.json"], p["c4_runtime_manifest.json"]
    if protocol.get("quality_gates") != {"mean_cosine_gte": .985, "mean_relative_l2_lte": .12,
                                          "winner_relative_l2_lt": "uniform10", "winner_modeled_kv_traffic_lt": "flat5"}:
        raise BundleError("frozen gate values differ")
    if protocol.get("targets", {}).get("order") != list(TARGETS):
        raise BundleError("frozen target order differs")
    actions = protocol.get("methods", {}).get("actions", {})
    if list(actions) != list(ACTIONS) or actions.get("A4", {}).get("kind") != "flat_q8k4":
        raise BundleError("frozen action menu differs")
    if runtime.get("protocol") != {"path": "configs/cascadekv_phi35_8k_c4_protocol.json", "sha256": _sha(root / "c4_protocol.json")}:
        raise BundleError("runtime/protocol binding differs")
    development_capture = p["development_capture_manifest.json"]
    if (development_capture.get("kind") != "development" or
            development_capture.get("protocol_sha256") != _sha(root / "c4_protocol.json") or
            development_capture.get("qualification_sha256") != _sha(root / "qualification.json") or
            development_capture.get("source_manifest_sha256") != _sha(root / "cascadekv_phi35_8k_dev_sources_v2.json") or
            development_capture.get("artifact_count") != 45 or development_capture.get("forwards") != 9 or
            any(development_capture.get(key) is not None for key in ("holdout_manifest_sha256", "development_result_sha256", "development_schedule_sha256"))):
        raise BundleError("development capture provenance or inventory differs")
    dev, holdout, capture, test = (p["development.json"], p["holdout.json"],
                                   p["holdout_capture_manifest.json"], p["test.json"])
    if (dev.get("protocol_sha256") != _sha(root / "c4_protocol.json") or
            dev.get("runtime_manifest_sha256") != _sha(root / "c4_runtime_manifest.json") or
            dev.get("development_source_manifest_sha256") != _sha(root / "cascadekv_phi35_8k_dev_sources_v2.json") or
            dev.get("capture_manifest_sha256") != _sha(root / "development_capture_manifest.json")):
        raise BundleError("development provenance binding differs")
    if dev.get("schedule_sha256") != _schedule_sha(dev.get("schedules")):
        raise BundleError("development schedule digest differs")
    if dev.get("opened_development_tensor_identities") != _identities("development") or dev.get("opened_holdout_tensor_identities") != []:
        raise BundleError("development firewall differs")
    if holdout.get("development_result_sha256") != _sha(root / "development.json") or holdout.get("development_schedule_sha256") != dev["schedule_sha256"]:
        raise BundleError("holdout/development binding differs")
    if (holdout.get("protocol_sha256") != _sha(root / "c4_protocol.json") or
            holdout.get("development_source_manifest_sha256") != _sha(root / "cascadekv_phi35_8k_dev_sources_v2.json")):
        raise BundleError("holdout provenance binding differs")
    selected = holdout.get("selected_sources")
    if not isinstance(selected, list) or {(x.get("family"), x.get("dataset_index")) for x in selected} != set(HOLDOUT.items()):
        raise BundleError("holdout identities differ")
    if holdout.get("next_untouched_frontier") != FRONTIER or holdout.get("raw_text_persisted") is not False:
        raise BundleError("holdout frontier or text boundary differs")
    if capture.get("holdout_manifest_sha256") != _sha(root / "holdout.json") or capture.get("development_result_sha256") != _sha(root / "development.json") or capture.get("development_schedule_sha256") != dev["schedule_sha256"]:
        raise BundleError("holdout capture binding differs")
    if (capture.get("kind") != "holdout" or capture.get("protocol_sha256") != _sha(root / "c4_protocol.json") or
            capture.get("qualification_sha256") != _sha(root / "qualification.json") or
            capture.get("source_manifest_sha256") is not None or
            capture.get("artifact_count") != 15 or capture.get("forwards") != 3):
        raise BundleError("holdout capture inventory differs")
    if test.get("development_result_sha256") != _sha(root / "development.json") or test.get("development_schedule_sha256") != dev["schedule_sha256"] or test.get("holdout_manifest_sha256") != _sha(root / "holdout.json") or test.get("capture_manifest_sha256") != _sha(root / "holdout_capture_manifest.json"):
        raise BundleError("test binding differs")
    if test.get("target_order") != list(TARGETS) or test.get("gate_values") != protocol["quality_gates"]:
        raise BundleError("test target order or gates differ")
    if (test.get("protocol_sha256") != _sha(root / "c4_protocol.json") or
            test.get("runtime_manifest_sha256") != _sha(root / "c4_runtime_manifest.json")):
        raise BundleError("test protocol/runtime binding differs")
    if test.get("opened_development_tensor_identities") != [] or test.get("opened_holdout_tensor_identities") != _identities("holdout", holdout):
        raise BundleError("prospective tensor firewall differs")
    proof = test.get("no_schedule_mutation_evidence")
    if proof != {"schedule_sha256_before": dev["schedule_sha256"], "schedule_sha256_after": dev["schedule_sha256"], "schedule_unchanged": True, "optimizer_rerun": False}:
        raise BundleError("no-schedule-mutation evidence differs")
    if test.get("first_passing_target") is not None and test["first_passing_target"] not in TARGETS:
        raise BundleError("invalid first passing target")
    for row in [*test.get("baseline_test_metrics", {}).values(), *test.get("target_test_metrics", {}).values()]:
        _metric(row)
    if set(test.get("target_test_metrics", {})) != set(TARGETS):
        raise BundleError("test target metrics differ")
    return {"root": root, "files": p, "sha256": {name: _sha(root / name) for name in REQUIRED}}


def _layer_metrics(dev: Mapping[str, Any], target: str) -> dict[str, Any]:
    table, stats = dev["schedules"][target]["table"], dev["action_cell_statistics"]
    result: dict[str, Any] = {}
    for layer in LAYERS:
        selected = [table[f"{layer}:{head}"] for head in HEADS]
        cells = [stats[f"{layer}:{head}:{action}"] for head, action in zip(HEADS, selected, strict=True)]
        result[str(layer)] = {"mean_cosine": mean(float(x["cosine"]) for x in cells),
                              "mean_relative_l2": mean(float(x["relative_l2"]) for x in cells),
                              "mean_total_kv_bytes": mean(float(x["total_kv_bytes"]) for x in cells),
                              "action_histogram": {a: selected.count(a) for a in ACTIONS}}
    return result


def postmortem(validated: Mapping[str, Any]) -> dict[str, Any]:
    """Produce only values mechanically derived from a validated bundle."""
    p = validated["files"]
    dev, test = p["development.json"], p["test.json"]
    base, targets = test["baseline_test_metrics"], test["target_test_metrics"]
    u10, f5, dense = (_metric(base["uniform10"]), _metric(base["flat5"]), _metric(base["dense"]))
    gates: dict[str, Any] = {}
    for target in TARGETS:
        cosine, error, traffic = _metric(targets[target])
        margins = {"cosine_minus_gate": cosine - .985, "relative_l2_gate_minus_value": .12 - error,
                   "uniform10_relative_l2_minus_target": u10[1] - error, "flat5_traffic_minus_target": f5[2] - traffic}
        failed = [name for name, value in margins.items() if value < 0]
        gates[target] = {"margins": margins, "failed_gates": failed, "passes": not failed}
    closest = min(TARGETS, key=lambda t: (len(gates[t]["failed_gates"]), -min(gates[t]["margins"].values()), TARGETS.index(t)))
    t5 = _metric(targets["T5"])
    layers = _layer_metrics(dev, "T5")
    action_counts = {target: {a: list(dev["schedules"][target]["table"].values()).count(a) for a in ACTIONS} for target in TARGETS}
    dev_targets = {target: {"mean_cosine": mean(x["mean_cosine"] for x in _layer_metrics(dev, target).values()),
                            "mean_relative_l2": mean(x["mean_relative_l2"] for x in _layer_metrics(dev, target).values()),
                            "mean_total_kv_bytes": mean(x["mean_total_kv_bytes"] for x in _layer_metrics(dev, target).values())} for target in TARGETS}
    delta = {t: {"holdout_minus_development_cosine": _metric(targets[t])[0] - dev_targets[t]["mean_cosine"],
                 "holdout_minus_development_relative_l2": _metric(targets[t])[1] - dev_targets[t]["mean_relative_l2"],
                 "holdout_minus_development_kv": _metric(targets[t])[2] - dev_targets[t]["mean_total_kv_bytes"]} for t in TARGETS}
    return {"classification": "C4-V2-PROSPECTIVE-NO-PASS" if test["first_passing_target"] is None else "C4-V2-PROSPECTIVE-PASS",
            "bundle_sha256": validated["sha256"], "first_passing_target": test["first_passing_target"],
            "gate_margins": gates, "closest_target": {"target": closest, "rule": "fewest failed gates; then greatest worst margin; then target order"},
            "t5_relative_comparison": {"relative_l2_improvement_vs_flat5": f5[1] - t5[1], "relative_l2_improvement_vs_uniform10": u10[1] - t5[1], "traffic_change_vs_flat5": t5[2] - f5[2], "traffic_fraction_of_dense": t5[2] / dense[2], "relative_l2_excess_above_absolute_gate": t5[1] - .12},
            "development_to_holdout": delta, "development_action_counts": action_counts,
            "t5_development_layers": layers,
            "t5_action_saturation": {"a6_layers_with_cells": [layer for layer, row in layers.items() if row["action_histogram"]["A6"] > 0], "a4_layers_with_cells": [layer for layer, row in layers.items() if row["action_histogram"]["A4"] > 0]},
            "next_untouched_frontier": p["holdout.json"]["next_untouched_frontier"]}


def markdown(report: Mapping[str, Any]) -> str:
    closest, t5 = report["closest_target"]["target"], report["t5_relative_comparison"]
    lines = ["# C4-v2 bundle-only postmortem", "", f"Classification: `{report['classification']}`.",
             f"First passing target: `{report['first_passing_target']}`. Closest target: `{closest}` ({report['closest_target']['rule']}).", "",
             "## Gate margins", "", "| Target | Cosine − gate | Gate − relative-L2 | Uniform10 L2 − target | Flat5 traffic − target | Failed |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for target, row in report["gate_margins"].items():
        m = row["margins"]; lines.append(f"| {target} | {m['cosine_minus_gate']:.12g} | {m['relative_l2_gate_minus_value']:.12g} | {m['uniform10_relative_l2_minus_target']:.12g} | {m['flat5_traffic_minus_target']:.12g} | {', '.join(row['failed_gates']) or 'none'} |")
    lines += ["", "## Supported diagnosis", "", f"T5 relative-L2 excess above the absolute gate: {t5['relative_l2_excess_above_absolute_gate']:.12g}. Its traffic fraction of dense: {t5['traffic_fraction_of_dense']:.12g}.", "The bundle proves execution bindings and gate outcomes; it does not by itself establish a causal optimizer or action-space failure.", "", "## Next-pass boundary", "", "Any C5 action menu and protocol must be frozen before inspecting the recorded next untouched frontier. A frozen alternative routing profile, if present in the validated protocol, is an additive experiment only; it must not mutate C4."]
    return "\n".join(lines) + "\n"


def _extract(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir(): return path, None
    if not tarfile.is_tarfile(path): raise BundleError("input must be a bundle directory or tar archive")
    tmp = tempfile.TemporaryDirectory(prefix="c4-v2-bundle-")
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        if any(not x.isfile() or Path(x.name).is_absolute() or ".." in Path(x.name).parts for x in members):
            tmp.cleanup(); raise BundleError("unsafe tar member")
        archive.extractall(tmp.name, members)
    children = list(Path(tmp.name).iterdir())
    return (children[0] if len(children) == 1 and children[0].is_dir() else Path(tmp.name)), tmp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    temp = None
    try:
        root, temp = _extract(args.bundle); result = postmortem(validate_bundle(root))
        args.json_output.write_bytes(_canon(result) + b"\n")
        args.report_output.write_text(markdown(result), encoding="utf-8")
        return 0
    except (BundleError, OSError, ValueError) as exc:
        print(f"C4-V2-BUNDLE-POSTMORTEM-FAILED: {exc}", file=__import__("sys").stderr); return 1
    finally:
        if temp: temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
