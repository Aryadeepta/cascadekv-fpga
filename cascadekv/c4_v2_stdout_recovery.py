"""Hash-gated recovery of C4-v2 evidence printed by the Kaggle stdout log.

This intentionally does not make an incomplete recovery look like a bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any

from . import c4_v2_bundle_postmortem as bundle

CLASS_BYTE = "BYTE_RECOVERED"
CLASS_FROZEN = "FROZEN_REPO_RECOVERED"
CLASS_ATTESTED = "STDOUT_ATTESTED_ONLY"
CLASS_MISSING = "MISSING_BYTES"
FROZEN = {
    "c4_protocol.json": "configs/cascadekv_phi35_8k_c4_protocol.json",
    "c4_runtime_manifest.json": "configs/cascadekv_phi35_8k_c4_runtime_manifest.json",
}
SCHEMAS = {
    "qualification.json": ("schema_version", "cascadekv-phi35-8k-kaggle-qualification-v2"),
    "development_capture_manifest.json": ("kind", "development"),
    "development.json": ("schema_version", "cascadekv-phi35-8k-c4-development-v1"),
    "holdout.json": ("schema_version", "cascadekv-phi35-8k-c4-holdout-selection-v1"),
    "holdout_capture_manifest.json": ("kind", "holdout"),
    "test.json": ("schema_version", "cascadekv-phi35-8k-c4-test-v1"),
}


class RecoveryError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_sha256s(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    active = False
    for line in text.splitlines():
        if line == "SHA256SUMS:":
            active = True
            continue
        if active:
            parts = line.split("  ")
            if len(parts) == 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
                rows[parts[1]] = parts[0]
            elif rows:
                break
    if set(rows) != set(bundle.REQUIRED):
        raise RecoveryError("stdout lacks the complete original SHA256SUMS block")
    return rows


def one_line_json_candidates(raw: bytes) -> list[tuple[int, bytes, dict[str, Any]]]:
    result = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not (line.startswith(b"{") and line.endswith(b"}")):
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            result.append((number, line, value))
    return result


def _matches(name: str, value: dict[str, Any]) -> bool:
    key, expected = SCHEMAS[name]
    return value.get(key) == expected


def recover(stdout: Path, output: Path, *, tag: str = "cascadekv-phi35-8k-c4-prep-v2") -> dict[str, Any]:
    raw = stdout.read_bytes()
    expected = parse_sha256s(raw.decode("utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    candidates = one_line_json_candidates(raw)
    for name in bundle.REQUIRED:
        record: dict[str, Any] = {"expected_sha256": expected[name], "classification": CLASS_MISSING,
                                  "recovered_path": None, "verified_recovered_sha256": None,
                                  "extraction": None, "candidate_byte_count": None, "candidate_sha256": None}
        if name in FROZEN:
            proc = subprocess.run(["git", "show", f"{tag}:{FROZEN[name]}"], check=True, stdout=subprocess.PIPE)
            data = proc.stdout
            digest = sha256_bytes(data)
            if digest != expected[name]:
                raise RecoveryError(f"frozen git bytes mismatch for {name}")
            path = output / name
            path.write_bytes(data)
            record.update(classification=CLASS_FROZEN, recovered_path=str(path), verified_recovered_sha256=digest,
                          extraction={"git_tag": tag, "git_path": FROZEN[name]}, candidate_byte_count=len(data), candidate_sha256=digest)
        elif name in SCHEMAS:
            found = [(line, data, value) for line, data, value in candidates if _matches(name, value)]
            # A unique printed object is required; only the literal line and literal line+LF are justified.
            if len(found) == 1:
                line, data, _ = found[0]
                for suffix, label in ((b"", "exact one-line JSON"), (b"\n", "exact one-line JSON plus LF")):
                    candidate = data + suffix
                    digest = sha256_bytes(candidate)
                    if digest == expected[name]:
                        path = output / name
                        path.write_bytes(candidate)
                        record.update(classification=CLASS_BYTE, recovered_path=str(path), verified_recovered_sha256=digest,
                                      extraction={"line": line, "variant": label}, candidate_byte_count=len(candidate), candidate_sha256=digest)
                        break
                else:
                    record.update(classification=CLASS_ATTESTED, extraction={"line": line, "variant": "literal candidate did not match"}, candidate_byte_count=len(data), candidate_sha256=sha256_bytes(data), reason="printed JSON is not authenticated original file bytes")
            else:
                record.update(classification=CLASS_MISSING, reason="no unique complete JSON payload printed")
        else:
            record.update(classification=CLASS_ATTESTED, reason="stdout attests its SHA and generation, but does not print complete serialized contents")
        rows[name] = record
    return {"schema_version": "cascadekv-c4-v2-stdout-recovery-v1", "original_c4_tag": tag,
            "original_c4_commit": "1d23e5e893614e4dc54d5b1988126680c791efb8", "raw_stdout": {"path": str(stdout), "sha256": sha256_bytes(raw), "byte_count": len(raw), "line_count": raw.count(b"\n")},
            "original_sha256sums": expected, "members": rows,
            "original_tar_gz_unavailable": True, "tar_reconstructed_byte_for_byte": False}


def _verified(manifest: dict[str, Any], name: str) -> Path:
    row = manifest["members"][name]
    if row["classification"] not in (CLASS_BYTE, CLASS_FROZEN) or not row["recovered_path"]:
        raise RecoveryError(f"quantitative postmortem requires authenticated {name}")
    path = Path(row["recovered_path"])
    if sha256_bytes(path.read_bytes()) != row["expected_sha256"]:
        raise RecoveryError(f"recovered bytes no longer authenticate for {name}")
    return path


def postmortem(manifest: dict[str, Any]) -> dict[str, Any]:
    dev = json.loads(_verified(manifest, "development.json").read_text())
    test = json.loads(_verified(manifest, "test.json").read_text())
    _verified(manifest, "c4_protocol.json")
    targets, bases = test["target_test_metrics"], test["baseline_test_metrics"]
    layers = bundle._layer_metrics(dev, "T5")
    dev_metrics = {t: {k: mean(row[k] for row in bundle._layer_metrics(dev, t).values()) for k in ("mean_cosine", "mean_relative_l2", "mean_total_kv_bytes")} for t in bundle.TARGETS}
    gates = {}
    deltas = {}
    for t in bundle.TARGETS:
        x, d = targets[t], dev_metrics[t]
        gates[t] = {"mean_cosine": x["mean_cosine"], "mean_relative_l2": x["mean_relative_l2"], "mean_modeled_kv": x["mean_total_kv_bytes"],
                    "cosine_margin": x["mean_cosine"] - .985, "relative_l2_margin": .120 - x["mean_relative_l2"],
                    "uniform10_margin": bases["uniform10"]["mean_relative_l2"] - x["mean_relative_l2"], "flat5_traffic_margin": bases["flat5"]["mean_total_kv_bytes"] - x["mean_total_kv_bytes"]}
        deltas[t] = {"holdout_minus_development_cosine": x["mean_cosine"] - d["mean_cosine"], "holdout_minus_development_relative_l2": x["mean_relative_l2"] - d["mean_relative_l2"], "holdout_minus_development_kv": x["mean_total_kv_bytes"] - d["mean_total_kv_bytes"], "relative_l2_percent_change": 100 * (x["mean_relative_l2"] / d["mean_relative_l2"] - 1)}
        stored = test["per_gate_pass_fail"][t]
        recomputed = {"feasible": True, "mean_cosine_gte": x["mean_cosine"] >= .985,
                      "mean_relative_l2_lte": x["mean_relative_l2"] <= .12,
                      "modeled_kv_traffic_lt_flat5": x["mean_total_kv_bytes"] < bases["flat5"]["mean_total_kv_bytes"],
                      "relative_l2_lt_uniform10": x["mean_relative_l2"] < bases["uniform10"]["mean_relative_l2"]}
        recomputed["passes"] = all(recomputed.values())
        if stored != recomputed:
            raise RecoveryError(f"stored per-gate table mismatch for {t}")
    if test["first_passing_target"] is not None or any(test["per_gate_pass_fail"][t]["passes"] for t in bundle.TARGETS):
        raise RecoveryError("first_passing_target mismatch")
    ordered = sorted(layers, key=lambda layer: layers[layer]["mean_relative_l2"], reverse=True)
    return {"schema_version": "cascadekv-c4-v2-stdout-recovery-postmortem-v1", "evidence_basis": "Recovered from preserved Kaggle stdout; original result tar archive unavailable.", "first_passing_target": test["first_passing_target"], "stored_gate_table_verified": True, "gate_margins": gates, "development_to_holdout": deltas, "t5_development_layers": layers, "t5_overall_action_histogram": {a: list(dev["schedules"]["T5"]["table"].values()).count(a) for a in bundle.ACTIONS}, "t5_worst_layers_by_relative_l2": ordered[:2], "optimizer": dev["optimizer"], "routing_profiles": dev["methods"]["routing_profiles"], "optimizer_conclusion": "Exact optimum within the frozen finite A0-A6 action space at each stated traffic cap; not a global optimum over all CascadeKV policies. Re-running deterministically on unchanged development statistics cannot improve its development relative-L2 objective.", "scientific_diagnosis": "Finite-action-space limitation is the leading explanation: the exact finite-space development optimum misses the absolute relative-L2 gate, while T5 holdout degradation is modest rather than catastrophic. Budget pressure also remains relevant because T5 is only 379.222 bytes below flat5 traffic.", "candidate_c5_hypothesis": "If evaluated later, add a small fixed set of hierarchy actions using already-frozen phi35-8k-depth-transfer-qwen10-v1 while retaining C4 candidate fractions; do not broaden the sweep or touch the protected frontier.", "untouched_frontier": {"narrative": 17, "qa": 18, "report": 20}}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# C4-v2 stdout-recovery postmortem", "", "**Recovered from preserved Kaggle stdout; original result tar archive unavailable.**", "", "The original tar archive was not reconstructed byte-for-byte.", "", "## Gate margins", "", "| Target | cosine margin | rel-L2 margin | uniform10 margin | flat5 traffic margin |", "|---|---:|---:|---:|---:|"]
    for t, x in report["gate_margins"].items(): lines.append(f"| {t} | {x['cosine_margin']:.9f} | {x['relative_l2_margin']:.9f} | {x['uniform10_margin']:.9f} | {x['flat5_traffic_margin']:.3f} |")
    lines += ["", f"first_passing_target: `{report['first_passing_target']}`.", "", "## T5 development layers", "", "| Layer | cosine | rel-L2 | KV | actions |", "|---:|---:|---:|---:|---|"]
    for layer, x in report["t5_development_layers"].items(): lines.append(f"| {layer} | {x['mean_cosine']:.9f} | {x['mean_relative_l2']:.9f} | {x['mean_total_kv_bytes']:.3f} | {x['action_histogram']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stdout", type=Path); p.add_argument("--recovery-dir", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--postmortem-json", type=Path, required=True); p.add_argument("--postmortem-md", type=Path, required=True)
    a = p.parse_args()
    m = recover(a.stdout, a.recovery_dir)
    a.manifest.parent.mkdir(parents=True, exist_ok=True); a.manifest.write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")
    r = postmortem(m)
    a.postmortem_json.parent.mkdir(parents=True, exist_ok=True); a.postmortem_json.write_text(json.dumps(r, sort_keys=True, indent=2) + "\n")
    a.postmortem_md.parent.mkdir(parents=True, exist_ok=True); a.postmortem_md.write_text(markdown(r))

if __name__ == "__main__": main()
