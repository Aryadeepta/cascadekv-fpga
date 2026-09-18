#!/usr/bin/env python3
"""Frozen, data-free protocol harness for the wide-band V-aware development.

This module intentionally has no capture, model, dataset, evaluation, merge, or
schedule-optimization entry point.  Its small pure functions are used only by
synthetic protocol regressions to lock the prospective rules before data work.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-1.7B"
MODEL_SHA = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CONFIG_SHA = "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
V1 = Path("configs/cascadekv_v1.json"); V1_SHA = "2964e4295719e696587f9853177c38399b64edc753e0630fc0d44208c719559b"
V2 = Path("configs/cascadekv_v2.json"); V2_SHA = "ef36c5bd1b4211083ac5680e554877281f0a09e37f3dea9e4eb877afba18259c"
V3 = Path("configs/cascadekv_v3.json"); V3_SHA = "5d364bc4e351061c243a9166c0a9ccdd68c7cefd74265e4135d937b323e6e505"
NARROW = Path("results/cascadekv_v3_qwen3_1p7b_vaware_dev.json")
NARROW_SHA = "9e847fd4fb97c6eaa996df9661ed80e8219d149c02cd311e83fcebff580789ac"
MANIFEST = Path("results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_manifest.json")
MANIFEST_SHA = "654596830b06bfc6e9740c5363273bcbc408b769fabd92a69ef80fb182a84fb9"
ACTIONS = ("A0", "A1", "A2", "A3", "A4")
ACTION_SEMANTICS = {"A0": [.05, "hierarchy using frozen 5%-routing profile"], "A1": [.075, "hierarchy using frozen 5%-routing profile"], "A2": [.10, "hierarchy using frozen 5%-routing profile"], "A3": [.15, "hierarchy using frozen 5%-routing profile"], "A4": [.05, "flat"]}
TARGET_NAMES = tuple(f"target_wide_T{i}" for i in range(6))

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(x): return (x["dataset"], x.get("config"), x["split"], x["index"], x.get("stable_example_id"))

def load_manifest():
    if sha(MANIFEST) != MANIFEST_SHA: raise RuntimeError("STOP frozen wide manifest hash differs")
    return json.loads(MANIFEST.read_text())

def require_inputs():
    for path, digest in ((V1,V1_SHA),(V2,V2_SHA),(V3,V3_SHA),(NARROW,NARROW_SHA)):
        if not path.is_file() or sha(path) != digest: raise RuntimeError(f"STOP immutable input hash differs: {path}")
    narrow = json.loads(NARROW.read_text())
    if narrow.get("status") != "complete" or narrow.get("classification") != "1P7B-V-AWARE-STATIC-NOT-PROMISING" or narrow.get("winner_target_name") is not None:
        raise RuntimeError("STOP narrow development provenance state differs")
    d = load_manifest()
    if d["target_model"]["resolved_commit_sha"] != MODEL_SHA or d["target_model"]["config_sha256"] != CONFIG_SHA: raise RuntimeError("STOP target binding")
    return d

def wide_targets(b_v2, b_u10, b_flat5):
    if not b_v2 < b_u10 < b_flat5: raise RuntimeError("WIDE-TARGET-BASELINE-ORDER-INVALID")
    return {"T0":b_v2, "T1":b_v2+.5*(b_u10-b_v2), "T2":b_u10,
            "T3":b_u10+(b_flat5-b_u10)/3, "T4":b_u10+2*(b_flat5-b_u10)/3, "T5":b_flat5}

def passes(candidate, flat5_kv, v2_rel_l2, v3_rel_l2):
    return (candidate["cosine"] >= .985 and candidate["relative_l2"] <= .120
            and candidate["kv"] < flat5_kv and candidate["relative_l2"] < v2_rel_l2
            and candidate["relative_l2"] < v3_rel_l2)

def choose_winner(candidates, flat5_kv, v2_rel_l2, v3_rel_l2):
    if not isinstance(candidates, list) or len(candidates) != 6: return None
    by = {}
    for row in candidates:
        if not isinstance(row, dict) or row.get("name") in by or row.get("name") not in TARGET_NAMES: return None
        by[row["name"]] = row
    if set(by) != set(TARGET_NAMES): return None
    try: return next((by[name] for name in TARGET_NAMES if passes(by[name],flat5_kv,v2_rel_l2,v3_rel_l2)), None)
    except (KeyError, TypeError): return None

def status():
    # Safe by construction: preparation never creates caches/shards/results.
    require_inputs()
    return {"manifest_sha256": MANIFEST_SHA, "valid_caches": 0, "total_caches": 30,
            "valid_action_shards": 0, "total_action_shards": 30,
            "result_exists": Path("results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev.json").exists()}

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument("--preflight",action="store_true"); g.add_argument("--status",action="store_true"); a=p.parse_args()
    print(json.dumps(require_inputs() if a.preflight else status(), indent=2, sort_keys=True))
if __name__ == "__main__": main()
