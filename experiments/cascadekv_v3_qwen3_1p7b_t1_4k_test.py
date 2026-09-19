#!/usr/bin/env python3
"""Pre-data-only harness for the immutable Qwen3-1.7B T1 confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cascadekv_v3_qwen3_1p7b_t1.json"
MANIFEST = ROOT / "results/cascadekv_v3_qwen3_1p7b_t1_4k_test_manifest.json"
DEVELOPMENT = ROOT / "results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev.json"
DEVELOPMENT_MANIFEST = ROOT / "results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_manifest.json"
CONFIG_SHA = "5995a0f7fd6c3ebe762f27959d10de0398980e1aa013b669379f6b794cf5e88e"
MANIFEST_SHA = "53af3b7dfe1dc7ccdd994851e4e22d329cf888f02c11d7f76e858735e106890b"
DEVELOPMENT_SHA = "16dad49612d5e02742e8b6cf75f7cac3f3a692668228fe36056beb32a02b4b2e"
DEVELOPMENT_MANIFEST_SHA = "654596830b06bfc6e9740c5363273bcbc408b769fabd92a69ef80fb182a84fb9"
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_CONFIG_SHA = "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
TAG = "cascadekv-v3-qwen3-1p7b-wide-vaware-dev-protocol-freeze"
TAG_COMMIT = "f0c5b5666ad089531fca68862c5be959088d49ef"
LAYERS = (0, 7, 14, 21, 27)
METHODS = ("dense_exact", "flat_q8k4_5", "flat_q8k4_10", "uniform_v1_5", "uniform_v1_10", "frozen_cascadekv_v2", "original_frozen_cascadekv_v3_zero_shot", "frozen_qwen3_1p7b_T1")
CACHE_DIR = ROOT / "results/cascadekv_v3_qwen3_1p7b_t1_4k_test_cache"
SHARD_DIR = ROOT / "results/cascadekv_v3_qwen3_1p7b_t1_4k_test_shards"
RESULT = ROOT / "results/cascadekv_v3_qwen3_1p7b_t1_4k_test.json"
EXPECTED_TABLE = {"0:0":"A3","0:1":"A0","0:2":"A0","0:3":"A1","0:4":"A1","0:5":"A2","0:6":"A0","0:7":"A2","7:0":"A1","7:1":"A0","7:2":"A3","7:3":"A2","7:4":"A0","7:5":"A0","7:6":"A0","7:7":"A3","14:0":"A4","14:1":"A1","14:2":"A1","14:3":"A4","14:4":"A0","14:5":"A4","14:6":"A1","14:7":"A4","21:0":"A3","21:1":"A4","21:2":"A1","21:3":"A0","21:4":"A4","21:5":"A4","21:6":"A4","21:7":"A1","27:0":"A1","27:1":"A0","27:2":"A0","27:3":"A2","27:4":"A0","27:5":"A0","27:6":"A1","27:7":"A1"}
LABELS = {"invalid":"T1-CONFIRMATORY-TEST-INVALID", "absolute_fail":"T1-CONFIRMATORY-OUTPUT-GATE-FAILED", "comparative_fail":"T1-CONFIRMATORY-COMPARATIVE-GATE-FAILED", "pass":"T1-CONFIRMATORY-TEST-PASSED"}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(source: dict[str, Any]) -> tuple[Any, ...]:
    return (source["dataset"], source.get("config"), source["split"], source["index"], source.get("stable_example_id"))

def require_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, digest in ((CONFIG, CONFIG_SHA), (MANIFEST, MANIFEST_SHA), (DEVELOPMENT, DEVELOPMENT_SHA), (DEVELOPMENT_MANIFEST, DEVELOPMENT_MANIFEST_SHA)):
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"T1-CONFIRMATORY-TEST-INVALID: immutable hash mismatch: {path.name}")
    if subprocess.check_output(["git", "rev-parse", f"{TAG}^{{commit}}"], text=True).strip() != TAG_COMMIT:
        raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: development tag mismatch")
    config, manifest, development = (json.loads(path.read_text()) for path in (CONFIG, MANIFEST, DEVELOPMENT))
    if (development.get("status"), development.get("classification"), development.get("winner_target_name")) != ("complete", "1P7B-WIDE-V-AWARE-STATIC-PROMISING", "target_wide_T1"):
        raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: development winner provenance")
    if development.get("winner_table") != EXPECTED_TABLE or config.get("exact_layer_head_action_table") != EXPECTED_TABLE:
        raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: frozen T1 table mismatch")
    if config.get("schedule_cell_count") != 40 or {a: list(EXPECTED_TABLE.values()).count(a) for a in ("A0", "A1", "A2", "A3", "A4")} != config.get("composition"):
        raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: schedule composition")
    if config.get("target_model") != {"name": MODEL, "revision": REVISION, "config_sha256": MODEL_CONFIG_SHA}:
        raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: target model")
    if config["action_semantics"]["A2"] != {"candidate_budget": .10, "routing_profile": "frozen 5%-routing profile", "mode": "hierarchy"}:
        raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: A2 semantics")
    return config, manifest, development

def load_manifest() -> dict[str, Any]:
    _, manifest, _ = require_inputs()
    if manifest.get("target_t1_config", {}).get("sha256") != CONFIG_SHA or tuple(manifest.get("methods", ())) != METHODS:
        raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: manifest bindings")
    sources = manifest.get("sources", {})
    if set(sources) != {"narrative", "report", "qa"} or len({identity(x) for x in sources.values()}) != 3:
        raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: sources")
    inventory = manifest.get("complete_consumed_identity_inventory", {})
    for family, source in sources.items():
        if source["index"] in inventory[family]["unavailable_indices"] or not source.get("eligibility", {}).get("at_least_4096"):
            raise RuntimeError("T1-CONFIRMATORY-TEST-INVALID: consumed or unproven source")
    return manifest

def preflight() -> dict[str, Any]:
    """Load-and-validate only. Selection is deliberately not implemented here."""
    return load_manifest()

def status() -> dict[str, Any]:
    load_manifest()
    # This preparation-only harness deliberately treats all artifacts as absent;
    # it has no capture/evaluate/merge implementation and never opens data/models.
    return {"manifest_sha256": MANIFEST_SHA, "valid_caches": 0, "total_caches": 15, "valid_shards": 0, "total_shards": 15, "result_exists": RESULT.exists(), "execution_enabled": False}

def classify(pooled: dict[str, Any]) -> str:
    try:
        t1, v2, v3, flat5 = (pooled[x] for x in ("frozen_qwen3_1p7b_T1", "frozen_cascadekv_v2", "original_frozen_cascadekv_v3_zero_shot", "flat_q8k4_5"))
        cosine = t1["metrics"]["cosine_similarity"]["mean"]; l2 = t1["metrics"]["relative_l2_error"]["mean"]; traffic = t1["traffic"]["total_kv_bytes"]["mean"]
        if cosine < .985 or l2 > .120: return LABELS["absolute_fail"]
        if l2 >= v2["metrics"]["relative_l2_error"]["mean"] or l2 >= v3["metrics"]["relative_l2_error"]["mean"] or traffic >= flat5["traffic"]["total_kv_bytes"]["mean"]: return LABELS["comparative_fail"]
        return LABELS["pass"]
    except (KeyError, TypeError):
        return LABELS["invalid"]

def main() -> None:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true"); group.add_argument("--status", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight() if args.preflight else status(), indent=2))

if __name__ == "__main__": main()
