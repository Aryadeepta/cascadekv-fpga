#!/usr/bin/env python3
"""Predeclared zero-shot Qwen3-1.7B transfer harness for frozen CascadeKV-v3.

The only source-construction operation is ``--preflight``.  It never loads
model weights.  Capture/evaluation are separate, explicit future phases.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import cascadekv_v2_4k_test as base

MODEL = "Qwen/Qwen3-1.7B"
MODEL_SHA = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
V2_CONFIG = Path("configs/cascadekv_v2.json")
V3_CONFIG = Path("configs/cascadekv_v3.json")
MANIFEST = Path("results/cascadekv_v3_qwen3_1p7b_4k_test_manifest.json")
V3_TEST_RESULT = Path("results/cascadekv_v3_4k_test.json")
V3_SHA = "5d364bc4e351061c243a9166c0a9ccdd68c7cefd74265e4135d937b323e6e505"
V3_TEST_SHA = "01c0bbda902a723599b709c8bc6be159c6191706a9c3531fd2da5fbb3fe5f09d"
V3_TEST_MANIFEST_SHA = "2181ca5b7b501926479901b4b56098481adbb2fe790183919e926e54c4e629aa"
V2_SHA = "ef36c5bd1b4211083ac5680e554877281f0a09e37f3dea9e4eb877afba18259c"
TEST_MANIFEST_SHA = "20927a2a34bb5b03faf622b0c44195e50f7a482942a3f08da0cc80ecaa568574"
LAYERS = (0, 7, 14, 21, 27)
POSITIONS = (2047, 3071, 4095)
SPECS = base.SPECS
METHODS = ("dense_exact", "flat_q8k4_5", "flat_q8k4_10", "uniform_v1_5", "uniform_v1_10", "frozen_cascadekv_v2", "frozen_cascadekv_v3")
METRIC_FIELDS = ("relative_exact_attention_mass", "top8_recall", "cosine_similarity", "relative_l2_error", "absolute_l2_error")
TRAFFIC_FIELDS = ("routing_index_bytes", "candidate_sketch_bytes", "authoritative_k_bytes", "selected_v_fp16_bytes", "total_k_bytes", "total_kv_bytes")
REQUIRED_ARCHITECTURE = {"model_type":"qwen3", "num_hidden_layers":28, "num_attention_heads":16, "num_key_value_heads":8, "head_dim":128, "hidden_size":2048}
CLASSIFICATION = {"absolute":{"pooled_cosine_gte":.985,"pooled_relative_l2_lte":.12},"comparative":{"v3_kv_bytes_lte_uniform10":True,"v3_relative_l2_lt_v2":True},"outcomes":{"pass":"TARGET-MODEL-TRANSFER-PASSED","absolute_fail":"TARGET-MODEL-OUTPUT-GATE-FAILED","comparative_fail":"TARGET-MODEL-COMPARATIVE-GATE-FAILED","invalid":"TARGET-MODEL-TEST-INVALID"},"diagnostics":"report-only and never classification inputs"}

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(x: dict[str, Any]) -> tuple[Any, ...]: return tuple(x.get(k) for k in ("dataset", "config", "split", "index", "stable_example_id"))

def require_inputs() -> None:
    for path, digest in ((V3_CONFIG,V3_SHA),(V3_TEST_RESULT,V3_TEST_SHA),(Path("results/cascadekv_v3_4k_test_manifest.json"),V3_TEST_MANIFEST_SHA),(V2_CONFIG,V2_SHA)):
        if not path.is_file() or sha(path) != digest: raise RuntimeError(f"STOP immutable input hash differs: {path}")
    parent=json.loads(V3_TEST_RESULT.read_text())
    if (parent.get("status"),parent.get("classification")) != ("complete","TEST-PASSED"): raise RuntimeError("STOP required parent result is not complete TEST-PASSED")
    cfg=json.loads(V3_CONFIG.read_text())
    table=cfg.get("exact_layer_head_action_table",{})
    if cfg.get("target_name") != "T0" or len(table) != 40 or set(table) != {f"({l}, {h})" for l in LAYERS for h in range(8)}: raise RuntimeError("STOP frozen v3 T0 table differs")

def verify_architecture(config: dict[str, Any]) -> None:
    actual={k:config.get(k) for k in REQUIRED_ARCHITECTURE}
    if actual != REQUIRED_ARCHITECTURE: raise RuntimeError(f"STOP target routing geometry differs: {actual}")

def load_manifest() -> dict[str, Any]:
    require_inputs()
    if not MANIFEST.is_file() or sha(MANIFEST) != TEST_MANIFEST_SHA: raise RuntimeError("STOP frozen target manifest hash differs before parse/use")
    d=json.loads(MANIFEST.read_text()); verify_architecture(d.get("target_model",{}).get("architecture",{}))
    if d.get("target_model",{}).get("config_sha256") != "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197": raise RuntimeError("STOP recorded target config differs")
    if d.get("target_model",{}).get("resolved_commit_sha") != MODEL_SHA or d.get("context_length") != 4096 or tuple(d.get("methods",())) != METHODS: raise RuntimeError("STOP target manifest binding differs")
    used={identity(x) for x in d["complete_prior_explicit_identity_inventory"]}
    for name,x in d["sources"].items():
        if name not in SPECS or identity(x) in used or not x.get("token_length_proof",{}).get("at_least"): raise RuntimeError("STOP consumed or unproven selected identity")
    return d

def preflight() -> dict[str, Any]: return load_manifest() # Existing manifests are never reselected.
def cache_path(root: Path,s: str,l: int) -> Path: return root/f"{s}_L4096_layer{l}_qkv.pt"
def shard_path(root: Path,s: str,l: int) -> Path: return root/f"{s}_L4096_layer{l}.json"

def validate_cache(path: Path,s: str,l: int) -> tuple[bool,str]:
    try:
        x=torch.load(path,map_location="cpu",weights_only=False); m=x["metadata"]; mf=load_manifest(); q,k,v=(x[n] for n in ("query","key","value"))
        expected=(TEST_MANIFEST_SHA,MODEL_SHA,V2_SHA,V3_SHA,s,l,4096,mf["sources"][s])
        actual=(m.get("manifest_sha256"),m.get("model_sha"),m.get("v2_config_sha256"),m.get("v3_config_sha256"),m.get("sequence"),m.get("layer"),m.get("context_length"),m.get("source_identity"))
        if actual != expected or m.get("model_revision") != MODEL_SHA: return False,"exact provenance binding"
        if (tuple(q.shape),tuple(k.shape),tuple(v.shape)) != ((1,16,4096,128),(1,8,4096,128),(1,8,4096,128)) or any(t.dtype != torch.float16 for t in (q,k,v)): return False,"exact FP16 shapes/dtypes"
        return True,"valid"
    except Exception as e: return False,str(e)

def validate_shard(path: Path,s: str,l: int) -> tuple[bool,str]:
    try:
        d=json.loads(path.read_text()); load_manifest()
        if (d.get("manifest_sha256"),d.get("model_sha"),d.get("v2_config_sha256"),d.get("v3_config_sha256"),d.get("sequence"),d.get("layer"),d.get("context_length")) != (TEST_MANIFEST_SHA,MODEL_SHA,V2_SHA,V3_SHA,s,l,4096): return False,"exact shard provenance"
        expected={(p,h,m) for p in POSITIONS for h in range(16) for m in METHODS}; rows=d.get("rows",[]); got={(r.get("position"),r.get("q_head"),r.get("method")) for r in rows}
        if len(rows) != 3*16*7 or got != expected: return False,"exact 3*16*7 rows"
        if any(r.get("kv_head") != r.get("q_head",-1)//2 or not isinstance(r.get("metrics"),dict) or not isinstance(r.get("traffic"),dict) or any(k not in r["metrics"] for k in METRIC_FIELDS) or any(k not in r["traffic"] for k in TRAFFIC_FIELDS) for r in rows): return False,"row mapping or metrics/traffic"
        return True,"valid"
    except Exception as e: return False,str(e)

def status(cache: Path, shards: Path) -> dict[str, Any]:
    load_manifest(); pairs=[]; nc=ns=0
    for s in SPECS:
        for l in LAYERS:
            cp,sp=cache_path(cache,s,l),shard_path(shards,s,l); c=validate_cache(cp,s,l) if cp.exists() else (False,"absent"); z=validate_shard(sp,s,l) if sp.exists() else (False,"absent"); nc+=c[0]; ns+=z[0]; pairs.append({"sequence":s,"layer":l,"cache":{"status":"valid" if c[0] else c[1]},"shard":{"status":"valid" if z[0] else z[1]}})
    return {"manifest_sha256":TEST_MANIFEST_SHA,"valid_caches":nc,"total_caches":15,"valid_shards":ns,"total_shards":15,"pairs":pairs}

def classify(pooled: dict[str,Any]) -> str:
    v,u,v2=(pooled[x] for x in ("frozen_cascadekv_v3","uniform_v1_10","frozen_cascadekv_v2"))
    if v["metrics"]["cosine_similarity"]["mean"] < .985 or v["metrics"]["relative_l2_error"]["mean"] > .12: return "TARGET-MODEL-OUTPUT-GATE-FAILED"
    if v["traffic"]["total_kv_bytes"]["mean"] > u["traffic"]["total_kv_bytes"]["mean"] or v["metrics"]["relative_l2_error"]["mean"] >= v2["metrics"]["relative_l2_error"]["mean"]: return "TARGET-MODEL-COMPARATIVE-GATE-FAILED"
    return "TARGET-MODEL-TRANSFER-PASSED"

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); p.add_argument("--status",action="store_true"); p.add_argument("--cache-dir",type=Path,default=Path("results/cascadekv_v3_qwen3_1p7b_4k_test_cache")); p.add_argument("--shard-dir",type=Path,default=Path("results/cascadekv_v3_qwen3_1p7b_4k_test_shards")); a=p.parse_args()
    if a.preflight: preflight()
    elif a.status: print(json.dumps(status(a.cache_dir,a.shard_dir),indent=2))
    else: p.error("pre-capture harness supports only --preflight or --status")
if __name__ == "__main__": main()
