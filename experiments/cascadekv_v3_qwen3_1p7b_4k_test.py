#!/usr/bin/env python3
"""Predeclared zero-shot Qwen3-1.7B transfer harness for frozen CascadeKV-v3.

The only source-construction operation is ``--preflight``.  It never loads
model weights.  Capture/evaluation are separate, explicit future phases.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, subprocess, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import cascadekv_v2_4k_test as base

# Keep immutable references: ``bound_base`` deliberately changes attributes on
# ``base`` and looking those functions up dynamically would recurse.
BASE_CAPTURE = base.capture
BASE_EVALUATE = base.evaluate
BASE_VALIDATE_CACHE = base.validate_cache

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
TAG = "cascadekv-v3-qwen3-1p7b-4k-test-freeze"
TAG_COMMIT = "890bdb1cbc3e097e324b480debbcb8ffd7385cd1"
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
    if subprocess.check_output(["git", "rev-parse", f"{TAG}^{{commit}}"], text=True).strip() != TAG_COMMIT: raise RuntimeError("STOP required protocol tag differs")

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
        # Reuse the immutable v2 validator under an explicitly target-bound
        # base module.  Do not consult base.validate_cache dynamically: this
        # wrapper is installed there during evaluation.
        with bound_base(V2_CONFIG,V2_SHA,BASE_METHODS):
            base_ok,base_msg=BASE_VALIDATE_CACHE(path,s,l)
        if not base_ok: return False,f"base validator: {base_msg}"
        x=torch.load(path,map_location="cpu",weights_only=False); m=x["metadata"]; mf=load_manifest(); q,k,v=(x[n] for n in ("query","key","value"))
        expected=(TEST_MANIFEST_SHA,MODEL_SHA,V2_SHA,V3_SHA,s,l,4096,mf["sources"][s])
        actual=(m.get("manifest_sha256"),m.get("model_sha"),m.get("v2_config_sha256"),m.get("v3_config_sha256"),m.get("sequence"),m.get("layer"),m.get("context_length"),m.get("source_identity"))
        if actual != expected or m.get("model_revision") != MODEL_SHA: return False,"exact provenance binding"
        shapes={"q":list(q.shape),"k":list(k.shape),"v":list(v.shape)}
        if m.get("tensor_shapes") != shapes or (tuple(q.shape),tuple(k.shape),tuple(v.shape)) != ((1,16,4096,128),(1,8,4096,128),(1,8,4096,128)) or any(t.dtype != torch.float16 for t in (q,k,v)): return False,"exact FP16 shapes/dtypes"
        return True,"valid"
    except Exception as e: return False,str(e)

def validate_shard(path: Path,s: str,l: int) -> tuple[bool,str]:
    try:
        d=json.loads(path.read_text()); load_manifest()
        if (d.get("manifest_sha256"),d.get("model_sha"),d.get("v2_config_sha256"),d.get("v3_config_sha256"),d.get("sequence"),d.get("layer"),d.get("context_length")) != (TEST_MANIFEST_SHA,MODEL_SHA,V2_SHA,V3_SHA,s,l,4096): return False,"exact shard provenance"
        expected={(p,h,m) for p in POSITIONS for h in range(16) for m in METHODS}; rows=d.get("rows",[]); got={(r.get("position"),r.get("q_head"),r.get("method")) for r in rows}
        if len(rows) != 3*16*7 or got != expected: return False,"exact 3*16*7 rows"
        if any((r.get("sequence"),r.get("layer"),r.get("context_length")) != (s,l,4096) or r.get("kv_head") != r.get("q_head",-1)//2 or not isinstance(r.get("metrics"),dict) or not isinstance(r.get("traffic"),dict) or any(k not in r["metrics"] for k in METRIC_FIELDS) or any(k not in r["traffic"] for k in TRAFFIC_FIELDS) for r in rows): return False,"row mapping or metrics/traffic"
        # Every summary and classification input must be a finite real number.
        # This prevents NaN comparison semantics from accidentally passing a
        # gate and makes malformed traffic fail before merge aggregation.
        for r in rows:
            for values,fields in ((r["metrics"],METRIC_FIELDS),(r["traffic"],TRAFFIC_FIELDS)):
                if any(not isinstance(values[k],(int,float)) or isinstance(values[k],bool) or not math.isfinite(values[k]) for k in fields): return False,"non-finite metric/traffic"
                if any(not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) for value in values.values()): return False,"non-finite metric/traffic"
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

BASE_METHODS = ("dense_exact", "flat_q8k4_5", "flat_q8k4_10", "uniform_v1_5", "uniform_v1_10", "cascadekv_v2")

@contextmanager
def bound_base(config_path: Path = V2_CONFIG, config_sha: str = V2_SHA, methods=BASE_METHODS, **overrides):
    """Bind the real v2 capture/evaluation call graph to 1.7B, then restore it.

    Call graph: wrapper -> base.capture -> AutoTokenizer/AutoModel using
    base.MODEL/base.MODEL_SHA -> post-norm/post-RoPE Q/K/V helper.  The base
    module imports MODEL as a mutable global, hence both name and revision are
    explicitly rebound here (not merely MODEL_SHA).
    """
    bindings={"MANIFEST":MANIFEST,"V2_CONFIG":config_path,"V2_SHA":config_sha,
              "TEST_MANIFEST_SHA":TEST_MANIFEST_SHA,"METHODS":methods,
              "POSITIONS":POSITIONS,"MODEL":MODEL,"MODEL_SHA":MODEL_SHA,
              "require_inputs":require_inputs,"load_manifest":load_manifest,
              **overrides}
    saved={name:getattr(base,name) for name in bindings}
    try:
        for name,value in bindings.items(): setattr(base,name,value)
        yield base
    finally:
        for name,value in saved.items(): setattr(base,name,value)

def base_evaluate_cache_adapter(path: Path, sequence: str, layer: int):
    return validate_cache(path, sequence, layer)

def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent); os.close(fd)
    try: torch.save(payload,tmp); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f: json.dump(payload,f,indent=2); f.write("\n")
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def capture(s: str, l: int, root: Path) -> None:
    load_manifest(); p=cache_path(root,s,l)
    if p.exists() and validate_cache(p,s,l)[0]: return
    # The preserved helper uses post-norm/post-RoPE FP16 Q/K/V semantics.
    # Capture to an isolated temporary directory so its non-atomic base write
    # can never expose a partial target artifact.
    with tempfile.TemporaryDirectory(dir=root.parent if root.parent.exists() else None) as td:
        staging=Path(td)
        with bound_base(V2_CONFIG,V2_SHA,BASE_METHODS) as bound: BASE_CAPTURE(s,l,staging)
        x=torch.load(cache_path(staging,s,l),map_location="cpu",weights_only=False)
        x["metadata"].update({"manifest_sha256":TEST_MANIFEST_SHA,"model_sha":MODEL_SHA,
          "model_revision":MODEL_SHA,"v2_config_sha256":V2_SHA,"v3_config_sha256":V3_SHA})
        _atomic_torch_save(x,p)
    ok,msg=validate_cache(p,s,l)
    if not ok: raise RuntimeError(f"STOP captured cache failed validation: {msg}")

def evaluate(s: str, l: int, cache: Path, shards: Path) -> None:
    p=shard_path(shards,s,l)
    if p.exists() and validate_shard(p,s,l)[0]: return
    ok,msg=validate_cache(cache_path(cache,s,l),s,l)
    if not ok: raise RuntimeError(f"STOP invalid cache: {msg}")
    with tempfile.TemporaryDirectory(dir=shards.parent if shards.parent.exists() else None) as td:
        stage=Path(td); v3stage=stage/"v3"; v3stage.mkdir()
        with bound_base(V2_CONFIG,V2_SHA,BASE_METHODS,validate_cache=base_evaluate_cache_adapter) as bound: BASE_EVALUATE(s,l,cache,stage)
        baseline=json.loads(shard_path(stage,s,l).read_text())
        with bound_base(V3_CONFIG,V3_SHA,BASE_METHODS,validate_cache=base_evaluate_cache_adapter) as bound: BASE_EVALUATE(s,l,cache,v3stage)
        v3=json.loads(shard_path(v3stage,s,l).read_text())
        baseline["rows"]=[{**r,"method":"frozen_cascadekv_v2"} if r["method"] == "cascadekv_v2" else r for r in baseline["rows"]] + [{**r,"method":"frozen_cascadekv_v3"} for r in v3["rows"] if r["method"] == "cascadekv_v2"]
        baseline.update({"manifest_sha256":TEST_MANIFEST_SHA,"model_sha":MODEL_SHA,"v2_config_sha256":V2_SHA,"v3_config_sha256":V3_SHA,"sequence":s,"layer":l,"context_length":4096})
        _atomic_json_save(baseline,p)
    ok,msg=validate_shard(p,s,l)
    if not ok: raise RuntimeError(f"STOP written shard failed validation: {msg}")

def merge(shards: Path, out: Path) -> None:
    load_manifest(); rows=[]; bad=[]
    for s in SPECS:
        for l in LAYERS:
            ok,msg=validate_shard(shard_path(shards,s,l),s,l)
            if ok: rows.extend(json.loads(shard_path(shards,s,l).read_text())["rows"])
            else: bad.append({"sequence":s,"layer":l,"reason":msg})
    if bad:
        _atomic_json_save({"status":"partial","classification":"TARGET-MODEL-TEST-INVALID","invalid_shards":bad,"manifest_sha256":TEST_MANIFEST_SHA},out); return
    pooled={m:base._summary([r for r in rows if r["method"]==m]) for m in METHODS}
    group=lambda field:{str(k):base._summary([r for r in rows if r["method"]=="frozen_cascadekv_v3" and r[field]==k]) for k in sorted({r[field] for r in rows})}
    def delta(other):
        a,b=pooled["frozen_cascadekv_v3"],pooled[other]
        return {"cosine_mean":a["metrics"]["cosine_similarity"]["mean"]-b["metrics"]["cosine_similarity"]["mean"],"relative_l2_mean":a["metrics"]["relative_l2_error"]["mean"]-b["metrics"]["relative_l2_error"]["mean"],"total_kv_bytes_mean":a["traffic"]["total_kv_bytes"]["mean"]-b["traffic"]["total_kv_bytes"]["mean"]}
    parent=json.loads(V3_TEST_RESULT.read_text())["pooled"]["frozen_cascadekv_v3"]
    a=pooled["frozen_cascadekv_v3"]
    parent_delta={"cosine_mean":a["metrics"]["cosine_similarity"]["mean"]-parent["metrics"]["cosine_similarity"]["mean"],"relative_l2_mean":a["metrics"]["relative_l2_error"]["mean"]-parent["metrics"]["relative_l2_error"]["mean"],"total_kv_bytes_mean":a["traffic"]["total_kv_bytes"]["mean"]-parent["traffic"]["total_kv_bytes"]["mean"],"classification_input":False}
    payload={"experiment":"cascadekv_v3_qwen3_1p7b_4k_test","status":"complete","classification":classify(pooled),"predeclared_classification":CLASSIFICATION,"target_model":{"name":MODEL,"revision":MODEL_SHA,"config_sha256":"1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"},"manifest_sha256":TEST_MANIFEST_SHA,"protocol_tag":{"name":TAG,"commit":TAG_COMMIT},"parent_v3_config_sha256":V3_SHA,"parent_0p6b_result_sha256":V3_TEST_SHA,"v2_config_sha256":V2_SHA,"methods":list(METHODS),"pooled":pooled,"diagnostics":{"per_source":group("sequence"),"per_layer":group("layer"),"per_position":group("position"),"per_q_head":group("q_head"),"per_kv_head":group("kv_head"),"v3_minus_completed_0p6b_v3":parent_delta},"direct_deltas":{"v3_minus_v2":delta("frozen_cascadekv_v2"),"v3_minus_uniform10":delta("uniform_v1_10"),"v3_minus_flat5":delta("flat_q8k4_5")}}
    _atomic_json_save(payload,out)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); p.add_argument("--status",action="store_true"); p.add_argument("--capture",action="store_true"); p.add_argument("--evaluate",action="store_true"); p.add_argument("--merge",action="store_true"); p.add_argument("--sequence",choices=SPECS); p.add_argument("--layer",type=int,choices=LAYERS); p.add_argument("--cache-dir",type=Path,default=Path("results/cascadekv_v3_qwen3_1p7b_4k_test_cache")); p.add_argument("--shard-dir",type=Path,default=Path("results/cascadekv_v3_qwen3_1p7b_4k_test_shards")); p.add_argument("--output",type=Path,default=Path("results/cascadekv_v3_qwen3_1p7b_4k_test.json")); a=p.parse_args()
    if a.preflight: preflight()
    elif a.status: print(json.dumps(status(a.cache_dir,a.shard_dir),indent=2))
    elif a.capture:
        if a.sequence is None or a.layer is None: p.error("--capture requires --sequence and --layer")
        capture(a.sequence,a.layer,a.cache_dir)
    elif a.evaluate:
        if a.sequence is None or a.layer is None: p.error("--evaluate requires --sequence and --layer")
        evaluate(a.sequence,a.layer,a.cache_dir,a.shard_dir)
    elif a.merge: merge(a.shard_dir,a.output)
    else: p.error("select phase")
if __name__ == "__main__": main()
