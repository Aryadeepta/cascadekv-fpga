#!/usr/bin/env python3
"""Frozen, data-free protocol harness for Qwen3-1.7B V-aware development.

This module deliberately has no capture, model-loading, or action-evaluation
entry point.  A later separately authorized execution implementation must emit
the cache and action-shard formats validated here; preflight/status are safe
read-only protocol operations.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from itertools import product
from pathlib import Path

MODEL="Qwen/Qwen3-1.7B"; MODEL_SHA="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CONFIG_SHA="1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
V3=Path("configs/cascadekv_v3.json"); V3_SHA="5d364bc4e351061c243a9166c0a9ccdd68c7cefd74265e4135d937b323e6e505"
V2=Path("configs/cascadekv_v2.json"); V2_SHA="ef36c5bd1b4211083ac5680e554877281f0a09e37f3dea9e4eb877afba18259c"
ARCHIVED=Path("results/cascadekv_v3_qwen3_1p7b_4k_test.json"); ARCHIVED_SHA="776ebaca657e8913215565b882bc55b491fe0ae390fc947a1320275e253604da"
ARCHIVED_MANIFEST=Path("results/cascadekv_v3_qwen3_1p7b_4k_test_manifest.json"); ARCHIVED_MANIFEST_SHA="20927a2a34bb5b03faf622b0c44195e50f7a482942a3f08da0cc80ecaa568574"
MANIFEST=Path("results/cascadekv_v3_qwen3_1p7b_vaware_dev_manifest.json")
MANIFEST_SHA="46341cc2c9de946879ef38ab3b7e28f50de19602b814e324f92a5aff27ccf637"
LAYERS=(0,7,14,21,27); POSITIONS=(2047,3071,4095); ACTIONS=("A0","A1","A2","A3","A4")
CAL=("narrative_calibration","report_calibration","qa_calibration")
VAL=("narrative_validation","report_validation","qa_validation")
METRICS=("relative_exact_attention_mass","top8_recall","cosine_similarity","relative_l2_error","absolute_l2_error")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(x): return (x["dataset"],x.get("config"),x["split"],x["index"],x.get("stable_example_id"))
def cache_path(root, source, layer): return Path(root)/f"{source}_L4096_layer{layer}_qkv.pt"
def shard_path(root, source, layer): return Path(root)/f"{source}_L4096_layer{layer}.json"

def require_inputs():
    if sha(V3)!=V3_SHA or sha(V2)!=V2_SHA: raise RuntimeError("STOP immutable routing config hash differs")
    if sha(ARCHIVED)!=ARCHIVED_SHA or sha(ARCHIVED_MANIFEST)!=ARCHIVED_MANIFEST_SHA: raise RuntimeError("STOP archived provenance hash differs")
    # Only status/classification are admissible reads from the consumed result.
    old=json.loads(ARCHIVED.read_text())
    if old.get("status")!="complete" or old.get("classification")!="TARGET-MODEL-OUTPUT-GATE-FAILED": raise RuntimeError("STOP archived result state differs")

def load_manifest():
    require_inputs()
    if not MANIFEST.is_file() or sha(MANIFEST)!=MANIFEST_SHA: raise RuntimeError("STOP frozen development manifest hash differs")
    d=json.loads(MANIFEST.read_text()); sources=d.get("sources",{})
    if d.get("target_model",{}).get("resolved_commit_sha")!=MODEL_SHA or d.get("target_model",{}).get("config_sha256")!=CONFIG_SHA: raise ValueError("target binding mismatch")
    if set(sources)!=set(CAL+VAL) or d.get("context_length")!=4096: raise ValueError("split/context schema mismatch")
    used={identity(x) for x in d["complete_consumed_identity_inventory"]}
    if any(identity(x) in used for x in sources.values()) or len({identity(x) for x in sources.values()})!=6: raise ValueError("source leakage/disjointness violation")
    if any(sources[s]["role"]!="calibration" for s in CAL) or any(sources[s]["role"]!="validation" for s in VAL): raise ValueError("role binding mismatch")
    return d

def validate_cache(path, source, layer):
    """Metadata/shape validator; does not load a model and imports no torch."""
    try:
        import torch
        d=torch.load(path,map_location="cpu",weights_only=False); m=d["metadata"]; src=load_manifest()["sources"][source]
        if (m.get("manifest_sha256"),m.get("model_sha"),m.get("source_identity"),m.get("role"),m.get("layer"),m.get("context_length")) != (MANIFEST_SHA,MODEL_SHA,src,src["role"],layer,4096): return False,"metadata binding"
        for name,shape in (("query",(1,16,4096,128)),("key",(1,8,4096,128)),("value",(1,8,4096,128))):
            if tuple(d[name].shape)!=shape or d[name].dtype!=torch.float16: return False,"authoritative FP16 Q/K/V geometry"
        return True,None
    except Exception as e: return False,str(e)

def validate_shard(path, source, layer):
    try:
        d=json.loads(Path(path).read_text()); src=load_manifest()["sources"][source]; rows=d["rows"]
        if (d.get("manifest_sha256"),d.get("model_sha"),d.get("source_identity"),d.get("role"),d.get("layer"),d.get("context_length")) != (MANIFEST_SHA,MODEL_SHA,src,src["role"],layer,4096): return False,"top-level binding"
        expected={(p,h,a) for p in POSITIONS for h in range(16) for a in ACTIONS}
        got={(r.get("position"),r.get("q_head"),r.get("action")) for r in rows}
        if len(rows)!=240 or got!=expected: return False,"requires exactly 3x16x5 action rows"
        for r in rows:
            if r.get("source")!=source or r.get("role")!=src["role"] or r.get("layer")!=layer or r.get("kv_head")!=r["q_head"]//2: return False,"row binding"
            if not all(isinstance(r.get("metrics",{}).get(k),(int,float)) and math.isfinite(r["metrics"][k]) for k in METRICS): return False,"nonfinite/missing metric"
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in r.get("traffic",{}).values()): return False,"nonfinite traffic"
        return True,None
    except Exception as e: return False,str(e)

def status(cache=Path("results/cascadekv_v3_qwen3_1p7b_vaware_dev_cache"), shards=Path("results/cascadekv_v3_qwen3_1p7b_vaware_dev_shards")):
    load_manifest(); caches=shard_count=0; pairs=[]
    for source,layer in product(CAL+VAL,LAYERS):
        c=cache_path(cache,source,layer); s=shard_path(shards,source,layer)
        okc,msgc=validate_cache(c,source,layer) if c.exists() else (False,"absent")
        oks,msgs=validate_shard(s,source,layer) if s.exists() else (False,"absent")
        caches+=okc; shard_count+=oks; pairs.append({"source":source,"layer":layer,"cache":"valid" if okc else msgc,"action_shard":"valid" if oks else msgs})
    return {"manifest_sha256":MANIFEST_SHA,"valid_caches":caches,"total_caches":30,"valid_action_shards":shard_count,"total_action_shards":30,"pairs":pairs}

def optimize_calibration(action_groups, target_kv):
    """Original exact static DP: integer microbytes, rel-L2, -cosine, traffic, A0..A4."""
    states=[(0,0.,0.,0)]; cap=round(target_kv*len(action_groups)*1000)
    for group in action_groups:
        nxt=[]
        for b,e,c,code in states:
            for i,a in enumerate(ACTIONS):
                x=group[a]; z=(b+round(x["total_kv_bytes"]*1000),e+x["relative_l2"],c+x["cosine"],code*5+i)
                if z[0]<=cap: nxt.append(z)
        by_bytes={}
        for z in sorted(nxt,key=lambda z:(z[0],z[1],-z[2],z[3])): by_bytes.setdefault(z[0],z)
        states=[]; best=None
        for z in by_bytes.values():
            q=(z[1],-z[2])
            if best is None or q<best: states.append(z); best=q
    if not states:return None
    b,e,c,code=min(states,key=lambda z:(z[1],-z[2],z[0],z[3])); table=[]
    for _ in action_groups: code,d=divmod(code,5); table.append(ACTIONS[d])
    return b,e,c,tuple(reversed(table))

def inherited_targets(v2_mean_kv, uniform10_mean_kv):
    return {"T0":v2_mean_kv,"T1":v2_mean_kv+.5*(uniform10_mean_kv-v2_mean_kv),"T2":uniform10_mean_kv}
def passes(candidate, uniform10, v2, zero_shot_v3):
    return candidate["cosine"]>=.985 and candidate["relative_l2"]<=.120 and candidate["kv"]<=uniform10 and candidate["relative_l2"]<v2 and candidate["relative_l2"]<zero_shot_v3
def choose_winner(candidates, uniform10, v2, zero_shot_v3):
    """Return the first passing predeclared target, or None on any invalid set.

    Exactly one named T0, T1, and T2 candidate is required.  Missing,
    duplicate, unknown, malformed, and non-iterable candidate collections fail
    closed as no winner.  Once this schema check succeeds, only the fixed
    target order determines selection; validation values are consulted solely
    by ``passes`` for their already-declared gates.
    """
    order=("target_vaware_T0","target_vaware_T1","target_vaware_T2")
    try:
        by_name={}
        for candidate in candidates:
            name=candidate.get("name")
            if name not in order or name in by_name:
                return None
            by_name[name]=candidate
    except (AttributeError, TypeError):
        return None
    if set(by_name)!=set(order):
        return None
    for name in order:
        try:
            if passes(by_name[name],uniform10,v2,zero_shot_v3):
                return by_name[name]
        except (KeyError, TypeError):
            return None
    return None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); p.add_argument("--status",action="store_true"); a=p.parse_args()
    if a.preflight: load_manifest(); print("frozen protocol manifest verified")
    elif a.status: print(json.dumps(status(),indent=2))
    else: p.error("only data-free --preflight and --status are enabled in this protocol-preparation harness")
if __name__=="__main__": main()
