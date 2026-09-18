#!/usr/bin/env python3
"""Untouched 4K confirmatory test for the immutable CascadeKV-v3 T0 table.

Source construction is deliberately identity/length-only.  No operational
phase selects a source or changes the frozen v3 schedule.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import cascadekv_v2_4k_test as base

# Keep the original implementation, rather than looking it up through the
# mutable base module attribute.  ``bound_base`` intentionally rebinds that
# attribute for evaluation, so using ``base.validate_cache`` from the wrapper
# would dispatch back into the wrapper's adapter.
BASE_VALIDATE_CACHE = base.validate_cache

MODEL_SHA = "c1899de289a04d12100db370d81485cdf75e47ca"
V1_SHA = "2964e4295719e696587f9853177c38399b64edc753e0630fc0d44208c719559b"
V2_SHA = "ef36c5bd1b4211083ac5680e554877281f0a09e37f3dea9e4eb877afba18259c"
V2_TEST_SHA = "31690daff3d60f4be0bed73139818c0adb3b71bed4e04ede77b20accbe3b140c"
V3_SHA = "5d364bc4e351061c243a9166c0a9ccdd68c7cefd74265e4135d937b323e6e505"
V3_RESULT_SHA = "1a5b0f78816b3748a832bc0cf9b6dc791795cfc3ee8c6050bc7130fad186268e"
V3_MANIFEST_SHA = "0bf48150465b3c6cc0e31c35181bda3e337397b2a9d327d0fc1c5de25b16de8a"
TAG = "cascadekv-v3-t0-dev-freeze"
TAG_COMMIT = "ada0613080839fb8cbdea7db4bc47ab30122ece2"
# Filled only after the source-only manifest bytes are frozen.
TEST_MANIFEST_SHA = "2181ca5b7b501926479901b4b56098481adbb2fe790183919e926e54c4e629aa"
MANIFEST = Path("results/cascadekv_v3_4k_test_manifest.json")
V3_CONFIG = Path("configs/cascadekv_v3.json")
V2_CONFIG = Path("configs/cascadekv_v2.json")
V1_CONFIG = Path("configs/cascadekv_v1.json")
V3_RESULT = Path("results/cascadekv_v3_vaware_dev_frozen.json")
V3_MANIFEST = Path("results/cascadekv_v3_vaware_dev_manifest.json")
V2_TEST = Path("results/cascadekv_v2_4k_test.json")
POSITIONS = (2047, 3071, 4095)
LAYERS = (0, 7, 14, 21, 27)
METHODS = ("dense_exact", "flat_q8k4_5", "flat_q8k4_10", "uniform_v1_5", "uniform_v1_10", "frozen_cascadekv_v2", "frozen_cascadekv_v3")
BASE_METHODS = ("dense_exact", "flat_q8k4_5", "flat_q8k4_10", "uniform_v1_5", "uniform_v1_10", "cascadekv_v2")
METRIC_FIELDS = ("relative_exact_attention_mass", "top8_recall", "cosine_similarity", "relative_l2_error", "absolute_l2_error")
TRAFFIC_FIELDS = ("routing_index_bytes", "candidate_sketch_bytes", "authoritative_k_bytes", "selected_v_fp16_bytes", "total_k_bytes", "total_kv_bytes", "total_k_vs_dense_fp16_k", "total_kv_vs_dense", "total_k_vs_flat_q8k4_5", "total_kv_vs_flat_q8k4_5", "total_k_vs_flat_q8k4_10", "total_kv_vs_flat_q8k4_10", "total_k_vs_uniform_v1_10", "total_kv_vs_uniform_v1_10")
SPECS = base.SPECS
CLASSIFICATION = {"absolute": {"pooled_mean_cosine_gte": .985, "pooled_mean_relative_l2_lte": .12}, "comparative": {"v3_total_kv_bytes_mean_lte_uniform_v1_10": True, "v3_relative_l2_mean_lt_frozen_cascadekv_v2": True}, "outcomes": {"pass": "TEST-PASSED", "absolute_fail": "TEST-OUTPUT-GATE-FAILED", "comparative_fail": "TEST-COMPARATIVE-GATE-FAILED", "invalid": "TEST-INVALID"}, "diagnostics": "report-only and never classification inputs"}

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def require_inputs() -> None:
    expected = ((V1_CONFIG,V1_SHA),(V2_CONFIG,V2_SHA),(V3_CONFIG,V3_SHA),(V3_RESULT,V3_RESULT_SHA),(V3_MANIFEST,V3_MANIFEST_SHA),(V2_TEST,V2_TEST_SHA))
    for p,d in expected:
        if not p.is_file() or sha(p) != d: raise RuntimeError(f"STOP immutable input hash differs: {p}")
    if subprocess.check_output(["git","rev-parse",f"{TAG}^{{commit}}"], text=True).strip() != TAG_COMMIT: raise RuntimeError("STOP required freeze tag is missing or differs")
    cfg=json.loads(V3_CONFIG.read_text())
    if cfg.get("target_name") != "T0" or cfg.get("schedule_family") != "LK_VAWARE" or len(cfg.get("exact_layer_head_action_table",{})) != 40: raise RuntimeError("STOP frozen v3 T0 table differs")

def explicit_inventory() -> list[dict[str,Any]]:
    """All recursive, explicitly declared identities in prior JSON manifests only."""
    found=[]
    for path in sorted(Path("results").glob("*manifest*.json")):
        if path == MANIFEST: continue
        try: doc=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError): continue
        def visit(x):
            if isinstance(x,dict):
                index=x.get("index",x.get("example_index")); stable=x.get("stable_example_id",x.get("example_identifier"))
                if {"dataset","config","split"} <= set(x) and isinstance(index,int): found.append({"manifest":str(path),"dataset":x["dataset"],"config":x["config"],"split":x["split"],"index":index,"stable_example_id":stable})
                for y in x.values(): visit(y)
            elif isinstance(x,list):
                for y in x: visit(y)
        visit(doc)
    return sorted({json.dumps(x,sort_keys=True):x for x in found}.values(),key=lambda x:(x["manifest"],x["dataset"],str(x["config"]),x["split"],x["index"],str(x["stable_example_id"])))
def overlaps(a,b): return all(a[k]==b[k] for k in ("dataset","config","split","index")) and (a.get("stable_example_id") is None or b.get("stable_example_id") is None or a["stable_example_id"]==b["stable_example_id"])
def build_manifest() -> dict[str,Any]:
    from transformers import AutoTokenizer
    require_inputs(); prior=explicit_inventory(); tok=AutoTokenizer.from_pretrained(base.MODEL,revision=MODEL_SHA); sources={}
    for family,spec in SPECS.items():
        dataset,_=base.streaming_dataset(spec,spec["revision"]); chosen=None
        for i,row in enumerate(dataset):
            text=row.get(spec["text_field"]); candidate={"dataset":spec["dataset"],"config":spec["config"],"split":spec["split"],"index":i,"stable_example_id":row.get("id",i)}
            if any(overlaps(candidate,p) for p in prior) or not isinstance(text,str): continue
            ok,n=base.token_length_at_least(tok,text,4096)
            if ok:
                chosen={**candidate,"exact_revision":spec["revision"],"text_field":spec["text_field"],"character_count":len(text),"token_length_proof":{"minimum":4096,"observed_capped":n,"at_least":True},"prior_manifest_disjointness":{"identity_not_present":True,"prior_explicit_identity_count":len(prior)}}; break
        if chosen is None: raise RuntimeError(f"STOP no untouched >=4096 identity: {family}")
        sources[family]=chosen
    return {"schema_version":1,"purpose":"untouched confirmatory test of frozen CascadeKV-v3 T0","model":{"name":base.MODEL,"resolved_commit_sha":MODEL_SHA},"context_length":4096,"frozen_inputs":{"v1_config_sha256":V1_SHA,"v2_config_sha256":V2_SHA,"v3_config_sha256":V3_SHA,"v3_development_result_sha256":V3_RESULT_SHA,"v3_development_manifest_sha256":V3_MANIFEST_SHA,"consumed_v2_test_result_sha256":V2_TEST_SHA,"consumed_v2_test_provenance_only":True},"git_freeze_tag":{"name":TAG,"resolved_pre_test_commit_sha":TAG_COMMIT},"complete_prior_explicit_identity_inventory":prior,"selection_rule":"deterministic dataset index order; first explicit identity absent from every prior explicit manifest identity with text str and bounded frozen-tokenizer proof >=4096; selection inspected only explicit identity/index, text existence/type, character count, and bounded tokenizer length; no model/routing/output data entered selection","sources":sources,"methods":list(METHODS),"metrics":["relative_exact_attention_mass","top8_recall","cosine_similarity","relative_l2_error","absolute_l2_error","total_k_bytes","selected_v_fp16_bytes","total_kv_bytes","total_k_vs_dense_fp16_k","total_kv_vs_dense"],"classification":CLASSIFICATION}
def load_manifest():
    require_inputs()
    if not MANIFEST.is_file(): raise RuntimeError("STOP test manifest missing")
    if sha(MANIFEST) != TEST_MANIFEST_SHA: raise RuntimeError("STOP frozen test manifest hash differs before parse/use")
    d=json.loads(MANIFEST.read_text())
    if d.get("context_length") != 4096 or tuple(d.get("methods",())) != METHODS or set(d.get("sources",())) != set(SPECS): raise RuntimeError("STOP frozen test manifest schema differs")
    prior=d["complete_prior_explicit_identity_inventory"]
    for s,x in d["sources"].items():
        if any(overlaps(x,p) for p in prior) or x.get("exact_revision") != SPECS[s]["revision"] or not x.get("token_length_proof",{}).get("at_least"): raise RuntimeError(f"STOP consumed/invalid source: {s}")
    return d
def preflight():
    if MANIFEST.exists(): return load_manifest() # Existing identities are never reselected.
    d=build_manifest(); MANIFEST.write_text(json.dumps(d,indent=2)+"\n")
    if sha(MANIFEST) != TEST_MANIFEST_SHA: raise RuntimeError("STOP deterministic manifest reconstruction did not reproduce frozen SHA")
    return load_manifest()

@contextmanager
def bound_base(config_path=V3_CONFIG, config_sha=V3_SHA, methods=BASE_METHODS, **overrides):
    """Temporarily bind frozen-v2 mechanics without leaking process globals."""
    bindings={"MANIFEST":MANIFEST,"V2_CONFIG":config_path,"V2_SHA":config_sha,
              "TEST_MANIFEST_SHA":TEST_MANIFEST_SHA,"METHODS":methods,
              "require_inputs":require_inputs,"load_manifest":load_manifest,
              "POSITIONS":POSITIONS, **overrides}
    saved={name:getattr(base,name) for name in bindings}
    try:
        for name,value in bindings.items(): setattr(base,name,value)
        yield base
    finally:
        for name,value in saved.items(): setattr(base,name,value)

def validate_cache(p,s,l):
    # The base validator is retained as an additional frozen-v2 mechanics check.
    # It must run with v2's provenance binding even when this wrapper was
    # entered through a v3 evaluation pass.  Call the import-time reference:
    # ``bound.validate_cache`` can be the evaluation adapter below.
    with bound_base(V2_CONFIG,V2_SHA,BASE_METHODS):
        ok,msg=BASE_VALIDATE_CACHE(p,s,l)
    if not ok:return ok,msg
    import torch
    x=torch.load(p,map_location="cpu",weights_only=False); m=x["metadata"]
    q,k,v=x["query"],x["key"],x["value"]
    return (m.get("manifest_sha256")==TEST_MANIFEST_SHA and m.get("model_sha")==MODEL_SHA and m.get("model_revision")==MODEL_SHA and m.get("v3_config_sha256")==V3_SHA and m.get("sequence")==s and m.get("layer")==l and m.get("context_length")==4096 and m.get("source_identity")==load_manifest()["sources"][s] and tuple(q.shape)==(1,16,4096,128) and tuple(k.shape)==(1,8,4096,128) and tuple(v.shape)==(1,8,4096,128) and q.dtype==torch.float16 and k.dtype==torch.float16 and v.dtype==torch.float16,"exact v3 cache binding")
def base_evaluate_cache_adapter(path,sequence,layer):
    """Validator supplied to base.evaluate; deliberately non-mutable dispatch."""
    return validate_cache(path,sequence,layer)
def validate_shard(p,s,l):
    try:
        d=json.loads(p.read_text()); expected={(z,h,m) for z in POSITIONS for h in range(16) for m in METHODS}; got=[(r.get("position"),r.get("q_head"),r.get("method")) for r in d["rows"]]
        # load_manifest makes direct validator calls fail closed on all immutable inputs.
        load_manifest()
        if (d.get("manifest_sha256"),d.get("model_sha"),d.get("sequence"),d.get("layer"),d.get("context_length"),d.get("v2_config_sha256"),d.get("v3_config_sha256")) != (TEST_MANIFEST_SHA,MODEL_SHA,s,l,4096,V2_SHA,V3_SHA): return False,"metadata binding"
        rows=d["rows"]
        if len(got)!=len(expected) or len(set(got))!=len(got) or set(got)!=expected: return False,"exact 3*16*7 cardinality"
        for r in rows:
            if r.get("position") not in POSITIONS or not isinstance(r.get("q_head"),int) or not 0<=r["q_head"]<16 or r.get("kv_head")!=r["q_head"]//2 or r.get("method") not in METHODS or (r.get("sequence"),r.get("layer"),r.get("context_length"))!=(s,l,4096): return False,"row provenance"
            if not isinstance(r.get("metrics"),dict) or not isinstance(r.get("traffic"),dict) or any(k not in r["metrics"] for k in METRIC_FIELDS) or any(k not in r["traffic"] for k in TRAFFIC_FIELDS): return False,"row metric/traffic completeness"
        return True,"exact 3*16*7 cardinality"
    except Exception as e:return False,str(e)
def cache_path(root,s,l): return root/f"{s}_L4096_layer{l}_qkv.pt"
def shard_path(root,s,l): return root/f"{s}_L4096_layer{l}.json"
def status(cache,shards):
    load_manifest(); pairs=[]; vc=vs=0
    for s in SPECS:
      for l in LAYERS:
        cp,sp=cache_path(cache,s,l),shard_path(shards,s,l); co,cm=validate_cache(cp,s,l) if cp.exists() else (False,"absent"); so,sm=validate_shard(sp,s,l) if sp.exists() else (False,"absent"); vc+=co;vs+=so;pairs.append({"sequence":s,"layer":l,"cache":{"status":"valid" if co else cm},"shard":{"status":"valid" if so else sm}})
    return {"manifest_sha256":TEST_MANIFEST_SHA,"valid_caches":vc,"total_caches":15,"valid_shards":vs,"total_shards":15,"pairs":pairs}
def classify(pooled):
    v,u,v2=pooled["frozen_cascadekv_v3"],pooled["uniform_v1_10"],pooled["frozen_cascadekv_v2"]
    if v["metrics"]["cosine_similarity"]["mean"] < .985 or v["metrics"]["relative_l2_error"]["mean"] > .12:return "TEST-OUTPUT-GATE-FAILED"
    if v["traffic"]["total_kv_bytes"]["mean"] > u["traffic"]["total_kv_bytes"]["mean"] or v["metrics"]["relative_l2_error"]["mean"] >= v2["metrics"]["relative_l2_error"]["mean"]: return "TEST-COMPARATIVE-GATE-FAILED"
    return "TEST-PASSED"
def capture(s,l,root):
    load_manifest()
    # Q/K/V capture is schedule-independent.  The preserved capture routine
    # records its mutable V2 provenance fields, so bind it to the immutable v2
    # reference provenance that the preserved validator will subsequently use.
    with bound_base(V2_CONFIG,V2_SHA,BASE_METHODS) as bound:
        bound.capture(s,l,root)
    # Record the independent frozen-v3 test binding without claiming that the
    # tensors were generated by a schedule.  Replace atomically before the
    # final, full wrapper validation.
    import torch
    p=cache_path(root,s,l); x=torch.load(p,map_location="cpu",weights_only=False)
    x["metadata"]["v3_config_sha256"]=V3_SHA; x["metadata"]["model_revision"]=MODEL_SHA
    fd,tmp=tempfile.mkstemp(prefix=f".{p.name}.",suffix=".tmp",dir=p.parent)
    os.close(fd)
    try:
        torch.save(x,tmp); os.replace(tmp,p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    ok,msg=validate_cache(p,s,l)
    if not ok: raise RuntimeError(f"STOP captured cache failed validation: {msg}")
def evaluate(s,l,cache,shards):
    """Evaluate fixed v2 and v3 tables separately; no schedule construction exists here."""
    p=shard_path(shards,s,l)
    if p.exists() and validate_shard(p,s,l)[0]: return
    ok,msg=validate_cache(cache_path(cache,s,l),s,l)
    if not ok: raise RuntimeError(f"STOP invalid cache: {msg}")
    # Base evaluation must accept the v3-provenanced cache while each pass is
    # otherwise bound to its respective frozen config.
    with bound_base(V2_CONFIG,V2_SHA,BASE_METHODS,validate_cache=base_evaluate_cache_adapter) as bound:
        bound.evaluate(s,l,cache,shards)
    baseline=json.loads(p.read_text())
    # Independently apply only the frozen v3 table, then retain its one method.
    tmp=shards/".v3_only"; tmp.mkdir(parents=True,exist_ok=True)
    with bound_base(V3_CONFIG,V3_SHA,BASE_METHODS,validate_cache=base_evaluate_cache_adapter) as bound:
        bound.evaluate(s,l,cache,tmp)
    v3=json.loads((tmp/p.name).read_text())
    baseline_rows=[{**r,"method":"frozen_cascadekv_v2"} if r["method"]=="cascadekv_v2" else r for r in baseline["rows"]]
    extra=[{**r,"method":"frozen_cascadekv_v3"} for r in v3["rows"] if r["method"]=="cascadekv_v2"]
    baseline["rows"]=baseline_rows+extra; baseline["manifest_sha256"]=TEST_MANIFEST_SHA; baseline["model_sha"]=MODEL_SHA; baseline["v2_config_sha256"]=V2_SHA; baseline["v3_config_sha256"]=V3_SHA
    p.write_text(json.dumps(baseline,indent=2)+"\n")
    ok,msg=validate_shard(p,s,l)
    if not ok: raise RuntimeError(f"STOP written shard failed validation: {msg}")
def merge(shards,out):
    load_manifest(); rows=[]; bad=[]
    for s in SPECS:
      for l in LAYERS:
        ok,msg=validate_shard(shard_path(shards,s,l),s,l)
        if ok: rows.extend(json.loads(shard_path(shards,s,l).read_text())["rows"])
        else: bad.append({"sequence":s,"layer":l,"reason":msg})
    if bad:
        out.write_text(json.dumps({"status":"partial","classification":"TEST-INVALID","invalid_shards":bad,"test_manifest_sha256":TEST_MANIFEST_SHA},indent=2)+"\n"); return
    pooled={m:base._summary([r for r in rows if r["method"]==m]) for m in METHODS}
    def groups(field): return {str(k):base._summary([r for r in rows if r["method"]=="frozen_cascadekv_v3" and r[field]==k]) for k in sorted({r[field] for r in rows})}
    def delta(other):
        a,b=pooled["frozen_cascadekv_v3"],pooled[other]
        return {"cosine_mean":a["metrics"]["cosine_similarity"]["mean"]-b["metrics"]["cosine_similarity"]["mean"],"relative_l2_mean":a["metrics"]["relative_l2_error"]["mean"]-b["metrics"]["relative_l2_error"]["mean"],"total_kv_bytes_mean":a["traffic"]["total_kv_bytes"]["mean"]-b["traffic"]["total_kv_bytes"]["mean"],"total_kv_bytes_ratio":a["traffic"]["total_kv_bytes"]["mean"]/b["traffic"]["total_kv_bytes"]["mean"]}
    payload={"experiment":"cascadekv_v3_t0_untouched_4k_confirmatory_test","status":"complete","classification":classify(pooled),"predeclared_classification":CLASSIFICATION,"test_manifest_sha256":TEST_MANIFEST_SHA,"frozen_input_hashes":{"v1":V1_SHA,"v2":V2_SHA,"v3":V3_SHA,"v3_result":V3_RESULT_SHA,"v3_manifest":V3_MANIFEST_SHA,"v2_test_provenance":V2_TEST_SHA},"methods":list(METHODS),"pooled":pooled,"frozen_v3_diagnostics":{"per_source":groups("sequence"),"per_layer":groups("layer"),"per_position":groups("position"),"per_q_head":groups("q_head"),"per_kv_head":groups("kv_head")},"direct_deltas":{"v3_minus_v2":delta("frozen_cascadekv_v2"),"v3_minus_uniform10":delta("uniform_v1_10"),"v3_minus_flat5":delta("flat_q8k4_5")}}
    out.write_text(json.dumps(payload,indent=2)+"\n")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true");p.add_argument("--status",action="store_true");p.add_argument("--capture",action="store_true");p.add_argument("--evaluate",action="store_true");p.add_argument("--merge",action="store_true");p.add_argument("--sequence",choices=SPECS);p.add_argument("--layer",type=int,choices=LAYERS);p.add_argument("--cache-dir",type=Path,default=Path("results/cascadekv_v3_4k_test_cache"));p.add_argument("--shard-dir",type=Path,default=Path("results/cascadekv_v3_4k_test_shards"));p.add_argument("--output",type=Path,default=Path("results/cascadekv_v3_4k_test.json")); a=p.parse_args()
    if a.preflight: preflight()
    elif a.status: print(json.dumps(status(a.cache_dir,a.shard_dir),indent=2))
    elif a.capture: capture(a.sequence,a.layer,a.cache_dir)
    elif a.evaluate: evaluate(a.sequence,a.layer,a.cache_dir,a.shard_dir)
    elif a.merge: merge(a.shard_dir,a.output)
    else:p.error("select phase")
if __name__ == "__main__": main()
