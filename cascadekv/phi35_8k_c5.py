"""C5's narrow A7--A9 expansion and development/holdout firewall.

This module is intentionally data-free on import and during preflight.  The
executor is the only component allowed to call C5 capture/selection commands.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from . import phi35_8k_c4 as c4

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cascadekv_phi35_8k_c5_protocol.json"
RUNTIME = ROOT / "configs/cascadekv_phi35_8k_c5_runtime_manifest.json"
C5_TAG = "cascadekv-phi35-8k-c5-freeze-v1"
C4_PARENT = {"tag": "cascadekv-phi35-8k-c4-result-recovered-v1", "commit": "98beb485c2ac7d9e209b5f8f1dfe2fdb7e4bda75", "recovery_manifest_sha256": "d6715a952313873988236f3dd497d7eb1932725398415f01aaf8b96c12bf6d14", "postmortem_json_sha256": "89aa6fa3a45831df8e7f8a09c816e74cb78086d2eea92c6a535e88ea19804419"}
ACTIONS = tuple(f"A{x}" for x in range(10)); TARGETS = c4.TARGETS
DEVELOPMENT = {"narrative": (13, 14, 15, 16), "report": (16, 17, 18, 19), "qa": (13, 15, 16, 17)}
FRONTIER = {"narrative": 17, "qa": 18, "report": 20}


class C5Error(RuntimeError): pass


def canonical_json(value: Any) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""): h.update(part)
    return h.hexdigest()
def _json(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc: raise C5Error(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict): raise C5Error("JSON root must be object")
    return value
def _git(ref: str) -> str:
    try: return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{ref}^{{commit}}"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc: raise C5Error(f"required git reference unavailable: {ref}") from exc


def action_definitions() -> dict[str, dict[str, Any]]:
    base = c4._protocol_actions()
    return {**base, "A7": {"kind":"hierarchy", "candidate_fraction":.10, "routing_profile":"phi35-8k-depth-transfer-qwen10-v1"}, "A8": {"kind":"hierarchy", "candidate_fraction":.15, "routing_profile":"phi35-8k-depth-transfer-qwen10-v1"}, "A9": {"kind":"hierarchy", "candidate_fraction":.20, "routing_profile":"phi35-8k-depth-transfer-qwen10-v1"}}


def validate_protocol(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = dict(_json(PROTOCOL) if payload is None else payload)
    required = {"schema_version","c4_result_parent","geometry","methods","development_sources","prospective_frontier","targets","quality_gates","optimizer","development_gate","holdout_selection","firewall","scientific_state"}
    if set(p) != required or p.get("schema_version") != "cascadekv-phi35-8k-c5-protocol-v1": raise C5Error("C5 protocol schema differs")
    if p["c4_result_parent"] != C4_PARENT: raise C5Error("C4 recovered-result parent differs")
    c4p = c4.validate_protocol()
    structural = ("context_length", "layers", "query_positions", "heads", "head_dim", "schedule_cells")
    if any(p["geometry"].get(key) != c4p["geometry"].get(key) for key in structural): raise C5Error("C5 structural geometry differs")
    if p["geometry"].get("observations_per_source_per_method") != 480 or p["geometry"].get("development_observations_per_method") != 5760 or p["geometry"].get("holdout_observations_per_method") != 1440: raise C5Error("C5 observation geometry differs")
    methods = p["methods"]
    if methods.get("hierarchy") != c4p["methods"]["hierarchy"] or methods.get("baselines") != c4p["methods"]["baselines"] or methods.get("routing_profiles") != c4p["methods"]["routing_profiles"]: raise C5Error("frozen method/profile mutation")
    if methods.get("actions") != action_definitions() or list(methods["actions"]) != list(ACTIONS): raise C5Error("C5 action order/menu differs")
    expected_dev = {"identities": {family: list(indices) for family, indices in DEVELOPMENT.items()}, "c4_consumed_holdout_now_development": {"narrative":[16],"report":[19],"qa":[17]}, "forbidden":["qa:14"], "source_text_stored":False}
    if p["development_sources"] != expected_dev: raise C5Error("C5 development inventory differs")
    if p["prospective_frontier"] != {"start_indices": FRONTIER, "untouched_during_preparation": True}: raise C5Error("prospective frontier differs")
    if p["targets"] != c4p["targets"] or p["quality_gates"] != c4p["quality_gates"]: raise C5Error("targets or gates differ")
    if p["optimizer"] != {"version":"c5-ten-action-integer-microbyte-dp-v1","scope":"development_only","objective":c4p["optimizer"]["objective"],"action_change_rule":"one of exactly A0..A9 per 160 layer/head cells; no iterative parameter mutation","tie_breaking":"relative-L2 ascending, cosine descending, traffic ascending, lexical action table ascending","ordering":"sampled layer ascending, then head ascending, then A0,A1,A2,A3,A4,A5,A6,A7,A8,A9"}: raise C5Error("optimizer differs")
    if p["development_gate"] != {"definition":"feasible and all four frozen quality gates on development","no_go_classification":"C5-DEVELOPMENT-NO-GO","go_requires_freeze_before_selection":True,"priority_order":list(TARGETS)}: raise C5Error("development firewall differs")
    if p["holdout_selection"] != {"kaggle_only":True,"start_indices":FRONTIER,"rule":"first mechanically eligible globally-new exact input SHA in ascending order","requires_frozen_development_result_and_schedule":True,"persist_raw_text":False}: raise C5Error("holdout contract differs")
    if p["firewall"] != {"test_opens_development_tensors":False,"test_optimizer_rerun":False,"schedule_sha256_unchanged":True,"development_and_holdout_tensor_roots_distinct":True}: raise C5Error("test firewall differs")
    if p["scientific_state"] != {"local_preparation_only":True,"prospective_frontier_accessed":False,"c5_metrics_observed":False}: raise C5Error("scientific state differs")
    return p


def preflight(*, execution: bool = False) -> dict[str, Any]:
    if _git(C4_PARENT["tag"]) != C4_PARENT["commit"]: raise C5Error("C4 parent tag commit differs")
    if sha256_path(ROOT / "results/archive/cascadekv_phi35_8k_c4_v2_stdout_recovery.json") != C4_PARENT["recovery_manifest_sha256"] or sha256_path(ROOT / "results/archive/cascadekv_phi35_8k_c4_v2_postmortem.json") != C4_PARENT["postmortem_json_sha256"]: raise C5Error("C4 parent evidence differs")
    if execution and _git(C5_TAG) != _git("HEAD"): raise C5Error("HEAD is not the required future C5 freeze tag commit")
    validate_protocol(); closure = verify_runtime_closure()
    return {"local_only": not execution, "protocol_sha256":sha256_path(PROTOCOL), "runtime_manifest_sha256":sha256_path(RUNTIME), "bound_files": closure}


def verify_runtime_closure() -> tuple[dict[str, str], ...]:
    manifest = _json(RUNTIME)
    if set(manifest) != {"schema_version", "purpose", "protocol", "bound_files"} or manifest.get("schema_version") != 1:
        raise C5Error("C5 runtime manifest schema differs")
    if manifest["protocol"] != {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256_path(PROTOCOL)}:
        raise C5Error("C5 runtime protocol binding differs")
    rows=[]; seen=set()
    for row in manifest["bound_files"]:
        if not isinstance(row, Mapping) or set(row) != {"path","sha256"} or not isinstance(row["path"],str) or row["path"] in seen:
            raise C5Error("unsafe runtime entry")
        seen.add(row["path"]); path=ROOT/row["path"]
        if not path.is_file() or sha256_path(path) != row["sha256"]: raise C5Error(f"runtime closure mismatch: {row['path']}")
        rows.append({"path":row["path"],"sha256":row["sha256"]})
    return tuple(rows)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise C5Error(f"refusing to overwrite immutable result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(canonical_json(value) + "\n"); temp = Path(handle.name)
    os.replace(temp, path)

def _finite(value: Any) -> bool: return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

def optimize_action_cells(groups: Iterable[Mapping[str, Mapping[str, float]]], cap: int, actions: tuple[str, ...] = ACTIONS) -> tuple[int,float,float,tuple[str,...]] | None:
    """C4's exact Pareto DP, generalized only from radix seven to action count."""
    rows=list(groups); radix=len(actions)
    if not rows or type(cap) is not int or cap < 0: raise C5Error("optimizer requires cells and nonnegative integer cap")
    states: dict[int, tuple[float,float,int]]={0:(0.,0.,0)}
    for row in rows:
        candidate: dict[int, tuple[float,float,int]]={}
        for budget,(error,cosine,code) in states.items():
            for index,action in enumerate(actions):
                stat=row.get(action,{}); traffic,rel,cos=stat.get("total_kv_bytes"),stat.get("relative_l2"),stat.get("cosine")
                if not all(_finite(x) for x in (traffic,rel,cos)): raise C5Error("incomplete action statistics")
                next_budget=budget+round(float(traffic)*1000)
                if next_budget>cap: continue
                item=(error+float(rel),cosine+float(cos),code*radix+index); prior=candidate.get(next_budget)
                if prior is None or (item[0],-item[1],item[2]) < (prior[0],-prior[1],prior[2]): candidate[next_budget]=item
        states={}; best_error,best_cos=math.inf,-math.inf
        for budget in sorted(candidate):
            error,cosine,code=candidate[budget]
            if error<best_error or (error==best_error and cosine>best_cos):
                states[budget]=(error,cosine,code)
                if error<best_error: best_error,best_cos=error,cosine
                elif cosine>best_cos: best_cos=cosine
    if not states: return None
    budget,(error,cosine,code)=min(states.items(), key=lambda item:(item[1][0],-item[1][1],item[0],item[1][2]))
    table=[]
    for _ in rows: code,digit=divmod(code,radix); table.append(actions[digit])
    return budget,error,cosine,tuple(reversed(table))

def _target_cap(protocol: Mapping[str, Any], target: str, dense: float, flat5: float) -> tuple[float,bool,int]:
    target_mean,strict=(flat5,True) if target=="T5" else (float(protocol["targets"]["fractions"][target])*dense,False)
    return target_mean,strict,math.ceil(target_mean*160*1000)-(1 if strict else 0)

def optimize_static_schedule(groups: Iterable[Mapping[str, Mapping[str, float]]], target_mean_kv: float, *, strict: bool=False) -> tuple[int,float,float,tuple[str,...]] | None:
    rows=list(groups)
    if len(rows)!=160 or not math.isfinite(target_mean_kv): raise C5Error("optimizer requires 160 cells and finite target")
    return optimize_action_cells(rows, math.ceil(target_mean_kv*len(rows)*1000)-(1 if strict else 0))

def schedule_digest(schedules: Mapping[str, Any]) -> str:
    if list(schedules)!=list(TARGETS): raise C5Error("schedule target order differs")
    fields=("feasible","target_mean_kv_bytes","strict","integer_microbyte_cap","integer_microbytes_used","table")
    rows=[]
    for target in TARGETS:
        row=schedules[target]
        if not isinstance(row,Mapping) or set(row)!=set(fields): raise C5Error("schedule schema differs")
        rows.append({key:row[key] for key in fields})
    return hashlib.sha256(canonical_json({"target_order":list(TARGETS),"schedules":rows}).encode("ascii")).hexdigest()


def development_passing_targets(result: Mapping[str, Any]) -> list[str]:
    p=validate_protocol(); schedules=result.get("schedules",{}); metrics=result.get("development_target_metrics",{}); bases=result.get("development_baseline_metrics",{})
    passed=[]
    for target in TARGETS:
        row=metrics.get(target,{}); schedule=schedules.get(target,{}); flat=bases.get("flat5",{}); uniform=bases.get("uniform10",{})
        if schedule.get("feasible") and row.get("mean_cosine",-math.inf)>=p["quality_gates"]["mean_cosine_gte"] and row.get("mean_relative_l2",math.inf)<=p["quality_gates"]["mean_relative_l2_lte"] and row.get("mean_relative_l2",math.inf)<uniform.get("mean_relative_l2",-math.inf) and row.get("mean_total_kv_bytes",math.inf)<flat.get("mean_total_kv_bytes",-math.inf): passed.append(target)
    return passed


def freeze_development(result: Mapping[str, Any]) -> dict[str, Any]:
    """Classify only after all schedules are present; it never selects data."""
    passing=development_passing_targets(result); out=dict(result); out["development_passing_targets"]=passing
    out["classification"]="C5-DEVELOPMENT-NO-GO" if not passing else "C5-DEVELOPMENT-GO"
    out["selected_target"] = None if not passing else passing[0]
    return out


def assert_holdout_may_start(development: Mapping[str, Any], development_sha256: str, development_path: Path | None=None) -> str:
    if development.get("classification") != "C5-DEVELOPMENT-GO": raise C5Error("C5 development GO is required before prospective selection")
    if development_path is not None and (not development_path.is_file() or sha256_path(development_path)!=development_sha256 or _json(development_path)!=dict(development)): raise C5Error("development bytes are not the frozen bytes")
    if not re.fullmatch(r"[0-9a-f]{64}", development_sha256) or development.get("schedule_sha256") != schedule_digest(development.get("schedules",{})): raise C5Error("development result/schedule digest not frozen")
    passing=development_passing_targets(development)
    if development.get("development_passing_targets") != passing or development.get("selected_target") not in passing: raise C5Error("selected target is not a frozen passing target")
    return str(development["selected_target"])


def audit_prospective_test(*, opened_development: list[str], opened_holdout: list[str], optimizer_rerun: bool, schedule_before: str, schedule_after: str) -> dict[str, Any]:
    if opened_development or optimizer_rerun or schedule_before != schedule_after: raise C5Error("prospective test firewall failed")
    return {"opened_development_tensor_identities":[],"opened_holdout_tensor_identities":opened_holdout,"optimizer_rerun":False,"schedule_sha256_before":schedule_before,"schedule_sha256_after":schedule_after,"schedule_unchanged":True}


# Execution adapters below deliberately reuse C4's frozen capture/evaluation
# primitives.  They are entered only by Kaggle commands, never by preflight.
def _expected_tensor_ids(kind: str, holdout: Mapping[str, Any] | None=None) -> list[str]:
    if kind=="development": return sorted(f"{f}:{i}:development:L{l}" for f in c4.FAMILIES for i in DEVELOPMENT[f] for l in c4.LAYERS)
    if not holdout: raise C5Error("holdout manifest required")
    return sorted(f"{x['family']}:{x['dataset_index']}:holdout:L{l}" for x in holdout["selected_sources"] for l in c4.LAYERS)

def _source_identity(source: Mapping[str, Any], role: str) -> str: return f"{source['family']}:{source['dataset_index']}:{role}"
def _no_raw_text(value: Any) -> None: c4._no_raw_text(value)

def validate_development_manifest(payload: Mapping[str, Any], path: Path | None=None) -> list[dict[str, Any]]:
    _no_raw_text(payload); required={"schema_version","protocol_sha256","target","tokenization_contract","source_specs","selected_sources","rejected_sources","global_input_ids_sha256_unique","source_text_stored"}
    if set(payload)!=required or payload.get("schema_version")!="cascadekv-phi35-8k-c5-development-sources-v1" or payload.get("protocol_sha256")!=sha256_path(PROTOCOL) or payload.get("target")!={"model":c4.MODEL,"model_revision":c4.REVISION,"tokenizer_revision":c4.REVISION} or payload.get("tokenization_contract")!={"add_special_tokens":True,"truncation":True,"max_length":8192,"required_input_ids_length":8192,"canonicalization":"CPU contiguous int64 bytes"} or payload.get("source_specs")!=c4.SOURCE_SPECS or payload.get("source_text_stored") is not False or payload.get("global_input_ids_sha256_unique") is not True: raise C5Error("C5 development manifest differs")
    sources=payload.get("selected_sources"); hashes=set(); out=[]
    if not isinstance(sources,list) or len(sources)!=12: raise C5Error("C5 requires exactly 12 development sources")
    for family in c4.FAMILIES:
        rows=[x for x in sources if isinstance(x,Mapping) and x.get("family")==family]
        if [x.get("dataset_index") for x in rows] != list(DEVELOPMENT[family]): raise C5Error("C5 development identity differs")
        for source in rows:
            if source.get("dataset_index")==14 and family=="qa" or any(source.get(k)!=c4.SOURCE_SPECS[family][k] for k in ("dataset","config","split","revision","field")): raise C5Error("forbidden/mutated C5 source")
            digest=source.get("input_ids_sha256")
            if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest) or digest in hashes: raise C5Error("C5 source hashes must be globally unique")
            hashes.add(digest); out.append(dict(source,role="development"))
    if path is not None and not path.is_file(): raise C5Error("C5 source manifest missing")
    return out

def build_development_manifest(*, output: Path) -> dict[str, Any]:
    """Kaggle-only reacquisition of the fixed 12 identities; stores no text."""
    if output.exists(): raise C5Error("refusing to overwrite development manifest")
    from cascadekv.phi35_8k_source_selection import frozen_specs,load_dataset_for_spec,load_frozen_tokenizer,mechanical_proof
    import torch
    tokenizer,specs=load_frozen_tokenizer(),frozen_specs(); selected=[]; rejected={f:[] for f in c4.FAMILIES}; seen=set()
    for family in c4.FAMILIES:
        dataset=load_dataset_for_spec(specs[family])
        for index in DEVELOPMENT[family]:
            value=dataset[index].get(specs[family]["field"]); proof=mechanical_proof(dataset,tokenizer,specs[family],index)
            if not c4._proof_eligible(proof) or not isinstance(value,str): raise C5Error("fixed development identity is no longer eligible")
            ids=torch.tensor([tokenizer(value,add_special_tokens=True,truncation=True,max_length=8192).input_ids],dtype=torch.int64).contiguous(); digest=c4._input_hash(ids)
            if digest in seen: raise C5Error("C5 development exact input duplicate")
            seen.add(digest); selected.append({"family":family,**{k:specs[family][k] for k in ("dataset","config","split","revision","field")},"identity_kind":"dataset_index","dataset_index":index,"input_ids_sha256":digest,"proof":proof})
    value={"schema_version":"cascadekv-phi35-8k-c5-development-sources-v1","protocol_sha256":sha256_path(PROTOCOL),"target":{"model":c4.MODEL,"model_revision":c4.REVISION,"tokenizer_revision":c4.REVISION},"tokenization_contract":{"add_special_tokens":True,"truncation":True,"max_length":8192,"required_input_ids_length":8192,"canonicalization":"CPU contiguous int64 bytes"},"source_specs":c4.SOURCE_SPECS,"selected_sources":selected,"rejected_sources":rejected,"global_input_ids_sha256_unique":True,"source_text_stored":False}; validate_development_manifest(value); atomic_json(output,value); return value

def select_holdout_from_inspections(inspections: Mapping[str, Iterable[Mapping[str, Any]]], development_hashes: Iterable[str], *, development_result_sha256: str, schedule_sha256: str, development_source_manifest_sha256: str) -> dict[str, Any]:
    known=set(development_hashes)
    if len(known)!=12 or any(not re.fullmatch(r"[0-9a-f]{64}",x) for x in known): raise C5Error("twelve development hashes required")
    selected=[]; rejected={f:[] for f in c4.FAMILIES}; next_frontier={}
    for family in c4.FAMILIES:
        accepted=None; last=FRONTIER[family]-1
        for row in sorted((dict(x) for x in inspections.get(family,[])),key=lambda x:x.get("dataset_index",-1)):
            index=row.get("dataset_index");
            if not isinstance(index,int) or index<FRONTIER[family]: continue
            last=index; proof=row.get("proof",{}); digest=row.get("input_ids_sha256")
            if not c4._proof_eligible(proof): rejected[family].append({"dataset_index":index,"reason":"mechanically_ineligible","proof":proof}); continue
            if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest): raise C5Error("malformed input hash")
            if digest in known: rejected[family].append({"dataset_index":index,"reason":"duplicate_exact_input_ids_sha256","input_ids_sha256":digest,"proof":proof}); continue
            source=dict(row["source"]); _no_raw_text(source)
            if source.get("family")!=family or source.get("dataset_index")!=index: raise C5Error("holdout source identity differs")
            accepted={**source,"role":"holdout","proof":proof,"input_ids_sha256":digest}; known.add(digest); break
        if accepted is None: raise C5Error("no new eligible C5 holdout")
        selected.append(accepted); next_frontier[family]=last+1
    return {"schema_version":"cascadekv-phi35-8k-c5-holdout-selection-v1","protocol_sha256":sha256_path(PROTOCOL),"development_source_manifest_sha256":development_source_manifest_sha256,"development_result_sha256":development_result_sha256,"schedule_sha256":schedule_sha256,"selected_target":None,"selection_rule":"first mechanically eligible globally-new exact input SHA in ascending order","selected_sources":selected,"rejected_sources":rejected,"next_untouched_frontier":next_frontier,"raw_text_persisted":False}

class CaptureResolver:
    """C5 manifest-first tensor resolver; validates before payload opening."""
    def __init__(self, root: Path, manifest: Path, kind: str, *, development_source_manifest: Path|None=None, holdout_manifest: Path|None=None, development_sha: str|None=None, schedule_sha: str|None=None):
        self.root=root.resolve(); self.kind=kind; payload=_json(manifest); _no_raw_text(payload); expected_count=60 if kind=="development" else 15; expected_forwards=12 if kind=="development" else 3
        required={"schema_version","kind","protocol_sha256","qualification_sha256","backend_id","development_source_manifest_sha256","holdout_manifest_sha256","development_result_sha256","schedule_sha256","artifact_count","forwards","artifacts"}
        if set(payload)!=required or payload.get("schema_version")!="cascadekv-phi35-8k-c5-capture-v1" or payload.get("kind")!=kind or payload.get("protocol_sha256")!=sha256_path(PROTOCOL) or payload.get("qualification_sha256")!=c4.QUALIFICATION_SHA or payload.get("backend_id")!=c4.BACKEND_ID or payload.get("artifact_count")!=expected_count or payload.get("forwards")!=expected_forwards: raise C5Error("C5 capture manifest differs")
        holdout = None
        if kind=="development":
            if development_source_manifest is None or not development_source_manifest.is_file() or payload.get("development_source_manifest_sha256") != sha256_path(development_source_manifest) or any(payload.get(x) is not None for x in ("holdout_manifest_sha256","development_result_sha256","schedule_sha256")): raise C5Error("development capture binding differs")
        else:
            if holdout_manifest is None or not holdout_manifest.is_file(): raise C5Error("actual holdout manifest path is required")
            holdout=_json(holdout_manifest)
            if payload.get("development_source_manifest_sha256") is not None or payload.get("holdout_manifest_sha256") != sha256_path(holdout_manifest) or payload.get("development_result_sha256") != development_sha or payload.get("schedule_sha256") != schedule_sha: raise C5Error("holdout capture binding differs")
        expected=_expected_tensor_ids(kind,holdout); records={}
        for row in payload.get("artifacts",[]):
            if not isinstance(row,Mapping) or set(row)!={"identity","layer","artifact_relative_path","artifact_sha256","provenance_relative_path","provenance_sha256","input_ids_sha256","source"}: raise C5Error("capture record schema differs")
            key=f"{row['identity']}:L{row['layer']}"
            if key not in expected or key in records or not all(isinstance(row[x],str) and re.fullmatch(r"[0-9a-f]{64}",row[x]) for x in ("artifact_sha256","provenance_sha256","input_ids_sha256")): raise C5Error("capture inventory differs")
            records[key]=dict(row)
        if sorted(records)!=expected: raise C5Error("capture inventory incomplete")
        self.records=records
    def ordered(self): return [self.records[x] for x in sorted(self.records)]

class TensorAccess:
    def __init__(self,resolver:CaptureResolver,*,allowed_kind:str):
        if resolver.kind!=allowed_kind: raise C5Error("tensor firewall denied capture kind")
        self.resolver=resolver; self.allowed_kind=allowed_kind; self.opened=[]
    def open(self,record:Mapping[str,Any]):
        root=self.resolver.root
        def safe(rel):
            path=(root/rel).resolve()
            if not path.is_relative_to(root) or path.is_symlink(): raise C5Error("artifact path escapes root")
            return path
        artifact,provenance=safe(str(record["artifact_relative_path"])),safe(str(record["provenance_relative_path"]))
        if not artifact.is_file() or not provenance.is_file() or sha256_path(artifact)!=record["artifact_sha256"] or sha256_path(provenance)!=record["provenance_sha256"]: raise C5Error("artifact digest differs")
        detail=_json(provenance); _no_raw_text(detail)
        if detail.get("protocol_sha256")!=sha256_path(PROTOCOL) or detail.get("identity")!=record["identity"] or detail.get("layer")!=record["layer"] or detail.get("input_ids_sha256")!=record["input_ids_sha256"] or detail.get("artifact_sha256")!=record["artifact_sha256"]: raise C5Error("artifact provenance differs")
        from safetensors import safe_open
        import torch
        with safe_open(str(artifact),framework="pt",device="cpu") as h:
            if set(h.keys())!={"q","k","v"}: raise C5Error("tensor keys differ")
            q,k,v=(h.get_tensor(x) for x in ("q","k","v"))
        if any(x.dtype!=torch.float16 or tuple(x.shape)!=c4.SHAPE for x in (q,k,v)): raise C5Error("tensor shape/dtype differs")
        self.opened.append(f"{record['identity']}:L{record['layer']}"); return q,k,v

def _method_rows(access: TensorAccess, protocol: Mapping[str,Any], schedules: Mapping[str,Mapping[str,str]]|None=None) -> dict[str,list[dict[str,Any]]]:
    from cascadekv.evaluation_core import advance_until_budget,build_forests,candidate_budget,exact_attention_reference,exact_rerank_and_output,initialize_route,reserve_ids,select_flat_k4,traffic
    from cascadekv.model_agnostic import PHI35_GEOMETRY
    methods={x:[] for x in (*c4.BASELINES,*ACTIONS)}
    if schedules: methods.update({x:[] for x in schedules})
    for record in access.resolver.ordered():
        q_all,k_all,v_all=access.open(record); layer=int(record["layer"]); forests=build_forests(k_all[0],PHI35_GEOMETRY,c4.POSITIONS,atom_size=8,groups=(4,))
        for position in c4.POSITIONS:
            n=position+1
            for head in c4.HEADS:
                q,k,v=q_all[0,head,position],k_all[0,head,:n],v_all[0,head,:n]; reference,forest=exact_attention_reference(q,k,v),forests[position,head]
                def flat(frac):
                    budget=candidate_budget(n,frac); return set(select_flat_k4(q,k,budget)),traffic({"active_root_reads":0,"detail_reads":0,"expanded_internal_nodes":0,"variance_scalar_reads":0},n,budget,96,variance=False,flat=True)
                def hierarchy(desc):
                    budget=candidate_budget(n,float(desc["candidate_fraction"])); profile=protocol["methods"]["routing_profiles"][desc["routing_profile"]]; lambdas={int(x):float(y) for x,y in profile["lambdas"].items()}; reserve=profile["reserve"]; route=advance_until_budget(initialize_route(forest,q,4,lambdas[layer],reserve_ids(n,int(reserve["sink"]),int(reserve["local"]),budget)),budget); return set(route["ids"]),traffic(route,n,budget,96,variance=True)
                choices={"dense":(set(range(n)),{"total_k_bytes":float(n*192),"selected_v_fp16_bytes":float(n*192)}),"flat5":flat(.05),"flat10":flat(.10),"uniform5":hierarchy({"candidate_fraction":.05,"routing_profile":"phi35-8k-depth-transfer-qwen5-v1"}),"uniform10":hierarchy({"candidate_fraction":.10,"routing_profile":"phi35-8k-depth-transfer-qwen10-v1"})}
                for action,desc in protocol["methods"]["actions"].items(): choices[action]=flat(.05) if desc["kind"]=="flat_q8k4" else hierarchy(desc)
                common={"source":record["identity"],"layer":layer,"position":position,"head":head}
                for name,(ids,t) in choices.items():
                    item=dict(t); item["total_kv_bytes"]=item["total_k_bytes"]+item["selected_v_fp16_bytes"]; methods[name].append({**common,"metrics":exact_rerank_and_output(q,k,v,ids,reference=reference),"traffic":item})
                if schedules:
                    for target,table in schedules.items(): methods[target].append(dict(methods[table[f"{layer}:{head}"]][-1]))
    return methods

def _summary(rows): return {"observation_count":len(rows),"mean_cosine":mean(float(x["metrics"]["cosine_similarity"]) for x in rows),"mean_relative_l2":mean(float(x["metrics"]["relative_l2_error"]) for x in rows),"mean_total_kv_bytes":mean(float(x["traffic"]["total_kv_bytes"]) for x in rows)}

def rows_for_schedule(method_rows: Mapping[str, list[Mapping[str, Any]]], schedule_table: Mapping[str, str]) -> list[Mapping[str, Any]]:
    """Select exactly one action observation per source/layer/position/head.

    This intentionally does not depend on the order used by the evaluator.  A
    schedule is a complete 160-cell map and each source observation key must
    occur exactly once in every action stream.
    """
    expected_cells={f"{layer}:{head}" for layer in c4.LAYERS for head in c4.HEADS}
    if not isinstance(schedule_table, Mapping) or set(schedule_table) != expected_cells:
        raise C5Error("schedule table is missing or contains extra cells")
    if any(action not in ACTIONS for action in schedule_table.values()):
        raise C5Error("schedule table contains unknown action")
    selected=[]; seen=set()
    for action in ACTIONS:
        indexed={}
        for row in method_rows.get(action, []):
            key=(row.get("source"), row.get("layer"), row.get("position"), row.get("head"))
            if key in indexed: raise C5Error("duplicate action observation")
            indexed[key]=row
        if len(indexed) != 5760: raise C5Error("action observation inventory differs")
        for key,row in indexed.items():
            if f"{key[1]}:{key[3]}" in expected_cells and schedule_table[f"{key[1]}:{key[3]}"] == action:
                if key in seen: raise C5Error("duplicate scheduled observation")
                seen.add(key); selected.append(row)
    if len(selected) != 5760: raise C5Error("scheduled observations missing")
    return sorted(selected, key=lambda r:(str(r["source"]),int(r["layer"]),int(r["position"]),int(r["head"])))

def _development_schedules(rows, protocol):
    stats={}; groups=[]
    for layer in c4.LAYERS:
        for head in c4.HEADS:
            group={}
            for action in ACTIONS:
                values=[x for x in rows[action] if x["layer"]==layer and x["head"]==head]
                if len(values)!=36: raise C5Error("each C5 cell requires 36 observations")
                group[action]={"relative_l2":mean(x["metrics"]["relative_l2_error"] for x in values),"cosine":mean(x["metrics"]["cosine_similarity"] for x in values),"total_kv_bytes":mean(x["traffic"]["total_kv_bytes"] for x in values)}; stats[f"{layer}:{head}:{action}"]=group[action]
            groups.append(group)
    dense,flat5=_summary(rows["dense"])["mean_total_kv_bytes"],_summary(rows["flat5"])["mean_total_kv_bytes"]; schedules={}
    for target in TARGETS:
        target_mean,strict,cap=_target_cap(protocol,target,dense,flat5); found=optimize_static_schedule(groups,target_mean,strict=strict)
        if found is None: schedules[target]={"feasible":False,"target_mean_kv_bytes":target_mean,"strict":strict,"integer_microbyte_cap":cap,"integer_microbytes_used":None,"table":None}; continue
        used,_error,_cos,actions=found; table={f"{l}:{h}":a for (l,h),a in zip(((l,h) for l in c4.LAYERS for h in c4.HEADS),actions,strict=True)}
        schedules[target]={"feasible":True,"target_mean_kv_bytes":target_mean,"strict":strict,"integer_microbyte_cap":cap,"integer_microbytes_used":used,"table":table}
    return stats,schedules

def develop(*, capture_root:Path,capture_manifest:Path,development_source_manifest:Path,qualification:Path,output:Path) -> dict[str,Any]:
    """Evaluate all A0--A9 once and write the final immutable development result once."""
    if output.exists(): raise C5Error("refusing to overwrite development result")
    protocol=validate_protocol(); sources=validate_development_manifest(_json(development_source_manifest),development_source_manifest); c4._validate_c2_qualification(qualification)
    resolver=CaptureResolver(capture_root,capture_manifest,"development",development_source_manifest=development_source_manifest); access=TensorAccess(resolver,allowed_kind="development"); rows=_method_rows(access,protocol); expected=_expected_tensor_ids("development")
    if access.opened!=expected: raise C5Error("development tensor firewall differs")
    summaries={x:_summary(value) for x,value in rows.items()}
    if any(x["observation_count"]!=5760 for x in summaries.values()): raise C5Error("C5 development observation count differs")
    stats,schedules=_development_schedules(rows,protocol)
    result={"schema_version":"cascadekv-phi35-8k-c5-development-v1","protocol_sha256":sha256_path(PROTOCOL),"runtime_manifest_sha256":sha256_path(RUNTIME),"qualification_sha256":sha256_path(qualification),"development_source_manifest_sha256":sha256_path(development_source_manifest),"capture_manifest_sha256":sha256_path(capture_manifest),"opened_development_tensor_identities":expected,"opened_holdout_tensor_identities":[],"methods":protocol["methods"],"development_baseline_metrics":{x:summaries[x] for x in c4.BASELINES},"development_action_metrics":{x:summaries[x] for x in ACTIONS},"action_cell_statistics":stats,"schedules":schedules,"development_target_metrics":{},"target_feasibility":{x:schedules[x]["feasible"] for x in TARGETS},"optimizer":protocol["optimizer"]}
    schedule_rows={}
    for target in TARGETS:
        if not schedules[target]["feasible"]: schedule_rows[target]={}; continue
        table=schedules[target]["table"]
        schedule_rows[target]=_summary(rows_for_schedule(rows,table))
    result["development_target_metrics"]=schedule_rows; result["schedule_sha256"]=schedule_digest(schedules); result=freeze_development(result); atomic_json(output,result); return result

def prospective_test(*, holdout_capture_root:Path,holdout_capture_manifest:Path,holdout_manifest:Path,development_result:Path,output:Path) -> dict[str,Any]:
    """One-shot holdout evaluator.  Its signature makes development tensors unavailable."""
    if output.exists(): raise C5Error("refusing to overwrite prospective result")
    dev=_json(development_result); dev_sha=sha256_path(development_result); selected=assert_holdout_may_start(dev,dev_sha,development_result); holdout=_json(holdout_manifest)
    if holdout.get("development_result_sha256")!=dev_sha or holdout.get("schedule_sha256")!=dev["schedule_sha256"] or holdout.get("selected_target")!=selected: raise C5Error("holdout is not bound to frozen development")
    resolver=CaptureResolver(holdout_capture_root,holdout_capture_manifest,"holdout",holdout_manifest=holdout_manifest,development_sha=dev_sha,schedule_sha=dev["schedule_sha256"]); access=TensorAccess(resolver,allowed_kind="holdout"); schedules={x:dev["schedules"][x]["table"] for x in TARGETS if dev["schedules"][x]["feasible"]}; rows=_method_rows(access,validate_protocol(),schedules); expected=_expected_tensor_ids("holdout",holdout)
    if access.opened!=expected: raise C5Error("holdout tensor firewall differs")
    summaries={x:_summary(v) for x,v in rows.items()}
    if any(x["observation_count"]!=1440 for x in summaries.values()): raise C5Error("holdout observation count differs")
    protocol=validate_protocol(); quality=protocol["quality_gates"]; gates={}; first=None
    for target in TARGETS:
        if target not in schedules: gates[target]={"feasible":False,"passes":False}; continue
        m=summaries[target]; uniform=quality["winner_relative_l2_lt"]; flat=quality["winner_modeled_kv_traffic_lt"]; row={"feasible":True,"mean_cosine_gte":m["mean_cosine"]>=quality["mean_cosine_gte"],"mean_relative_l2_lte":m["mean_relative_l2"]<=quality["mean_relative_l2_lte"],"relative_l2_lt_uniform10":m["mean_relative_l2"]<summaries[uniform]["mean_relative_l2"],"modeled_kv_traffic_lt_flat5":m["mean_total_kv_bytes"]<summaries[flat]["mean_total_kv_bytes"]}; row["passes"]=all(v for k,v in row.items() if k not in {"feasible","passes"}); gates[target]=row
        if first is None and row["passes"]: first=target
    before=dev["schedule_sha256"]; after=schedule_digest(dev["schedules"])
    audit=audit_prospective_test(opened_development=[],opened_holdout=expected,optimizer_rerun=False,schedule_before=before,schedule_after=after)
    result={"schema_version":"cascadekv-phi35-8k-c5-test-v1","protocol_sha256":sha256_path(PROTOCOL),"runtime_manifest_sha256":sha256_path(RUNTIME),"development_result_sha256":dev_sha,"holdout_manifest_sha256":sha256_path(holdout_manifest),"capture_manifest_sha256":sha256_path(holdout_capture_manifest),"selected_target":selected,"development_passing_targets":dev["development_passing_targets"],"baseline_test_metrics":{x:summaries[x] for x in c4.BASELINES},"target_test_metrics":{x:summaries[x] for x in TARGETS if x in summaries},"per_gate_pass_fail":gates,"first_passing_target_descriptive":first,"firewall_evidence":audit,"classification":"C5-PROSPECTIVE-PASS" if gates[selected]["passes"] else "C5-PROSPECTIVE-NO-PASS"}; atomic_json(output,result); return result

def capture_sources(*, kind:str, sources:list[dict[str,Any]], output:Path, qualification:Path, development_source_sha:str|None=None, holdout_manifest:Path|None=None, development_sha:str|None=None, schedule_sha:str|None=None) -> dict[str,Any]:
    """C5 capture: exactly one pinned model forward and five artifacts per source."""
    if output.exists() and (output/"final_capture_manifest.json").exists(): raise C5Error("refusing to overwrite completed capture")
    qrecord=c4._validate_c2_qualification(qualification)
    from cascadekv import phi35_kaggle_c2 as c2
    from cascadekv.phi35_8k_source_selection import load_dataset_for_spec,load_frozen_tokenizer,mechanical_proof
    import torch
    tokenizer=load_frozen_tokenizer(); model=c2._load_pinned_model(); records=[]
    try:
        c2.validate_qualification(qrecord); c2.assert_capture_backend(model,qrecord)
        for source in sources:
            spec={x:source[x] for x in ("dataset","config","split","revision","field")}; dataset=load_dataset_for_spec(spec); proof=mechanical_proof(dataset,tokenizer,spec,int(source["dataset_index"])); value=dataset[int(source["dataset_index"])].get(source["field"])
            if not c4._proof_eligible(proof) or not isinstance(value,str): raise C5Error("captured source no longer eligible")
            ids=torch.tensor([tokenizer(value,add_special_tokens=True,truncation=True,max_length=8192).input_ids],dtype=torch.int64).contiguous(); digest=c4._input_hash(ids)
            if digest!=source["input_ids_sha256"]: raise C5Error("captured input hash reproof differs")
            identity=_source_identity(source,kind); captured=c2.capture_five_layers_post_rope_qkv(model,ids.to(c2._first_device(model)))
            for layer in c4.LAYERS:
                stem=f"{source['family']}_{kind}_index{source['dataset_index']}_layer{layer}"; artifact_rel=f"artifacts/{stem}.safetensors"; provenance_rel=f"artifacts/{stem}.provenance.json"; artifact=output/artifact_rel; provenance=output/provenance_rel
                if artifact.exists() or provenance.exists(): raise C5Error("never overwrite C5 artifact")
                detail={"schema_version":"cascadekv-phi35-8k-c5-artifact-v1","protocol_sha256":sha256_path(PROTOCOL),"qualification_sha256":c4.QUALIFICATION_SHA,"backend_id":c4.BACKEND_ID,"identity":identity,"layer":layer,"input_ids_sha256":digest,"target":{"model":c4.MODEL,"revision":c4.REVISION,"tokenizer_revision":c4.REVISION},"source":{**{x:source[x] for x in ("family","dataset","config","split","revision","field","identity_kind","dataset_index")},"role":kind},"source_proof":proof,"q_shape":list(c4.SHAPE),"k_shape":list(c4.SHAPE),"v_shape":list(c4.SHAPE),"storage_dtype":"float16","model_compute_dtype":"float16","attention_implementation":"sdpa","use_cache":False,"quantization":"none","capture_adapter":"phi3-post-rope-qkv-v1"}
                # The canonical C2 writer writes the artifact once, computes its
                # final digest, and atomically writes the matching provenance.
                written=c2.write_artifact(artifact,provenance,captured[layer],detail)
                artifact_sha=sha256_path(artifact)
                if written.get("artifact_sha256") != artifact_sha or _json(provenance) != written:
                    raise C5Error("capture writer provenance chain differs")
                records.append({"identity":identity,"layer":layer,"artifact_relative_path":artifact_rel,"artifact_sha256":artifact_sha,"provenance_relative_path":provenance_rel,"provenance_sha256":sha256_path(provenance),"input_ids_sha256":digest,"source":{"family":source["family"],"dataset_index":source["dataset_index"],"role":kind}})
        final={"schema_version":"cascadekv-phi35-8k-c5-capture-v1","kind":kind,"protocol_sha256":sha256_path(PROTOCOL),"qualification_sha256":c4.QUALIFICATION_SHA,"backend_id":c4.BACKEND_ID,"development_source_manifest_sha256":development_source_sha if kind=="development" else None,"holdout_manifest_sha256":sha256_path(holdout_manifest) if holdout_manifest else None,"development_result_sha256":development_sha,"schedule_sha256":schedule_sha,"artifact_count":len(records),"forwards":len(sources),"artifacts":records}; atomic_json(output/"final_capture_manifest.json",final); return final
    finally: del model

def select_holdout(*, development_source_manifest:Path, development_result:Path, output:Path) -> dict[str,Any]:
    """Kaggle-only prospective selector; all freeze checks happen before imports."""
    dev=_json(development_result); devsha=sha256_path(development_result); selected_target=assert_holdout_may_start(dev,devsha,development_result); sources=validate_development_manifest(_json(development_source_manifest),development_source_manifest)
    from cascadekv.phi35_8k_source_selection import frozen_specs,load_dataset_for_spec,load_frozen_tokenizer,mechanical_proof
    import torch
    tokenizer,specs=load_frozen_tokenizer(),frozen_specs(); inspections={f:[] for f in c4.FAMILIES}; known={x["input_ids_sha256"] for x in sources}
    for family in c4.FAMILIES:
        dataset=load_dataset_for_spec(specs[family]); index=FRONTIER[family]
        while index<len(dataset):
            proof=mechanical_proof(dataset,tokenizer,specs[family],index); value=dataset[index].get(specs[family]["field"]); digest="0"*64
            if c4._proof_eligible(proof) and isinstance(value,str): digest=c4._input_hash(torch.tensor([tokenizer(value,add_special_tokens=True,truncation=True,max_length=8192).input_ids],dtype=torch.int64).contiguous())
            inspections[family].append({"dataset_index":index,"proof":proof,"input_ids_sha256":digest,"source":{"family":family,**{x:specs[family][x] for x in ("dataset","config","split","revision","field")},"identity_kind":"dataset_index","dataset_index":index}})
            if c4._proof_eligible(proof) and digest not in known: break
            index+=1
    result=select_holdout_from_inspections(inspections,[x["input_ids_sha256"] for x in sources],development_result_sha256=devsha,schedule_sha256=dev["schedule_sha256"],development_source_manifest_sha256=sha256_path(development_source_manifest)); result["selected_target"]=selected_target; atomic_json(output,result); return result
