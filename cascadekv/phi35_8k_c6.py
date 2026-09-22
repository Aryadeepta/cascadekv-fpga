"""C6's narrow A7--A9 expansion and development/holdout firewall.

This module is intentionally data-free on import and during preflight.  The
executor is the only component allowed to call C6 capture/selection commands.
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

from . import phi35_8k_c5 as c5
from . import phi35_8k_c4 as c4

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cascadekv_phi35_8k_c6_protocol.json"
RUNTIME = ROOT / "configs/cascadekv_phi35_8k_c6_runtime_manifest.json"
C6_TAG = "cascadekv-phi35-8k-c6-freeze-v1"
C5_PARENT = {"recovered_result_tag":"cascadekv-phi35-8k-c5-result-recovered-v1", "closure_commit":"62e4a5a8dc1977908b8c9b9848e4ac4a087f4ce4", "freeze_tag":"cascadekv-phi35-8k-c5-freeze-v2", "freeze_commit":"36783eb8c75e9d4006d8ef4bae7d527942e95331", "protocol_sha256":"2ab29d9ef370ba2f9308763e916b0c15795fbe379e317bf95bba20ae27b412d1", "runtime_sha256":"c6644332a676a839450814bcd2c8c22d5bfee4bb445c5e34099527b07511a818", "recovered_result_sha256":"024869d35ee3956a58cf2dec7fec6ada3f629259d7bc557861b7a43e6e82754f", "recovered_postmortem_sha256":"df95eefbac6b726c454f0c5261e85928ed45127295dd027b9b591a2b5208d117", "stdout_reconstruction_is_not_original_kaggle_container":True}
ACTIONS = tuple(f"A{x}" for x in range(12)); BASE_ACTIONS=ACTIONS[:10]; TARGETS = c4.TARGETS
DEVELOPMENT = {"narrative": (13, 14, 15, 16, 17), "report": (16, 17, 18, 19, 21), "qa": (13, 15, 16, 17, 18)}
FRONTIER = {"narrative": 18, "qa": 19, "report": 22}


class C6Error(RuntimeError): pass


def canonical_json(value: Any) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""): h.update(part)
    return h.hexdigest()
def _json(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc: raise C6Error(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict): raise C6Error("JSON root must be object")
    return value
def _git(ref: str) -> str:
    try: return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{ref}^{{commit}}"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc: raise C6Error(f"required git reference unavailable: {ref}") from exc


def action_definitions() -> dict[str, dict[str, Any]]:
    return {**c5.action_definitions(), "A10":{"kind":"hierarchy","candidate_fraction":.25,"routing_profile":"phi35-8k-depth-transfer-qwen5-v1"}, "A11":{"kind":"hierarchy","candidate_fraction":.25,"routing_profile":"phi35-8k-depth-transfer-qwen10-v1"}}

def allowed_actions(layer: int) -> tuple[str, ...]:
    if layer not in c4.LAYERS: raise C6Error("unsampled layer")
    return ACTIONS if layer == 0 else BASE_ACTIONS

def action_domain_by_cell() -> tuple[tuple[str, ...], ...]:
    return tuple(allowed_actions(layer) for layer in c4.LAYERS for _ in c4.HEADS)

def validate_schedule_actions(table: Mapping[str, str]) -> None:
    expected={f"{layer}:{head}" for layer in c4.LAYERS for head in c4.HEADS}
    if not isinstance(table, Mapping) or set(table)!=expected: raise C6Error("schedule table inventory differs")
    if any(action not in allowed_actions(int(cell.split(":")[0])) for cell,action in table.items()): raise C6Error("schedule violates C6 layer action domain")

def validate_schedule(table: Iterable[str]) -> tuple[str, ...]:
    values=tuple(table)
    if len(values)!=160: raise C6Error("schedule must contain 160 cells")
    validate_schedule_actions({f"{layer}:{head}": values[n] for n,(layer,head) in enumerate((x,y) for x in c4.LAYERS for y in c4.HEADS)})
    return values


def validate_protocol(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = dict(_json(PROTOCOL) if payload is None else payload)
    required = {"schema_version","c5_result_parent","geometry","methods","development_sources","prospective_frontier","targets","quality_gates","optimizer","development_gate","holdout_selection","firewall","scientific_state","action_validity","lifecycle","raw_text_persisted"}
    if set(p) != required or p.get("schema_version") != "cascadekv-phi35-8k-c6-protocol-v1": raise C6Error("C6 protocol schema differs")
    if p["c5_result_parent"] != C5_PARENT: raise C6Error("C5 recovered-result parent differs")
    c4p = c4.validate_protocol()
    structural = ("context_length", "layers", "query_positions", "heads", "head_dim", "schedule_cells")
    if any(p["geometry"].get(key) != c4p["geometry"].get(key) for key in structural): raise C6Error("C6 structural geometry differs")
    if p["geometry"].get("observations_per_source_per_method") != 480 or p["geometry"].get("development_observations_per_method") != 7200 or p["geometry"].get("holdout_observations_per_method") != 1440: raise C6Error("C6 observation geometry differs")
    methods = p["methods"]
    if methods.get("hierarchy") != c4p["methods"]["hierarchy"] or methods.get("baselines") != c4p["methods"]["baselines"] or methods.get("routing_profiles") != c4p["methods"]["routing_profiles"]: raise C6Error("frozen method/profile mutation")
    if methods.get("actions") != action_definitions() or list(methods["actions"]) != list(ACTIONS): raise C6Error("C6 action order/menu differs")
    expected_dev = {"identities": {family: list(indices) for family, indices in DEVELOPMENT.items()}, "report20_excluded":True, "source_text_stored":False}
    if p["development_sources"] != expected_dev: raise C6Error("C6 development inventory differs")
    if p["prospective_frontier"] != {"start_indices": FRONTIER, "untouched_during_preparation": True}: raise C6Error("prospective frontier differs")
    if p["targets"] != c4p["targets"] or p["quality_gates"] != c4p["quality_gates"]: raise C6Error("targets or gates differ")
    if p["optimizer"] != {"version":"c6-masked-c5-parity-integer-microbyte-dp-v1","scope":"development_only","objective":c4p["optimizer"]["objective"],"action_change_rule":"one of exactly A0..A11 per 160 layer/head cells subject to frozen per-layer action domains; no iterative parameter mutation","tie_breaking":"relative-L2 ascending, cosine descending, traffic ascending, lexical action table ascending","ordering":"sampled layer ascending, then head ascending, then lexical action"}: raise C6Error("optimizer differs")
    if p["development_gate"] != {"definition":"feasible and all four frozen quality gates on development","no_go_classification":"C6-DEVELOPMENT-NO-GO","go_requires_freeze_before_selection":True,"priority_order":list(TARGETS)}: raise C6Error("development firewall differs")
    if p["holdout_selection"] != {"kaggle_only":True,"start_indices":FRONTIER,"rule":"first mechanically eligible globally-new exact input SHA in ascending order","requires_frozen_development_result_and_schedule":True,"persist_raw_text":False}: raise C6Error("holdout contract differs")
    if p["firewall"] != {"test_opens_development_tensors":False,"test_optimizer_rerun":False,"schedule_sha256_unchanged":True,"development_and_holdout_tensor_roots_distinct":True}: raise C6Error("test firewall differs")
    if p["scientific_state"] != {"local_preparation_only":True,"prospective_frontier_accessed":False,"c6_metrics_observed":False}: raise C6Error("scientific state differs")
    if p["action_validity"] != {"layer_0":list(ACTIONS),"nonzero_sampled_layers":list(BASE_ACTIONS),"enforcement":"DP enumeration domain; independent schedule validation"}: raise C6Error("action validity differs")
    if p["lifecycle"] != {"development_no_go":"C6-DEVELOPMENT-NO-GO","development_go":"C6-DEVELOPMENT-GO","freeze_before_prospective":True,"selected_target_locks_prospective_verdict":True} or p["raw_text_persisted"] is not False: raise C6Error("C6 lifecycle/raw text policy differs")
    return p


def preflight(*, execution: bool = False) -> dict[str, Any]:
    if _git(C5_PARENT["recovered_result_tag"]) != C5_PARENT["closure_commit"]: raise C6Error("C5 parent tag commit differs")
    if sha256_path(ROOT / "results/archive/cascadekv_phi35_8k_c5_v2_recovered_result.json") != C5_PARENT["recovered_result_sha256"] or sha256_path(ROOT / "results/archive/cascadekv_phi35_8k_c5_v2_recovered_postmortem.json") != C5_PARENT["recovered_postmortem_sha256"]: raise C6Error("C5 parent evidence differs")
    # Preparation is intentionally possible before the freeze tag exists.  An
    # executor, however, must be cryptographically tied to the commit that was
    # reviewed and tagged for this protocol.
    if execution and _git(C6_TAG) != _git("HEAD"):
        raise C6Error("HEAD is not the required future C6 freeze tag commit")
    validate_protocol(); closure = verify_runtime_closure()
    return {"local_only": not execution, "protocol_sha256":sha256_path(PROTOCOL), "runtime_manifest_sha256":sha256_path(RUNTIME), "bound_files": closure}


def verify_runtime_closure() -> tuple[dict[str, str], ...]:
    manifest = _json(RUNTIME)
    if set(manifest) != {"schema_version", "purpose", "protocol", "bound_files"} or manifest.get("schema_version") != 1:
        raise C6Error("C6 runtime manifest schema differs")
    if manifest["protocol"] != {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256_path(PROTOCOL)}:
        raise C6Error("C6 runtime protocol binding differs")
    rows=[]; seen=set()
    for row in manifest["bound_files"]:
        if not isinstance(row, Mapping) or set(row) != {"path","sha256"} or not isinstance(row["path"],str) or row["path"] in seen:
            raise C6Error("unsafe runtime entry")
        rel=Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))): raise C6Error("unsafe runtime entry")
        seen.add(row["path"]); path=(ROOT/rel).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.is_symlink() or sha256_path(path) != row["sha256"]: raise C6Error(f"runtime closure mismatch: {row['path']}")
        rows.append({"path":row["path"],"sha256":row["sha256"]})
    return tuple(rows)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise C6Error(f"refusing to overwrite immutable result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(canonical_json(value) + "\n"); temp = Path(handle.name)
    os.replace(temp, path)

def _finite(value: Any) -> bool: return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

def optimize_action_cells(groups: Iterable[Mapping[str, Mapping[str, float]]], cap: int, actions: tuple[str, ...] = ACTIONS, domains: Iterable[tuple[str, ...]] | None = None) -> tuple[int,float,float,tuple[str,...]] | None:
    """C4's exact Pareto DP, generalized only from radix seven to action count."""
    rows=list(groups); radix=len(actions); domains=tuple(action_domain_by_cell() if domains is None and len(rows)==160 else ((actions,)*len(rows) if domains is None else domains))
    if not rows or len(rows)!=len(domains) or type(cap) is not int or cap < 0: raise C6Error("optimizer requires cells and nonnegative integer cap")
    states: dict[int, tuple[float,float,int]]={0:(0.,0.,0)}
    for row,domain in zip(rows,domains):
        candidate: dict[int, tuple[float,float,int]]={}
        for budget,(error,cosine,code) in states.items():
            for action in domain:
                index=actions.index(action)
                stat=row.get(action,{}); traffic,rel,cos=stat.get("total_kv_bytes"),stat.get("relative_l2"),stat.get("cosine")
                if not all(_finite(x) for x in (traffic,rel,cos)): raise C6Error("incomplete action statistics")
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
    if len(rows)!=160 or not math.isfinite(target_mean_kv): raise C6Error("optimizer requires 160 cells and finite target")
    return optimize_action_cells(rows, math.ceil(target_mean_kv*len(rows)*1000)-(1 if strict else 0))

def schedule_digest(schedules: Mapping[str, Any]) -> str:
    if list(schedules)!=list(TARGETS): raise C6Error("schedule target order differs")
    fields=("feasible","target_mean_kv_bytes","strict","integer_microbyte_cap","integer_microbytes_used","table")
    rows=[]
    for target in TARGETS:
        row=schedules[target]
        if not isinstance(row,Mapping) or set(row)!=set(fields): raise C6Error("schedule schema differs")
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
    out["classification"]="C6-DEVELOPMENT-NO-GO" if not passing else "C6-DEVELOPMENT-GO"
    out["selected_target"] = None if not passing else passing[0]
    return out


def assert_holdout_may_start(development: Mapping[str, Any], development_sha256: str, development_path: Path | None=None) -> str:
    if development.get("classification") != "C6-DEVELOPMENT-GO": raise C6Error("C6 development GO is required before prospective selection")
    if development_path is not None and (not development_path.is_file() or sha256_path(development_path)!=development_sha256 or _json(development_path)!=dict(development)): raise C6Error("development bytes are not the frozen bytes")
    if not re.fullmatch(r"[0-9a-f]{64}", development_sha256) or development.get("schedule_sha256") != schedule_digest(development.get("schedules",{})): raise C6Error("development result/schedule digest not frozen")
    passing=development_passing_targets(development)
    if development.get("development_passing_targets") != passing or development.get("selected_target") not in passing: raise C6Error("selected target is not a frozen passing target")
    return str(development["selected_target"])


def audit_prospective_test(*, opened_development: list[str], opened_holdout: list[str], optimizer_rerun: bool, schedule_before: str, schedule_after: str) -> dict[str, Any]:
    if opened_development or optimizer_rerun or schedule_before != schedule_after: raise C6Error("prospective test firewall failed")
    return {"opened_development_tensor_identities":[],"opened_holdout_tensor_identities":opened_holdout,"optimizer_rerun":False,"schedule_sha256_before":schedule_before,"schedule_sha256_after":schedule_after,"schedule_unchanged":True}

def prospective_verdict(selected_target: str, gates: Mapping[str, Mapping[str, Any]]) -> str:
    """Return a verdict for the one target frozen by development.

    Other target rows remain descriptive evidence only.  In particular, a
    missing selected row is a provenance error rather than an accidental
    ``KeyError`` or an opportunity to substitute another passing target.
    """
    row = gates.get(selected_target)
    if not isinstance(row, Mapping) or type(row.get("passes")) is not bool:
        raise C6Error("selected prospective target metrics are missing")
    return "C6-PROSPECTIVE-PASS" if row["passes"] else "C6-PROSPECTIVE-NO-PASS"


# Execution adapters below deliberately reuse C4's frozen capture/evaluation
# primitives.  They are entered only by Kaggle commands, never by preflight.
def _expected_tensor_ids(kind: str, holdout: Mapping[str, Any] | None=None) -> list[str]:
    if kind=="development": return sorted(f"{f}:{i}:development:L{l}" for f in c4.FAMILIES for i in DEVELOPMENT[f] for l in c4.LAYERS)
    if not holdout: raise C6Error("holdout manifest required")
    return sorted(f"{x['family']}:{x['dataset_index']}:holdout:L{l}" for x in holdout["selected_sources"] for l in c4.LAYERS)

def _source_identity(source: Mapping[str, Any], role: str) -> str: return f"{source['family']}:{source['dataset_index']}:{role}"
def _no_raw_text(value: Any) -> None: c4._no_raw_text(value)

def validate_observation_inventory(rows: Iterable[Mapping[str, Any]], *, action: str, role: str, expected_holdout_sources: Iterable[str] | None = None) -> list[Mapping[str, Any]]:
    """Fail closed on the complete Cartesian evidence inventory, not merely its count."""
    if action not in ACTIONS or role not in {"development", "holdout"}: raise C6Error("invalid inventory contract")
    rows=list(rows)
    identities = [(f,i) for f in c4.FAMILIES for i in (DEVELOPMENT[f] if role=="development" else ())]
    if role=="holdout":
        if expected_holdout_sources is None: raise C6Error("holdout inventory requires validated manifest identities")
        source_set=set(expected_holdout_sources)
        if len(source_set)!=3 or any(not re.fullmatch(r"(?:narrative|report|qa):[0-9]+:holdout", x) for x in source_set): raise C6Error("holdout source inventory differs")
    else:
        source_set={f"{f}:{i}:{role}" for f,i in identities}
    expected={(source,layer,position,head,action) for source in source_set for layer in c4.LAYERS for position in c4.POSITIONS for head in c4.HEADS}
    actual=set(); out=[]
    for row in rows:
        if not isinstance(row,Mapping) or set(row) != {"source","layer","position","head","metrics","traffic"}: raise C6Error("malformed observation row")
        key=(row["source"],row["layer"],row["position"],row["head"],action)
        if not isinstance(row["source"],str) or row["source"] not in source_set or row["layer"] not in c4.LAYERS or row["position"] not in c4.POSITIONS or type(row["head"]) is not int or row["head"] not in c4.HEADS or key in actual: raise C6Error("observation inventory differs")
        metrics,rowtraffic=row["metrics"],row["traffic"]
        if not isinstance(metrics,Mapping) or not isinstance(rowtraffic,Mapping) or not _finite(metrics.get("cosine_similarity")) or not _finite(metrics.get("relative_l2_error")) or not _finite(rowtraffic.get("total_kv_bytes")): raise C6Error("invalid observation numeric metric")
        actual.add(key); out.append(row)
    if actual != expected: raise C6Error("observation Cartesian inventory differs")
    return out

def validate_holdout_manifest(payload: Mapping[str, Any], path: Path | None = None) -> list[dict[str, Any]]:
    """Validate the frozen, selected holdout identities before any tensor use."""
    _no_raw_text(payload)
    required={"schema_version","protocol_sha256","development_source_manifest_sha256","development_result_sha256","schedule_sha256","selected_target","selection_rule","selected_sources","rejected_sources","next_untouched_frontier","raw_text_persisted"}
    if set(payload)!=required or payload.get("schema_version")!="cascadekv-phi35-8k-c6-holdout-selection-v1" or payload.get("protocol_sha256")!=sha256_path(PROTOCOL) or payload.get("selection_rule")!="first mechanically eligible globally-new exact input SHA in ascending order" or payload.get("raw_text_persisted") is not False:
        raise C6Error("C6 holdout manifest differs")
    sources=payload.get("selected_sources")
    if not isinstance(sources,list) or len(sources)!=3: raise C6Error("C6 holdout source count differs")
    by_family={}
    for source in sources:
        if not isinstance(source, Mapping) or source.get("family") not in c4.FAMILIES or source.get("role")!="holdout" or source.get("identity_kind")!="dataset_index": raise C6Error("C6 holdout source differs")
        family=str(source["family"]); index=source.get("dataset_index")
        if type(index) is not int or index < FRONTIER[family] or family in by_family or any(source.get(k)!=c4.SOURCE_SPECS[family][k] for k in ("dataset","config","split","revision","field")):
            raise C6Error("C6 holdout identity differs")
        digest=source.get("input_ids_sha256")
        if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest): raise C6Error("C6 holdout input hash differs")
        by_family[family]=dict(source)
    if set(by_family)!=set(c4.FAMILIES) or payload.get("next_untouched_frontier",{}).keys()!=set(c4.FAMILIES): raise C6Error("C6 holdout family inventory differs")
    if path is not None and not path.is_file(): raise C6Error("C6 holdout manifest missing")
    return [by_family[family] for family in c4.FAMILIES]

def validate_observation_count(rows: Iterable[Mapping[str, Any]], method: str, *, holdout: bool=False) -> None:
    """Compatibility entry point for compact inventory fixtures; validates exact keys."""
    rows=list(rows); expected_sources={(f,i) for f in c4.FAMILIES for i in (DEVELOPMENT[f] if not holdout else ())}
    if holdout: expected_sources={(r.get("family"),r.get("dataset_index")) for r in rows}
    expected={(f,i,l,p,h,method) for f,i in expected_sources for l in c4.LAYERS for p in c4.POSITIONS for h in c4.HEADS}
    actual=set()
    for r in rows:
        key=(r.get("family"),r.get("dataset_index"),r.get("layer"),r.get("query_position"),r.get("head"),r.get("method"))
        if key in actual or key not in expected: raise C6Error("observation inventory differs")
        actual.add(key)
    if actual!=expected: raise C6Error("observation Cartesian inventory differs")

def validate_development_manifest(payload: Mapping[str, Any], path: Path | None=None) -> list[dict[str, Any]]:
    _no_raw_text(payload); required={"schema_version","protocol_sha256","target","tokenization_contract","source_specs","selected_sources","rejected_sources","global_input_ids_sha256_unique","source_text_stored"}
    if set(payload)!=required or payload.get("schema_version")!="cascadekv-phi35-8k-c6-development-sources-v1" or payload.get("protocol_sha256")!=sha256_path(PROTOCOL) or payload.get("target")!={"model":c4.MODEL,"model_revision":c4.REVISION,"tokenizer_revision":c4.REVISION} or payload.get("tokenization_contract")!={"add_special_tokens":True,"truncation":True,"max_length":8192,"required_input_ids_length":8192,"canonicalization":"CPU contiguous int64 bytes"} or payload.get("source_specs")!=c4.SOURCE_SPECS or payload.get("source_text_stored") is not False or payload.get("global_input_ids_sha256_unique") is not True: raise C6Error("C6 development manifest differs")
    sources=payload.get("selected_sources"); hashes=set(); out=[]
    if not isinstance(sources,list) or len(sources)!=15: raise C6Error("C6 requires exactly 15 development sources")
    for family in c4.FAMILIES:
        rows=[x for x in sources if isinstance(x,Mapping) and x.get("family")==family]
        if [x.get("dataset_index") for x in rows] != list(DEVELOPMENT[family]): raise C6Error("C6 development identity differs")
        for source in rows:
            if source.get("dataset_index")==14 and family=="qa" or any(source.get(k)!=c4.SOURCE_SPECS[family][k] for k in ("dataset","config","split","revision","field")): raise C6Error("forbidden/mutated C6 source")
            digest=source.get("input_ids_sha256")
            if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest) or digest in hashes: raise C6Error("C6 source hashes must be globally unique")
            hashes.add(digest); out.append(dict(source,role="development"))
    if path is not None and not path.is_file(): raise C6Error("C6 source manifest missing")
    return out

def build_development_manifest(*, output: Path) -> dict[str, Any]:
    """Kaggle-only reacquisition of the fixed 15 identities; stores no text."""
    if output.exists(): raise C6Error("refusing to overwrite development manifest")
    from cascadekv.phi35_8k_source_selection import frozen_specs,load_dataset_for_spec,load_frozen_tokenizer,mechanical_proof
    import torch
    tokenizer,specs=load_frozen_tokenizer(),frozen_specs(); selected=[]; rejected={f:[] for f in c4.FAMILIES}; seen=set()
    for family in c4.FAMILIES:
        dataset=load_dataset_for_spec(specs[family])
        for index in DEVELOPMENT[family]:
            value=dataset[index].get(specs[family]["field"]); proof=mechanical_proof(dataset,tokenizer,specs[family],index)
            if not c4._proof_eligible(proof) or not isinstance(value,str): raise C6Error("fixed development identity is no longer eligible")
            ids=torch.tensor([tokenizer(value,add_special_tokens=True,truncation=True,max_length=8192).input_ids],dtype=torch.int64).contiguous(); digest=c4._input_hash(ids)
            if digest in seen: raise C6Error("C6 development exact input duplicate")
            seen.add(digest); selected.append({"family":family,**{k:specs[family][k] for k in ("dataset","config","split","revision","field")},"identity_kind":"dataset_index","dataset_index":index,"input_ids_sha256":digest,"proof":proof})
    value={"schema_version":"cascadekv-phi35-8k-c6-development-sources-v1","protocol_sha256":sha256_path(PROTOCOL),"target":{"model":c4.MODEL,"model_revision":c4.REVISION,"tokenizer_revision":c4.REVISION},"tokenization_contract":{"add_special_tokens":True,"truncation":True,"max_length":8192,"required_input_ids_length":8192,"canonicalization":"CPU contiguous int64 bytes"},"source_specs":c4.SOURCE_SPECS,"selected_sources":selected,"rejected_sources":rejected,"global_input_ids_sha256_unique":True,"source_text_stored":False}; validate_development_manifest(value); atomic_json(output,value); return value

def select_holdout_from_inspections(inspections: Mapping[str, Iterable[Mapping[str, Any]]], development_hashes: Iterable[str], *, development_result_sha256: str, schedule_sha256: str, development_source_manifest_sha256: str) -> dict[str, Any]:
    known=set(development_hashes)
    if len(known)!=15 or any(not re.fullmatch(r"[0-9a-f]{64}",x) for x in known): raise C6Error("fifteen development hashes required")
    selected=[]; rejected={f:[] for f in c4.FAMILIES}; next_frontier={}
    for family in c4.FAMILIES:
        accepted=None; last=FRONTIER[family]-1
        for row in sorted((dict(x) for x in inspections.get(family,[])),key=lambda x:x.get("dataset_index",-1)):
            index=row.get("dataset_index");
            if not isinstance(index,int) or index<FRONTIER[family]: continue
            last=index; proof=row.get("proof",{}); digest=row.get("input_ids_sha256")
            if not c4._proof_eligible(proof): rejected[family].append({"dataset_index":index,"reason":"mechanically_ineligible","proof":proof}); continue
            if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest): raise C6Error("malformed input hash")
            if digest in known: rejected[family].append({"dataset_index":index,"reason":"duplicate_exact_input_ids_sha256","input_ids_sha256":digest,"proof":proof}); continue
            source=dict(row["source"]); _no_raw_text(source)
            if source.get("family")!=family or source.get("dataset_index")!=index: raise C6Error("holdout source identity differs")
            accepted={**source,"role":"holdout","proof":proof,"input_ids_sha256":digest}; known.add(digest); break
        if accepted is None: raise C6Error("no new eligible C6 holdout")
        selected.append(accepted); next_frontier[family]=last+1
    return {"schema_version":"cascadekv-phi35-8k-c6-holdout-selection-v1","protocol_sha256":sha256_path(PROTOCOL),"development_source_manifest_sha256":development_source_manifest_sha256,"development_result_sha256":development_result_sha256,"schedule_sha256":schedule_sha256,"selected_target":None,"selection_rule":"first mechanically eligible globally-new exact input SHA in ascending order","selected_sources":selected,"rejected_sources":rejected,"next_untouched_frontier":next_frontier,"raw_text_persisted":False}

class CaptureResolver:
    """C6 manifest-first tensor resolver; validates before payload opening."""
    def __init__(self, root: Path, manifest: Path, kind: str, *, development_source_manifest: Path|None=None, holdout_manifest: Path|None=None, development_sha: str|None=None, schedule_sha: str|None=None):
        self.root=root.resolve(); self.kind=kind; payload=_json(manifest); _no_raw_text(payload); expected_count=75 if kind=="development" else 15; expected_forwards=15 if kind=="development" else 3
        required={"schema_version","kind","protocol_sha256","qualification_sha256","backend_id","development_source_manifest_sha256","holdout_manifest_sha256","development_result_sha256","schedule_sha256","artifact_count","forwards","artifacts"}
        if set(payload)!=required or payload.get("schema_version")!="cascadekv-phi35-8k-c6-capture-v1" or payload.get("kind")!=kind or payload.get("protocol_sha256")!=sha256_path(PROTOCOL) or payload.get("qualification_sha256")!=c4.QUALIFICATION_SHA or payload.get("backend_id")!=c4.BACKEND_ID or payload.get("artifact_count")!=expected_count or payload.get("forwards")!=expected_forwards: raise C6Error("C6 capture manifest differs")
        sources: list[dict[str, Any]]
        holdout = None
        if kind=="development":
            if development_source_manifest is None or not development_source_manifest.is_file() or payload.get("development_source_manifest_sha256") != sha256_path(development_source_manifest) or any(payload.get(x) is not None for x in ("holdout_manifest_sha256","development_result_sha256","schedule_sha256")): raise C6Error("development capture binding differs")
            sources=validate_development_manifest(_json(development_source_manifest), development_source_manifest)
        else:
            if holdout_manifest is None or not holdout_manifest.is_file(): raise C6Error("actual holdout manifest path is required")
            holdout=_json(holdout_manifest)
            if payload.get("development_source_manifest_sha256") is not None or payload.get("holdout_manifest_sha256") != sha256_path(holdout_manifest) or payload.get("development_result_sha256") != development_sha or payload.get("schedule_sha256") != schedule_sha: raise C6Error("holdout capture binding differs")
            sources=validate_holdout_manifest(holdout, holdout_manifest)
        expected=_expected_tensor_ids(kind,holdout); records={}
        expected_sources={_source_identity(source, kind): source for source in sources}
        for row in payload.get("artifacts",[]):
            if not isinstance(row,Mapping) or set(row)!={"identity","layer","artifact_relative_path","artifact_sha256","provenance_relative_path","provenance_sha256","input_ids_sha256","source"}: raise C6Error("capture record schema differs")
            for field in ("artifact_relative_path","provenance_relative_path"):
                rel=row.get(field)
                if not isinstance(rel,str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts: raise C6Error("unsafe capture path")
            key=f"{row['identity']}:L{row['layer']}"
            if key not in expected or key in records or not all(isinstance(row[x],str) and re.fullmatch(r"[0-9a-f]{64}",row[x]) for x in ("artifact_sha256","provenance_sha256","input_ids_sha256")): raise C6Error("capture inventory differs")
            source=expected_sources.get(row["identity"])
            if source is None or row.get("source")!={"family":source["family"],"dataset_index":source["dataset_index"],"role":kind} or row["input_ids_sha256"]!=source["input_ids_sha256"]: raise C6Error("capture source binding differs")
            for field in ("artifact_relative_path", "provenance_relative_path"):
                path=(self.root/str(row[field])).resolve()
                if not path.is_relative_to(self.root) or path.is_symlink() or not path.is_file(): raise C6Error("capture artifact missing or escapes root")
                digest_field="artifact_sha256" if field.startswith("artifact") else "provenance_sha256"
                if sha256_path(path)!=row[digest_field]: raise C6Error("capture artifact digest differs")
            detail=_json((self.root/str(row["provenance_relative_path"])).resolve()); _no_raw_text(detail)
            if detail.get("protocol_sha256")!=sha256_path(PROTOCOL) or detail.get("identity")!=row["identity"] or detail.get("layer")!=row["layer"] or detail.get("input_ids_sha256")!=row["input_ids_sha256"] or detail.get("artifact_sha256")!=row["artifact_sha256"]: raise C6Error("capture provenance differs")
            records[key]=dict(row)
        if sorted(records)!=expected: raise C6Error("capture inventory incomplete")
        self.records=records
    def ordered(self): return [self.records[x] for x in sorted(self.records)]

class TensorAccess:
    def __init__(self,resolver:CaptureResolver,*,allowed_kind:str):
        if resolver.kind!=allowed_kind: raise C6Error("tensor firewall denied capture kind")
        self.resolver=resolver; self.allowed_kind=allowed_kind; self.opened=[]
    def open(self,record:Mapping[str,Any]):
        root=self.resolver.root
        def safe(rel):
            path=(root/rel).resolve()
            if not path.is_relative_to(root) or path.is_symlink(): raise C6Error("artifact path escapes root")
            return path
        artifact,provenance=safe(str(record["artifact_relative_path"])),safe(str(record["provenance_relative_path"]))
        if not artifact.is_file() or not provenance.is_file() or sha256_path(artifact)!=record["artifact_sha256"] or sha256_path(provenance)!=record["provenance_sha256"]: raise C6Error("artifact digest differs")
        detail=_json(provenance); _no_raw_text(detail)
        if detail.get("protocol_sha256")!=sha256_path(PROTOCOL) or detail.get("identity")!=record["identity"] or detail.get("layer")!=record["layer"] or detail.get("input_ids_sha256")!=record["input_ids_sha256"] or detail.get("artifact_sha256")!=record["artifact_sha256"]: raise C6Error("artifact provenance differs")
        from safetensors import safe_open
        import torch
        with safe_open(str(artifact),framework="pt",device="cpu") as h:
            if set(h.keys())!={"q","k","v"}: raise C6Error("tensor keys differ")
            q,k,v=(h.get_tensor(x) for x in ("q","k","v"))
        if any(x.dtype!=torch.float16 or tuple(x.shape)!=c4.SHAPE for x in (q,k,v)): raise C6Error("tensor shape/dtype differs")
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
        raise C6Error("schedule table is missing or contains extra cells")
    validate_schedule_actions(schedule_table)
    selected=[]; seen=set()
    for action in ACTIONS:
        indexed={}
        for row in method_rows.get(action, []):
            key=(row.get("source"), row.get("layer"), row.get("position"), row.get("head"))
            if key in indexed: raise C6Error("duplicate action observation")
            indexed[key]=row
        if len(indexed) != 7200: raise C6Error("action observation inventory differs")
        for key,row in indexed.items():
            if f"{key[1]}:{key[3]}" in expected_cells and schedule_table[f"{key[1]}:{key[3]}"] == action:
                if key in seen: raise C6Error("duplicate scheduled observation")
                seen.add(key); selected.append(row)
    if len(selected) != 7200: raise C6Error("scheduled observations missing")
    return sorted(selected, key=lambda r:(str(r["source"]),int(r["layer"]),int(r["position"]),int(r["head"])))

def _development_schedules(rows, protocol):
    stats={}; groups=[]
    for layer in c4.LAYERS:
        for head in c4.HEADS:
            group={}
            for action in ACTIONS:
                values=[x for x in rows[action] if x["layer"]==layer and x["head"]==head]
                if len(values)!=45: raise C6Error("each C6 cell requires 45 observations")
                group[action]={"relative_l2":mean(x["metrics"]["relative_l2_error"] for x in values),"cosine":mean(x["metrics"]["cosine_similarity"] for x in values),"total_kv_bytes":mean(x["traffic"]["total_kv_bytes"] for x in values)}; stats[f"{layer}:{head}:{action}"]=group[action]
            groups.append(group)
    dense,flat5=_summary(rows["dense"])["mean_total_kv_bytes"],_summary(rows["flat5"])["mean_total_kv_bytes"]; schedules={}
    for target in TARGETS:
        target_mean,strict,cap=_target_cap(protocol,target,dense,flat5); found=optimize_action_cells(groups, cap, domains=action_domain_by_cell())
        if found is None: schedules[target]={"feasible":False,"target_mean_kv_bytes":target_mean,"strict":strict,"integer_microbyte_cap":cap,"integer_microbytes_used":None,"table":None}; continue
        used,_error,_cos,actions=found; table={f"{l}:{h}":a for (l,h),a in zip(((l,h) for l in c4.LAYERS for h in c4.HEADS),actions,strict=True)}
        schedules[target]={"feasible":True,"target_mean_kv_bytes":target_mean,"strict":strict,"integer_microbyte_cap":cap,"integer_microbytes_used":used,"table":table}
    return stats,schedules

def develop(*, capture_root:Path,capture_manifest:Path,development_source_manifest:Path,qualification:Path,output:Path) -> dict[str,Any]:
    """Evaluate all A0--A9 once and write the final immutable development result once."""
    if output.exists(): raise C6Error("refusing to overwrite development result")
    protocol=validate_protocol(); sources=validate_development_manifest(_json(development_source_manifest),development_source_manifest); c4._validate_c2_qualification(qualification)
    resolver=CaptureResolver(capture_root,capture_manifest,"development",development_source_manifest=development_source_manifest); access=TensorAccess(resolver,allowed_kind="development"); rows=_method_rows(access,protocol); expected=_expected_tensor_ids("development")
    if access.opened!=expected: raise C6Error("development tensor firewall differs")
    for action in ACTIONS: validate_observation_inventory(rows[action], action=action, role="development")
    summaries={x:_summary(value) for x,value in rows.items()}
    if any(x["observation_count"]!=7200 for x in summaries.values()): raise C6Error("C6 development observation count differs")
    stats,schedules=_development_schedules(rows,protocol)
    result={"schema_version":"cascadekv-phi35-8k-c6-development-v1","protocol_sha256":sha256_path(PROTOCOL),"runtime_manifest_sha256":sha256_path(RUNTIME),"qualification_sha256":sha256_path(qualification),"development_source_manifest_sha256":sha256_path(development_source_manifest),"capture_manifest_sha256":sha256_path(capture_manifest),"opened_development_tensor_identities":expected,"opened_holdout_tensor_identities":[],"methods":protocol["methods"],"development_baseline_metrics":{x:summaries[x] for x in c4.BASELINES},"development_action_metrics":{x:summaries[x] for x in ACTIONS},"action_cell_statistics":stats,"schedules":schedules,"development_target_metrics":{},"target_feasibility":{x:schedules[x]["feasible"] for x in TARGETS},"optimizer":protocol["optimizer"]}
    schedule_rows={}
    for target in TARGETS:
        if not schedules[target]["feasible"]: schedule_rows[target]={}; continue
        table=schedules[target]["table"]
        schedule_rows[target]=_summary(rows_for_schedule(rows,table))
    result["development_target_metrics"]=schedule_rows; result["schedule_sha256"]=schedule_digest(schedules); result=freeze_development(result); atomic_json(output,result); return result

def prospective_test(*, holdout_capture_root:Path,holdout_capture_manifest:Path,holdout_manifest:Path,development_result:Path,output:Path) -> dict[str,Any]:
    """One-shot holdout evaluator.  Its signature makes development tensors unavailable."""
    if output.exists(): raise C6Error("refusing to overwrite prospective result")
    dev=_json(development_result); dev_sha=sha256_path(development_result); selected=assert_holdout_may_start(dev,dev_sha,development_result); holdout=_json(holdout_manifest); holdout_sources=validate_holdout_manifest(holdout, holdout_manifest)
    if holdout.get("development_result_sha256")!=dev_sha or holdout.get("schedule_sha256")!=dev["schedule_sha256"] or holdout.get("selected_target")!=selected: raise C6Error("holdout is not bound to frozen development")
    resolver=CaptureResolver(holdout_capture_root,holdout_capture_manifest,"holdout",holdout_manifest=holdout_manifest,development_sha=dev_sha,schedule_sha=dev["schedule_sha256"]); access=TensorAccess(resolver,allowed_kind="holdout"); schedules={x:dev["schedules"][x]["table"] for x in TARGETS if dev["schedules"][x]["feasible"]}; rows=_method_rows(access,validate_protocol(),schedules); expected=_expected_tensor_ids("holdout",holdout)
    if access.opened!=expected: raise C6Error("holdout tensor firewall differs")
    expected_holdout_sources=[_source_identity(source,"holdout") for source in holdout_sources]
    for action in ACTIONS: validate_observation_inventory(rows[action], action=action, role="holdout", expected_holdout_sources=expected_holdout_sources)
    summaries={x:_summary(v) for x,v in rows.items()}
    if any(x["observation_count"]!=1440 for x in summaries.values()): raise C6Error("holdout observation count differs")
    protocol=validate_protocol(); quality=protocol["quality_gates"]; gates={}; first=None
    for target in TARGETS:
        if target not in schedules: gates[target]={"feasible":False,"passes":False}; continue
        m=summaries[target]; uniform=quality["winner_relative_l2_lt"]; flat=quality["winner_modeled_kv_traffic_lt"]; row={"feasible":True,"mean_cosine_gte":m["mean_cosine"]>=quality["mean_cosine_gte"],"mean_relative_l2_lte":m["mean_relative_l2"]<=quality["mean_relative_l2_lte"],"relative_l2_lt_uniform10":m["mean_relative_l2"]<summaries[uniform]["mean_relative_l2"],"modeled_kv_traffic_lt_flat5":m["mean_total_kv_bytes"]<summaries[flat]["mean_total_kv_bytes"]}; row["passes"]=all(v for k,v in row.items() if k not in {"feasible","passes"}); gates[target]=row
        if first is None and row["passes"]: first=target
    before=dev["schedule_sha256"]; after=schedule_digest(dev["schedules"])
    audit=audit_prospective_test(opened_development=[],opened_holdout=expected,optimizer_rerun=False,schedule_before=before,schedule_after=after)
    result={"schema_version":"cascadekv-phi35-8k-c6-test-v1","protocol_sha256":sha256_path(PROTOCOL),"runtime_manifest_sha256":sha256_path(RUNTIME),"development_result_sha256":dev_sha,"holdout_manifest_sha256":sha256_path(holdout_manifest),"capture_manifest_sha256":sha256_path(holdout_capture_manifest),"selected_target":selected,"development_passing_targets":dev["development_passing_targets"],"baseline_test_metrics":{x:summaries[x] for x in c4.BASELINES},"target_test_metrics":{x:summaries[x] for x in TARGETS if x in summaries},"per_gate_pass_fail":gates,"first_passing_target_descriptive":first,"firewall_evidence":audit,"classification":prospective_verdict(selected,gates)}; atomic_json(output,result); return result

def capture_sources(*, kind:str, sources:list[dict[str,Any]], output:Path, qualification:Path, development_source_sha:str|None=None, holdout_manifest:Path|None=None, development_sha:str|None=None, schedule_sha:str|None=None) -> dict[str,Any]:
    """C6 capture: exactly one pinned model forward and five artifacts per source."""
    if output.exists() and (output/"final_capture_manifest.json").exists(): raise C6Error("refusing to overwrite completed capture")
    qrecord=c4._validate_c2_qualification(qualification)
    from cascadekv import phi35_kaggle_c2 as c2
    from cascadekv.phi35_8k_source_selection import load_dataset_for_spec,load_frozen_tokenizer,mechanical_proof
    import torch
    tokenizer=load_frozen_tokenizer(); model=c2._load_pinned_model(); records=[]
    try:
        c2.validate_qualification(qrecord); c2.assert_capture_backend(model,qrecord)
        for source in sources:
            spec={x:source[x] for x in ("dataset","config","split","revision","field")}; dataset=load_dataset_for_spec(spec); proof=mechanical_proof(dataset,tokenizer,spec,int(source["dataset_index"])); value=dataset[int(source["dataset_index"])].get(source["field"])
            if not c4._proof_eligible(proof) or not isinstance(value,str): raise C6Error("captured source no longer eligible")
            ids=torch.tensor([tokenizer(value,add_special_tokens=True,truncation=True,max_length=8192).input_ids],dtype=torch.int64).contiguous(); digest=c4._input_hash(ids)
            if digest!=source["input_ids_sha256"]: raise C6Error("captured input hash reproof differs")
            identity=_source_identity(source,kind); captured=c2.capture_five_layers_post_rope_qkv(model,ids.to(c2._first_device(model)))
            for layer in c4.LAYERS:
                stem=f"{source['family']}_{kind}_index{source['dataset_index']}_layer{layer}"; artifact_rel=f"artifacts/{stem}.safetensors"; provenance_rel=f"artifacts/{stem}.provenance.json"; artifact=output/artifact_rel; provenance=output/provenance_rel
                if artifact.exists() or provenance.exists(): raise C6Error("never overwrite C6 artifact")
                detail={"schema_version":"cascadekv-phi35-8k-c6-artifact-v1","protocol_sha256":sha256_path(PROTOCOL),"qualification_sha256":c4.QUALIFICATION_SHA,"backend_id":c4.BACKEND_ID,"identity":identity,"layer":layer,"input_ids_sha256":digest,"target":{"model":c4.MODEL,"revision":c4.REVISION,"tokenizer_revision":c4.REVISION},"source":{**{x:source[x] for x in ("family","dataset","config","split","revision","field","identity_kind","dataset_index")},"role":kind},"source_proof":proof,"q_shape":list(c4.SHAPE),"k_shape":list(c4.SHAPE),"v_shape":list(c4.SHAPE),"storage_dtype":"float16","model_compute_dtype":"float16","attention_implementation":"sdpa","use_cache":False,"quantization":"none","capture_adapter":"phi3-post-rope-qkv-v1"}
                # The canonical C2 writer writes the artifact once, computes its
                # final digest, and atomically writes the matching provenance.
                written=c2.write_artifact(artifact,provenance,captured[layer],detail)
                artifact_sha=sha256_path(artifact)
                if written.get("artifact_sha256") != artifact_sha or _json(provenance) != written:
                    raise C6Error("capture writer provenance chain differs")
                records.append({"identity":identity,"layer":layer,"artifact_relative_path":artifact_rel,"artifact_sha256":artifact_sha,"provenance_relative_path":provenance_rel,"provenance_sha256":sha256_path(provenance),"input_ids_sha256":digest,"source":{"family":source["family"],"dataset_index":source["dataset_index"],"role":kind}})
        final={"schema_version":"cascadekv-phi35-8k-c6-capture-v1","kind":kind,"protocol_sha256":sha256_path(PROTOCOL),"qualification_sha256":c4.QUALIFICATION_SHA,"backend_id":c4.BACKEND_ID,"development_source_manifest_sha256":development_source_sha if kind=="development" else None,"holdout_manifest_sha256":sha256_path(holdout_manifest) if holdout_manifest else None,"development_result_sha256":development_sha,"schedule_sha256":schedule_sha,"artifact_count":len(records),"forwards":len(sources),"artifacts":records}; atomic_json(output/"final_capture_manifest.json",final); return final
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
