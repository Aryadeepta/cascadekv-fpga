"""C7 local pre-freeze executor; imports and preflight are data/model free."""
from __future__ import annotations
import hashlib,json,math,re,subprocess,tarfile
from pathlib import Path
from statistics import mean
from typing import Any,Iterable,Mapping
from types import MappingProxyType
from . import phi35_8k_c4 as c4
from . import phi35_8k_c6 as c6

ROOT=Path(__file__).resolve().parents[1]; PROTOCOL=ROOT/"configs/cascadekv_phi35_8k_c7_protocol.json"; RUNTIME=ROOT/"configs/cascadekv_phi35_8k_c7_runtime_manifest.json"
ARCHIVE=ROOT/"artifacts/c6-v1/cascadekv-c6-result.tar.gz"; C7_TAG="cascadekv-phi35-8k-c7-freeze-v2"; RAW_TEXT_SENTINEL="SHOULD_NOT_PERSIST_RAW_SOURCE_TEXT_C7_6B29"
C6_PARENT={"result_tag":"cascadekv-phi35-8k-c6-result-v1","result_commit":"de7fdda55ae567442d26c525b974a5e32649b8f1","freeze_tag":"cascadekv-phi35-8k-c6-freeze-v1","freeze_commit":"f9dfdfffb5f2922a471f2bb955645ed52dda096c","protocol_sha256":"78b7b61e0cf4f3304496a9277e6f2fd966c93403f56349d235bf3ecf0fdc615a","runtime_sha256":"c19665dae088682758880a0b9dd8635b08de89339835bc737a13c5538f250cf2","original_kaggle_tarball_sha256":"4e95dfde98e1e35a61dd08dffed9e7ef93ca466791485f9068392d1b2bded685","repository_stdout_sha256":"c8efdf27958af3de3c31f68456bedc7debb6ed9b067e273e155519097c391af4","result_archive_sha256":"f63d8e563462c0876836d7858a00435daf55fd0f48542e7a0944fa8092daf587","postmortem_archive_sha256":"39160d77722e0af5dd68f27d234f420db35b68798420418f6f477dc4672fd417","postmortem_markdown_sha256":"d5308375e08eae376d0fc075855be394261f778a40641d8cb83ffe8a397a11eb"}
DEVELOPMENT={"narrative":(13,14,15,16,17,18),"report":(16,17,18,19,21,22),"qa":(13,15,16,17,18,19)}; FRONTIER={"narrative":19,"qa":20,"report":23}
ACTIONS,BASE_ACTIONS,TARGETS=c6.ACTIONS,c6.BASE_ACTIONS,c6.TARGETS
canonical_json,sha256_path,atomic_json=c6.canonical_json,c6.sha256_path,c6.atomic_json
action_definitions,allowed_actions,action_domain_by_cell=c6.action_definitions,c6.allowed_actions,c6.action_domain_by_cell
validate_schedule,validate_schedule_actions=c6.validate_schedule,c6.validate_schedule_actions
class C7Error(RuntimeError):pass
def _json(p):
 try:x=json.loads(Path(p).read_text())
 except (OSError,json.JSONDecodeError) as e:raise C7Error("cannot read JSON") from e
 if not isinstance(x,dict):raise C7Error("JSON root must be object")
 return x
def _git(ref):
 try:return subprocess.check_output(["git","-C",str(ROOT),"rev-parse",f"{ref}^{{commit}}"],text=True,stderr=subprocess.DEVNULL).strip()
 except (OSError,subprocess.CalledProcessError) as e:raise C7Error("required git reference unavailable") from e
def _finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))
def _no_raw(x):
 if RAW_TEXT_SENTINEL in canonical_json(x):raise C7Error("raw text persisted")
 c4._no_raw_text(x)
def _freeze(x):
 if isinstance(x,Mapping):return MappingProxyType({k:_freeze(v) for k,v in x.items()})
 if isinstance(x,list):return tuple(_freeze(v) for v in x)
 return x

def validate_protocol(payload=None):
 p=dict(_json(PROTOCOL) if payload is None else payload);req={"schema_version","c6_result_parent","geometry","methods","development_sources","prospective_frontier","targets","quality_gates","optimizer","action_validity","lifecycle","raw_text_persisted"}
 if set(p)!=req or p.get("schema_version")!="cascadekv-phi35-8k-c7-protocol-v1":raise C7Error("C7 protocol schema differs")
 if p["c6_result_parent"]!=C6_PARENT or p["methods"]["actions"]!=action_definitions() or list(p["methods"]["actions"])!=list(ACTIONS):raise C7Error("C6 parent/action contract differs")
 g=p["geometry"]
 if any(g.get(k)!=c4.validate_protocol()["geometry"].get(k) for k in ("context_length","layers","query_positions","heads","head_dim","schedule_cells")) or g.get("observations_per_source_per_method")!=480 or g.get("development_observations_per_method")!=8640 or g.get("family_observations_per_cell_per_method")!=18:raise C7Error("geometry differs")
 if p["development_sources"]!={"identities":{f:list(DEVELOPMENT[f]) for f in c4.FAMILIES},"report20_excluded":True,"source_text_stored":False,"archived_consumed_provenance_only":True} or p["prospective_frontier"]!={"start_indices":FRONTIER,"untouched_during_preparation":True} or p["targets"]!=c4.validate_protocol()["targets"] or p["quality_gates"]!=c4.validate_protocol()["quality_gates"]:raise C7Error("C7 scientific contract differs")
 if p["optimizer"]!={"version":"c7-robust-max-family-integer-microbyte-dp-v1","scope":"development_only","objective":"summed_max_family_mean_final_attention_output_relative_l2","family_aggregation":"max of narrative/report/qa means; exactly 6 sources x 3 positions each","tie_breaking":"robust relative-L2 ascending, global cosine descending, traffic ascending, frozen action-table ordering"}:raise C7Error("robust objective differs")
 if p["action_validity"]!={"layer_0":list(ACTIONS),"nonzero_sampled_layers":list(BASE_ACTIONS),"enforcement":"DP enumeration domain; independent schedule validation"} or p["lifecycle"]!={"development_no_go":"C7-DEVELOPMENT-NO-GO","development_go":"C7-DEVELOPMENT-GO","freeze_before_prospective":True,"selected_target_locks_prospective_verdict":True} or p["raw_text_persisted"] is not False:raise C7Error("C7 safety contract differs")
 return p
def verify_runtime_closure():
 m=_json(RUNTIME)
 if set(m)!={"schema_version","purpose","protocol","bound_files"} or m.get("schema_version")!=1 or not isinstance(m.get("bound_files"),list) or m["protocol"]!={"path":str(PROTOCOL.relative_to(ROOT)),"sha256":sha256_path(PROTOCOL)}:raise C7Error("runtime manifest differs")
 out=[];seen=set()
 for r in m["bound_files"]:
  if not isinstance(r,Mapping) or set(r)!={"path","sha256"} or not isinstance(r["path"],str) or r["path"]==m["protocol"]["path"] or r["path"] in seen:raise C7Error("unsafe runtime entry")
  rel=Path(r["path"]);raw=ROOT/rel;p=raw.resolve()
  if rel.is_absolute() or ".." in rel.parts or not re.fullmatch(r"[0-9a-f]{64}",str(r["sha256"])) or not p.is_relative_to(ROOT.resolve()) or raw.is_symlink() or not p.is_file() or sha256_path(p)!=r["sha256"]:raise C7Error("runtime closure mismatch")
  seen.add(r["path"]);out.append(dict(r))
 return tuple(out)
def preflight(*,execution=False):
 if _git(C6_PARENT["result_tag"])!=C6_PARENT["result_commit"] or _git(C6_PARENT["freeze_tag"])!=C6_PARENT["freeze_commit"]:raise C7Error("C6 parent tag differs")
 if execution and _git(C7_TAG)!=_git("HEAD"):raise C7Error("C7 execution requires matching freeze tag")
 for p,k in ((ROOT/"results/archive/cascadekv_phi35_8k_c6_v1_result.json","result_archive_sha256"),(ROOT/"results/archive/cascadekv_phi35_8k_c6_v1_postmortem.json","postmortem_archive_sha256"),(ROOT/"docs/archive/cascadekv_phi35_8k_c6_v1_postmortem.md","postmortem_markdown_sha256")):
  if not p.is_file() or sha256_path(p)!=C6_PARENT[k]:raise C7Error("C6 parent archive differs")
 if not ARCHIVE.is_file() or sha256_path(ARCHIVE)!=C6_PARENT["original_kaggle_tarball_sha256"]:raise C7Error("original C6 tar missing or differs")
 stdout=ROOT/"artifacts/c6-v1/evidence/c6_v1_kaggle_stdout.md"
 if not stdout.is_file() or sha256_path(stdout)!=C6_PARENT["repository_stdout_sha256"]:raise C7Error("repository stdout missing or differs")
 validate_protocol();return {"local_only":not execution,"protocol_sha256":sha256_path(PROTOCOL),"runtime_manifest_sha256":sha256_path(RUNTIME),"bound_files":verify_runtime_closure()}

def _archived(name):
 with tarfile.open(ARCHIVE,"r:gz") as t:
  x=t.extractfile(name)
  if x is None:raise C7Error("archived C6 member missing")
  return json.loads(x.read())
def archived_development_sources():
 old=_archived("development_sources.json")["selected_sources"];hold=_archived("holdout.json")["selected_sources"];by={(x["family"],x["dataset_index"]):x for x in old+hold};out=[]
 for f in c4.FAMILIES:
  for i in DEVELOPMENT[f]:
   if (f,i) not in by:raise C7Error("archived source missing")
   out.append(dict(by[f,i],role="development"))
 return out
def archived_development_hashes():
 """The immutable C6 tarball, not a C7 manifest, is the hash authority."""
 rows=archived_development_sources(); mapping={(x["family"],x["dataset_index"]):x["input_ids_sha256"] for x in rows}
 if set(mapping)!={(f,i) for f in c4.FAMILIES for i in DEVELOPMENT[f]} or len(set(mapping.values()))!=18:raise C7Error("archived development hash authority differs")
 return mapping
def validate_development_manifest(payload,path=None):
 _no_raw(payload);req={"schema_version","protocol_sha256","target","tokenization_contract","source_specs","selected_sources","archived_c6_result_sha256","global_input_ids_sha256_unique","source_text_stored"}
 if set(payload)!=req or payload.get("schema_version")!="cascadekv-phi35-8k-c7-development-sources-v1" or payload.get("protocol_sha256")!=sha256_path(PROTOCOL) or payload.get("target")!={"model":c4.MODEL,"model_revision":c4.REVISION,"tokenizer_revision":c4.REVISION} or payload.get("tokenization_contract")!={"add_special_tokens":True,"truncation":True,"max_length":8192,"required_input_ids_length":8192,"canonicalization":"CPU contiguous int64 bytes"} or payload.get("source_specs")!=c4.SOURCE_SPECS or payload.get("archived_c6_result_sha256")!=C6_PARENT["result_archive_sha256"] or payload.get("global_input_ids_sha256_unique") is not True or payload.get("source_text_stored") is not False:raise C7Error("development manifest differs")
 src=payload["selected_sources"];archived=archived_development_hashes()
 if not isinstance(src,list) or len(src)!=18:raise C7Error("C7 needs 18 sources")
 seen=set();out=[]
 for f in c4.FAMILIES:
  rows=[x for x in src if isinstance(x,Mapping) and x.get("family")==f]
  if [x.get("dataset_index") for x in rows]!=list(DEVELOPMENT[f]):raise C7Error("source identity differs")
  for x in rows:
   h=x.get("input_ids_sha256")
   if x.get("role") not in {None,"development"} or x.get("identity_kind")!="dataset_index" or any(x.get(k)!=c4.SOURCE_SPECS[f][k] for k in ("dataset","config","split","revision","field")) or not isinstance(h,str) or not re.fullmatch(r"[0-9a-f]{64}",h) or h!=archived[(f,x["dataset_index"])] or h in seen:raise C7Error("source metadata/hash differs")
   seen.add(h);out.append(dict(x,role="development"))
 if path is not None and not Path(path).is_file():raise C7Error("manifest missing")
 return out
def build_development_manifest(*,output):
 if Path(output).exists() or Path(output).is_symlink():raise C7Error("refusing to overwrite development source evidence")
 x={"schema_version":"cascadekv-phi35-8k-c7-development-sources-v1","protocol_sha256":sha256_path(PROTOCOL),"target":{"model":c4.MODEL,"model_revision":c4.REVISION,"tokenizer_revision":c4.REVISION},"tokenization_contract":{"add_special_tokens":True,"truncation":True,"max_length":8192,"required_input_ids_length":8192,"canonicalization":"CPU contiguous int64 bytes"},"source_specs":c4.SOURCE_SPECS,"selected_sources":archived_development_sources(),"archived_c6_result_sha256":C6_PARENT["result_archive_sha256"],"global_input_ids_sha256_unique":True,"source_text_stored":False};validate_development_manifest(x);atomic_json(Path(output),x);return x

def validate_observation_inventory(rows,*,action,role="development",expected_holdout_sources=None):
 if action not in ACTIONS or role not in {"development","holdout"}:raise C7Error("invalid inventory contract")
 sources={f"{f}:{i}:{role}" for f in c4.FAMILIES for i in DEVELOPMENT[f]} if role=="development" else set(expected_holdout_sources or ())
 if role=="holdout" and (len(sources)!=3 or any(not re.fullmatch(r"(?:narrative|report|qa):[0-9]+:holdout",x) for x in sources)):raise C7Error("holdout source inventory differs")
 expected={(s,l,p,h) for s in sources for l in c4.LAYERS for p in c4.POSITIONS for h in c4.HEADS};actual=set();out=[]
 for r in rows:
  if not isinstance(r,Mapping) or set(r)!={"source","layer","position","head","metrics","traffic"}:raise C7Error("malformed observation")
  k=(r["source"],r["layer"],r["position"],r["head"]);m,t=r["metrics"],r["traffic"]
  if k in actual or k not in expected or not isinstance(m,Mapping) or not isinstance(t,Mapping) or not all(_finite(x) for x in (m.get("relative_l2_error"),m.get("cosine_similarity"),t.get("total_kv_bytes"))):raise C7Error("observation inventory differs")
  actual.add(k);out.append(r)
 if actual!=expected:raise C7Error("observation Cartesian inventory differs")
 return out
def family_cell_statistics(rows_by_action):
 out={}
 for a in ACTIONS:
  rows=validate_observation_inventory(rows_by_action[a],action=a)
  for l in c4.LAYERS:
   for h in c4.HEADS:
    vals=[x for x in rows if x["layer"]==l and x["head"]==h];fam={}
    for f in c4.FAMILIES:
     ids={f"{f}:{i}:development" for i in DEVELOPMENT[f]};v=[x for x in vals if x["source"] in ids]
     if len(v)!=18:raise C7Error("family cell count differs")
     fam[f]=mean(float(x["metrics"]["relative_l2_error"]) for x in v)
    if len(vals)!=54:raise C7Error("cell count differs")
    out[f"{l}:{h}:{a}"]={"narrative_mean_relative_l2":fam["narrative"],"report_mean_relative_l2":fam["report"],"qa_mean_relative_l2":fam["qa"],"robust_relative_l2":max(fam.values()),"global_relative_l2":mean(float(x["metrics"]["relative_l2_error"]) for x in vals),"global_cosine":mean(float(x["metrics"]["cosine_similarity"]) for x in vals),"global_total_kv_bytes":mean(float(x["traffic"]["total_kv_bytes"]) for x in vals),"family_observation_count":18,"global_observation_count":54}
 return out
def robust_groups(stats):return [{a:{"relative_l2":stats[f"{l}:{h}:{a}"]["robust_relative_l2"],"robust_relative_l2":stats[f"{l}:{h}:{a}"]["robust_relative_l2"],"cosine":stats[f"{l}:{h}:{a}"]["global_cosine"],"total_kv_bytes":stats[f"{l}:{h}:{a}"]["global_total_kv_bytes"]} for a in ACTIONS} for l in c4.LAYERS for h in c4.HEADS]
def optimize_action_cells(groups,cap,actions=ACTIONS,domains=None):
 # Exact C6 DP: only the supplied primary field is replaced by robust loss.
 projected=[{a:{"relative_l2":x.get("robust_relative_l2",x.get("relative_l2")),"cosine":x.get("cosine"),"total_kv_bytes":x.get("total_kv_bytes")} for a,x in row.items()} for row in groups]
 try:return c6.optimize_action_cells(projected,cap,actions,domains)
 except c6.C6Error as e:raise C7Error(str(e)) from e
def _summary(rows):return {"observation_count":len(rows),"mean_cosine":mean(float(x["metrics"]["cosine_similarity"]) for x in rows),"mean_relative_l2":mean(float(x["metrics"]["relative_l2_error"]) for x in rows),"mean_total_kv_bytes":mean(float(x["traffic"]["total_kv_bytes"]) for x in rows)}
def rows_for_schedule(rows,table):
 if set(table)!={f"{l}:{h}" for l in c4.LAYERS for h in c4.HEADS}:raise C7Error("schedule table differs")
 validate_schedule_actions(table);out=[];seen=set()
 for a in ACTIONS:
  ind={(r["source"],r["layer"],r["position"],r["head"]):r for r in rows[a]}
  if len(ind)!=8640:raise C7Error("action inventory differs")
  for k,r in ind.items():
   if table[f"{k[1]}:{k[3]}"]==a:
    if k in seen:raise C7Error("duplicate schedule row")
    seen.add(k);out.append(r)
 if len(out)!=8640:raise C7Error("scheduled observation missing")
 return sorted(out,key=lambda r:(r["source"],r["layer"],r["position"],r["head"]))
def _target_cap(protocol,target,dense,flat5):return c6._target_cap(protocol,target,dense,flat5)
def _development_schedules(rows):
 stats=family_cell_statistics({a:rows[a] for a in ACTIONS});groups=robust_groups(stats);dense=_summary(rows["dense"])["mean_total_kv_bytes"];flat5=_summary(rows["flat5"])["mean_total_kv_bytes"];out={}
 for t in TARGETS:
  tm,strict,cap=_target_cap(validate_protocol(),t,dense,flat5);found=optimize_action_cells(groups,cap,domains=action_domain_by_cell())
  if found is None:out[t]={"feasible":False,"target_mean_kv_bytes":tm,"strict":strict,"integer_microbyte_cap":cap,"integer_microbytes_used":None,"table":None};continue
  used,loss,cos,actions=found;out[t]={"feasible":True,"target_mean_kv_bytes":tm,"strict":strict,"integer_microbyte_cap":cap,"integer_microbytes_used":used,"robust_objective_value":loss,"summed_global_cosine":cos,"table":{f"{l}:{h}":a for (l,h),a in zip(((l,h) for l in c4.LAYERS for h in c4.HEADS),actions,strict=True)}}
 return stats,out
def schedule_digest(schedules):
 # Keep C6 canonical schedule digest semantics; robust diagnostics are not
 # schedule identity fields.
 stripped={t:{k:v for k,v in schedules[t].items() if k in {"feasible","target_mean_kv_bytes","strict","integer_microbyte_cap","integer_microbytes_used","table"}} for t in TARGETS}
 return c6.schedule_digest(stripped)
def development_passing_targets(result):
 p=validate_protocol();b=result.get("development_baseline_metrics",{});out=[]
 for t in TARGETS:
  m=result.get("development_target_metrics",{}).get(t,{});s=result.get("schedules",{}).get(t,{})
  if s.get("feasible") and m.get("mean_cosine",-math.inf)>=p["quality_gates"]["mean_cosine_gte"] and m.get("mean_relative_l2",math.inf)<=p["quality_gates"]["mean_relative_l2_lte"] and m.get("mean_relative_l2",math.inf)<b.get("uniform10",{}).get("mean_relative_l2",-math.inf) and m.get("mean_total_kv_bytes",math.inf)<b.get("flat5",{}).get("mean_total_kv_bytes",-math.inf):out.append(t)
 return out
def freeze_development(r):
 x=dict(r);x["development_passing_targets"]=development_passing_targets(x);x["classification"]="C7-DEVELOPMENT-GO" if x["development_passing_targets"] else "C7-DEVELOPMENT-NO-GO";x["selected_target"]=x["development_passing_targets"][0] if x["development_passing_targets"] else None;return x
def develop_from_rows(*,rows,development_source_manifest,qualification,capture_manifest,output):
 if Path(output).exists() or Path(output).is_symlink():raise C7Error("refusing overwrite")
 validate_development_manifest(_json(development_source_manifest),development_source_manifest);c4._validate_c2_qualification(qualification)
 for a in ACTIONS:validate_observation_inventory(rows[a],action=a)
 summaries={x:_summary(v) for x,v in rows.items()}
 if any(x["observation_count"]!=8640 for x in summaries.values()):raise C7Error("C7 count differs")
 stats,schedules=_development_schedules(rows);targets={t:(_summary(rows_for_schedule(rows,s["table"])) if s["feasible"] else {}) for t,s in schedules.items()}
 r={"schema_version":"cascadekv-phi35-8k-c7-development-v1","protocol_sha256":sha256_path(PROTOCOL),"runtime_manifest_sha256":sha256_path(RUNTIME),"qualification_sha256":sha256_path(qualification),"development_source_manifest_sha256":sha256_path(development_source_manifest),"capture_manifest_sha256":sha256_path(capture_manifest),"development_baseline_metrics":{x:summaries[x] for x in c4.BASELINES},"development_action_metrics":{x:summaries[x] for x in ACTIONS},"action_cell_statistics":stats,"schedules":schedules,"development_target_metrics":targets,"target_feasibility":{x:schedules[x]["feasible"] for x in TARGETS},"optimizer":validate_protocol()["optimizer"],"observation_counts":{x:summaries[x]["observation_count"] for x in ACTIONS}}
 r["schedule_sha256"]=schedule_digest(schedules);r=freeze_development(r);atomic_json(Path(output),r);return r
def assert_holdout_may_start(dev,sha,path=None):
 if dev.get("classification")!="C7-DEVELOPMENT-GO":raise C7Error("development GO required")
 if path is not None and (sha256_path(path)!=sha or _json(path)!=dict(dev)):raise C7Error("development bytes not frozen")
 if dev.get("schedule_sha256")!=schedule_digest(dev.get("schedules",{})) or dev.get("selected_target") not in development_passing_targets(dev):raise C7Error("selected target not frozen")
 return dev["selected_target"]
def prospective_verdict(selected_target,gates):
 row=gates.get(selected_target)
 if not isinstance(row,Mapping) or type(row.get("passes")) is not bool:raise C7Error("selected target missing")
 return "C7-PROSPECTIVE-PASS" if row["passes"] else "C7-PROSPECTIVE-NO-PASS"

def final_audit(root):
 """Offline evidence audit run immediately before C7 result bundling."""
 root=Path(root)
 for name in ("protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json"):
  if not (root/name).is_file():raise C7Error("final audit missing required evidence")
 # The bundled root copies are evidence, not merely convenience inputs.  Bind
 # them to the frozen repository bytes before trusting any hashes recorded in
 # downstream JSON.  Otherwise a mutually-consistent replacement copy could
 # evade the development-result bindings below.
 if sha256_path(root/"protocol.json")!=sha256_path(PROTOCOL) or sha256_path(root/"runtime_manifest.json")!=sha256_path(RUNTIME):raise C7Error("final root configuration evidence differs")
 qrecord=c4._validate_c2_qualification(root/"qualification.json")
 if sha256_path(root/"qualification.json")!=c4.QUALIFICATION_SHA or qrecord.get("backend_id")!=c4.BACKEND_ID:raise C7Error("final qualification evidence differs")
 for path in root.rglob("*"):
  if path.is_file() and not path.is_symlink():
   try:_no_raw(path.read_text())
   except UnicodeDecodeError:pass
 dev=_json(root/"development.json")
 if dev.get("protocol_sha256")!=sha256_path(PROTOCOL) or dev.get("runtime_manifest_sha256")!=sha256_path(RUNTIME) or dev.get("qualification_sha256")!=sha256_path(root/"qualification.json") or dev.get("development_source_manifest_sha256")!=sha256_path(root/"development_sources.json") or dev.get("capture_manifest_sha256")!=sha256_path(root/"development_capture_manifest.json"):raise C7Error("final development bindings differ")
 if dev.get("development_passing_targets")!=development_passing_targets(dev) or dev.get("classification")!=("C7-DEVELOPMENT-GO" if dev["development_passing_targets"] else "C7-DEVELOPMENT-NO-GO"):raise C7Error("final development classification differs")
 if dev.get("classification")=="C7-DEVELOPMENT-NO-GO":
  if any((root/x).exists() for x in ("holdout.json","holdout_capture_manifest.json","test.json")):raise C7Error("NO-GO contains prospective evidence")
  return "C7-DEVELOPMENT-NO-GO"
 if assert_holdout_may_start(dev,sha256_path(root/"development.json"),root/"development.json") is None:raise C7Error("final selected target absent")
 for name in ("holdout.json","holdout_capture_manifest.json","test.json"):
  if not (root/name).is_file():raise C7Error("GO final evidence missing")
 hold=_json(root/"holdout.json");validate_holdout_manifest(hold,root/"holdout.json",development_sources=root/"development_sources.json",development_result=root/"development.json")
 test=_json(root/"test.json")
 expected=prospective_verdict(dev["selected_target"],test.get("per_gate_pass_fail",{}))
 firewall=test.get("firewall_evidence",{})
 if (test.get("protocol_sha256")!=sha256_path(PROTOCOL) or test.get("runtime_manifest_sha256")!=sha256_path(RUNTIME) or
     test.get("capture_manifest_sha256")!=sha256_path(root/"holdout_capture_manifest.json") or
     hold.get("selected_target")!=dev.get("selected_target") or
     test.get("selected_target")!=dev.get("selected_target") or
     dev.get("selected_target") not in dev.get("development_passing_targets",[]) or
     hold.get("development_result_sha256")!=sha256_path(root/"development.json") or
     test.get("development_result_sha256")!=sha256_path(root/"development.json") or
     test.get("holdout_manifest_sha256")!=sha256_path(root/"holdout.json") or
     hold.get("schedule_sha256")!=dev.get("schedule_sha256") or
     firewall.get("schedule_sha256_before")!=dev.get("schedule_sha256") or
     firewall.get("schedule_sha256_after")!=dev.get("schedule_sha256") or
     firewall.get("optimizer_rerun") is not False or firewall.get("schedule_unchanged") is not True or
     firewall.get("opened_development_tensor_identities")!=[] or
     test.get("classification")!=expected):raise C7Error("final prospective bindings differ")
 return test["classification"]

# The following adapters deliberately retain C6's tensor format and evaluator,
# but make every C7 identity, count, and manifest binding explicit.
def _expected_tensor_ids(kind, holdout=None):
 if kind=="development": return sorted(f"{f}:{i}:development:L{l}" for f in c4.FAMILIES for i in DEVELOPMENT[f] for l in c4.LAYERS)
 if not holdout: raise C7Error("holdout manifest required")
 return sorted(f"{x['family']}:{x['dataset_index']}:holdout:L{l}" for x in holdout["selected_sources"] for l in c4.LAYERS)
def _source_identity(s,role): return f"{s['family']}:{s['dataset_index']}:{role}"

def validate_holdout_manifest(payload,path=None,*,development_sources=None,development_result=None):
 _no_raw(payload); req={"schema_version","protocol_sha256","development_source_manifest_sha256","development_result_sha256","schedule_sha256","selected_target","selection_rule","selected_sources","rejected_sources","next_untouched_frontier","raw_text_persisted"}
 if set(payload)!=req or payload.get("schema_version")!="cascadekv-phi35-8k-c7-holdout-selection-v1" or payload.get("protocol_sha256")!=sha256_path(PROTOCOL) or payload.get("selection_rule")!="first mechanically eligible globally-new exact input SHA in ascending order" or payload.get("raw_text_persisted") is not False: raise C7Error("holdout manifest differs")
 if development_sources is None:raise C7Error("frozen development sources required")
 dev=validate_development_manifest(_json(development_sources) if not isinstance(development_sources,Mapping) else development_sources)
 if not isinstance(development_sources,Mapping) and payload.get("development_source_manifest_sha256")!=sha256_path(development_sources):raise C7Error("holdout development source binding differs")
 if development_result is not None:
  result=_json(development_result) if not isinstance(development_result,Mapping) else dict(development_result)
  result_sha=sha256_path(development_result) if not isinstance(development_result,Mapping) else None
  # A selection has no authority apart from the exact frozen GO result.
  if result_sha is None or payload.get("development_result_sha256")!=result_sha or payload.get("schedule_sha256")!=result.get("schedule_sha256") or payload.get("selected_target")!=assert_holdout_may_start(result,result_sha,development_result):raise C7Error("holdout frozen development binding differs")
 development_hashes={x["input_ids_sha256"] for x in dev};out=[]; seen=set()
 for f in c4.FAMILIES:
  rows=[x for x in payload.get("selected_sources",[]) if isinstance(x,Mapping) and x.get("family")==f]
  if len(rows)!=1: raise C7Error("holdout family inventory differs")
  x=rows[0]; h=x.get("input_ids_sha256")
  if x.get("role")!="holdout" or x.get("identity_kind")!="dataset_index" or type(x.get("dataset_index")) is not int or x["dataset_index"]<FRONTIER[f] or any(x.get(k)!=c4.SOURCE_SPECS[f][k] for k in ("dataset","config","split","revision","field")) or not isinstance(h,str) or not re.fullmatch(r"[0-9a-f]{64}",h) or h in seen or h in development_hashes: raise C7Error("holdout source differs")
  seen.add(h);out.append(dict(x))
 if len(payload.get("selected_sources",[]))!=3 or set(payload.get("next_untouched_frontier",{}))!=set(c4.FAMILIES) or any(payload["next_untouched_frontier"][x["family"]]!=x["dataset_index"]+1 for x in out) or (path is not None and not Path(path).is_file()): raise C7Error("holdout manifest incomplete")
 # Selection is auditable only if the skipped prefix is a complete, ordered
 # account of why the selected index was not reached sooner.  Do not merely
 # retain this field as narrative evidence: bind it to the frozen frontier and
 # to the exact development hash set.
 rejected=payload.get("rejected_sources")
 if not isinstance(rejected,Mapping) or set(rejected)!=set(c4.FAMILIES): raise C7Error("holdout rejected-source inventory differs")
 selected_by_family={x["family"]:x for x in out}
 for f in c4.FAMILIES:
  rows=rejected[f]; selected_row=selected_by_family[f]
  if not isinstance(rows,list) or [x.get("dataset_index") if isinstance(x,Mapping) else None for x in rows]!=list(range(FRONTIER[f],selected_row["dataset_index"])): raise C7Error("holdout rejected-source sequence differs")
  for row in rows:
   proof=row.get("proof")
   if row.get("reason")=="mechanically_ineligible":
    if set(row)!={"dataset_index","reason","proof"} or c4._proof_eligible(proof): raise C7Error("invalid mechanically rejected source")
   elif row.get("reason")=="duplicate_exact_input_ids_sha256":
    digest=row.get("input_ids_sha256")
    if set(row)!={"dataset_index","reason","input_ids_sha256","proof"} or not c4._proof_eligible(proof) or not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest) or digest not in development_hashes: raise C7Error("invalid duplicate rejected source")
   else: raise C7Error("unknown rejected-source reason")
 return out

def select_holdout_from_inspections(inspections,development_hashes,*,development_result_sha256,schedule_sha256,development_source_manifest_sha256):
 known=set(development_hashes)
 if len(known)!=18 or any(not isinstance(x,str) or not re.fullmatch(r"[0-9a-f]{64}",x) for x in known): raise C7Error("eighteen development hashes required")
 selected=[]; rejected={f:[] for f in c4.FAMILIES}; next_frontier={}
 for f in c4.FAMILIES:
  accepted=None;last=FRONTIER[f]-1
  for row in sorted((dict(x) for x in inspections.get(f,[])),key=lambda x:x.get("dataset_index",-1)):
   i=row.get("dataset_index")
   if type(i) is not int or i<FRONTIER[f]: continue
   last=i; proof=row.get("proof",{}); digest=row.get("input_ids_sha256")
   if not c4._proof_eligible(proof): rejected[f].append({"dataset_index":i,"reason":"mechanically_ineligible","proof":proof});continue
   if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest): raise C7Error("malformed input hash")
   if digest in known: rejected[f].append({"dataset_index":i,"reason":"duplicate_exact_input_ids_sha256","input_ids_sha256":digest,"proof":proof});continue
   source=dict(row["source"]);_no_raw(source)
   if source.get("family")!=f or source.get("dataset_index")!=i: raise C7Error("holdout source identity differs")
   accepted={**source,"role":"holdout","proof":proof,"input_ids_sha256":digest};known.add(digest);break
  if accepted is None: raise C7Error("no new eligible C7 holdout")
  selected.append(accepted);next_frontier[f]=last+1
 return {"schema_version":"cascadekv-phi35-8k-c7-holdout-selection-v1","protocol_sha256":sha256_path(PROTOCOL),"development_source_manifest_sha256":development_source_manifest_sha256,"development_result_sha256":development_result_sha256,"schedule_sha256":schedule_sha256,"selected_target":None,"selection_rule":"first mechanically eligible globally-new exact input SHA in ascending order","selected_sources":selected,"rejected_sources":rejected,"next_untouched_frontier":next_frontier,"raw_text_persisted":False}

class CaptureResolver:
 """Manifest/provenance-first C7 resolver; no tensor library is imported here."""
 def __init__(self,root,manifest,kind,*,development_source_manifest=None,holdout_manifest=None,development_sha=None,schedule_sha=None):
  self.root=Path(root).resolve();self.kind=kind; payload=_json(manifest);_no_raw(payload); count,forwards=(90,18) if kind=="development" else (15,3)
  req={"schema_version","kind","protocol_sha256","qualification_sha256","backend_id","development_source_manifest_sha256","holdout_manifest_sha256","development_result_sha256","schedule_sha256","artifact_count","forwards","artifacts"}
  if set(payload)!=req or payload.get("schema_version")!="cascadekv-phi35-8k-c7-capture-v1" or payload.get("kind")!=kind or payload.get("protocol_sha256")!=sha256_path(PROTOCOL) or payload.get("qualification_sha256")!=c4.QUALIFICATION_SHA or payload.get("backend_id")!=c4.BACKEND_ID or payload.get("artifact_count")!=count or payload.get("forwards")!=forwards: raise C7Error("capture manifest differs")
  if kind=="development":
   if development_source_manifest is None or payload.get("development_source_manifest_sha256")!=sha256_path(Path(development_source_manifest)) or any(payload.get(k) is not None for k in ("holdout_manifest_sha256","development_result_sha256","schedule_sha256")): raise C7Error("development capture binding differs")
   sources=validate_development_manifest(_json(development_source_manifest),development_source_manifest); hold=None
  else:
   if holdout_manifest is None or payload.get("holdout_manifest_sha256")!=sha256_path(Path(holdout_manifest)) or payload.get("development_result_sha256")!=development_sha or payload.get("schedule_sha256")!=schedule_sha: raise C7Error("holdout capture binding differs")
   hold=_json(holdout_manifest)
   if development_source_manifest is None:raise C7Error("holdout resolver requires development manifest")
   sources=validate_holdout_manifest(hold,holdout_manifest,development_sources=development_source_manifest)
  expected=set(_expected_tensor_ids(kind,hold)); source_map={_source_identity(x,kind):x for x in sources}; records={}
  for row in payload["artifacts"]:
   if not isinstance(row,Mapping) or set(row)!={"identity","layer","artifact_relative_path","artifact_sha256","provenance_relative_path","provenance_sha256","input_ids_sha256","source"}: raise C7Error("capture record schema differs")
   key=f"{row.get('identity')}:L{row.get('layer')}"; source=source_map.get(row.get("identity"))
   if key not in expected or key in records or source is None or row.get("source")!={"family":source["family"],"dataset_index":source["dataset_index"],"role":kind} or row.get("input_ids_sha256")!=source["input_ids_sha256"]: raise C7Error("capture inventory differs")
   for field,digest in (("artifact_relative_path","artifact_sha256"),("provenance_relative_path","provenance_sha256")):
    rel=row.get(field)
    if not isinstance(rel,str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts or not isinstance(row.get(digest),str) or not re.fullmatch(r"[0-9a-f]{64}",row[digest]): raise C7Error("unsafe capture path")
    raw=self.root/rel; p=raw.resolve()
    # Check the submitted pathname before resolution: ``resolve()`` erases
    # the symlink bit and would otherwise admit a symlink that points back
    # inside the capture root.
    if not p.is_relative_to(self.root) or raw.is_symlink() or not p.is_file() or sha256_path(p)!=row[digest]: raise C7Error("capture artifact missing or differs")
   detail=_json((self.root/row["provenance_relative_path"]).resolve());_no_raw(detail)
   expected_source={**{k:source[k] for k in ("family","dataset","config","split","revision","field","identity_kind","dataset_index")},"role":kind}
   if not _c7_artifact_provenance_matches(detail,row,expected_source):
    raise C7Error("capture provenance differs")
   records[key]=dict(row)
  if set(records)!=expected: raise C7Error("capture inventory incomplete")
  self.records=records
  # TensorAccess compares caller-supplied records to this independent snapshot.
  # ``ordered`` also returns copies so a caller cannot mutate its authorization.
  self._authorized_records=_freeze({k:json.loads(canonical_json(v)) for k,v in records.items()})
  self._authorized_sources=_freeze({k:{**{name:source_map[v["identity"]][name] for name in ("family","dataset","config","split","revision","field","identity_kind","dataset_index")},"role":kind} for k,v in records.items()})
 def ordered(self): return [json.loads(canonical_json(self.records[k])) for k in sorted(self.records)]

def _c7_artifact_provenance_matches(detail,row,expected_source):
 return (detail.get("schema_version")=="cascadekv-phi35-8k-c7-artifact-v1" and
         detail.get("protocol_sha256")==sha256_path(PROTOCOL) and detail.get("qualification_sha256")==c4.QUALIFICATION_SHA and
         detail.get("backend_id")==c4.BACKEND_ID and detail.get("identity")==row["identity"] and detail.get("layer")==row["layer"] and
         detail.get("input_ids_sha256")==row["input_ids_sha256"] and detail.get("artifact_sha256")==row["artifact_sha256"] and
         detail.get("source")==expected_source and
         detail.get("target")=={"model":c4.MODEL,"revision":c4.REVISION,"tokenizer_revision":c4.REVISION} and
         detail.get("q_shape")==list(c4.SHAPE) and detail.get("k_shape")==list(c4.SHAPE) and detail.get("v_shape")==list(c4.SHAPE) and
         detail.get("storage_dtype")=="float16" and detail.get("model_compute_dtype")=="float16" and
         detail.get("attention_implementation")=="sdpa" and detail.get("use_cache") is False and
         detail.get("quantization")=="none" and detail.get("capture_adapter")=="phi3-post-rope-qkv-v1")

class TensorAccess(c6.TensorAccess):
 def __init__(self,resolver,*,allowed_kind):
  if resolver.kind!=allowed_kind: raise C7Error("tensor firewall denied capture kind")
  self.resolver=resolver;self.allowed_kind=allowed_kind;self.opened=[]
 def open(self,record):
  """Open only the resolver-authorized C7 artifact, immediately rechecking it."""
  if not isinstance(record,Mapping):raise C7Error("tensor record differs")
  key=f"{record.get('identity')}:L{record.get('layer')}"
  authorized=self.resolver._authorized_records.get(key)
  expected_source=self.resolver._authorized_sources.get(key)
  if authorized is None or expected_source is None or dict(record)!=authorized:raise C7Error("tensor record differs")
  root=self.resolver.root
  def safe(field):
   rel=authorized[field]
   raw=root/rel;p=raw.resolve()
   if raw.is_symlink() or not p.is_relative_to(root) or not p.is_file():raise C7Error("artifact path escapes root")
   return p
  artifact,provenance=safe("artifact_relative_path"),safe("provenance_relative_path")
  if sha256_path(artifact)!=authorized["artifact_sha256"] or sha256_path(provenance)!=authorized["provenance_sha256"]:raise C7Error("artifact digest differs")
  detail=_json(provenance);_no_raw(detail)
  if not _c7_artifact_provenance_matches(detail,authorized,expected_source):raise C7Error("artifact provenance differs")
  from safetensors import safe_open
  import torch
  with safe_open(str(artifact),framework="pt",device="cpu") as h:
   if set(h.keys())!={"q","k","v"}:raise C7Error("tensor keys differ")
   q,k,v=(h.get_tensor(x) for x in ("q","k","v"))
  if any(x.dtype!=torch.float16 or tuple(x.shape)!=c4.SHAPE for x in (q,k,v)):raise C7Error("tensor shape/dtype differs")
  self.opened.append(f"{authorized['identity']}:L{authorized['layer']}");return q,k,v

def audit_prospective_test(*,opened_development,opened_holdout,optimizer_rerun,schedule_before,schedule_after):
 if opened_development or optimizer_rerun or schedule_before!=schedule_after: raise C7Error("prospective test firewall failed")
 return {"opened_development_tensor_identities":[],"opened_holdout_tensor_identities":opened_holdout,"optimizer_rerun":False,"schedule_sha256_before":schedule_before,"schedule_sha256_after":schedule_after,"schedule_unchanged":True}

def develop(*,capture_root,capture_manifest,development_source_manifest,qualification,output):
 if Path(output).exists() or Path(output).is_symlink(): raise C7Error("refusing overwrite")
 protocol=validate_protocol();validate_development_manifest(_json(development_source_manifest),development_source_manifest);c4._validate_c2_qualification(qualification)
 resolver=CaptureResolver(capture_root,capture_manifest,"development",development_source_manifest=development_source_manifest);access=TensorAccess(resolver,allowed_kind="development")
 rows=c6._method_rows(access,protocol);expected=_expected_tensor_ids("development")
 if access.opened!=expected: raise C7Error("development tensor firewall differs")
 for a in ACTIONS: validate_observation_inventory(rows[a],action=a)
 summaries={x:_summary(v) for x,v in rows.items()}
 if any(x["observation_count"]!=8640 for x in summaries.values()): raise C7Error("development observation count differs")
 stats,schedules=_development_schedules(rows); target_metrics={t:(_summary(rows_for_schedule(rows,s["table"])) if s["feasible"] else {}) for t,s in schedules.items()}
 result={"schema_version":"cascadekv-phi35-8k-c7-development-v1","protocol_sha256":sha256_path(PROTOCOL),"runtime_manifest_sha256":sha256_path(RUNTIME),"qualification_sha256":sha256_path(qualification),"development_source_manifest_sha256":sha256_path(development_source_manifest),"capture_manifest_sha256":sha256_path(capture_manifest),"opened_development_tensor_identities":expected,"opened_holdout_tensor_identities":[],"methods":protocol["methods"],"development_baseline_metrics":{x:summaries[x] for x in c4.BASELINES},"development_action_metrics":{x:summaries[x] for x in ACTIONS},"action_cell_statistics":stats,"schedules":schedules,"development_target_metrics":target_metrics,"target_feasibility":{x:schedules[x]["feasible"] for x in TARGETS},"optimizer":protocol["optimizer"],"observation_counts":{x:summaries[x]["observation_count"] for x in ACTIONS}}
 result["schedule_sha256"]=schedule_digest(schedules);result=freeze_development(result);atomic_json(Path(output),result);return result

def prospective_test(*,holdout_capture_root,holdout_capture_manifest,holdout_manifest,development_result,development_source_manifest,output):
 if Path(output).exists() or Path(output).is_symlink(): raise C7Error("refusing overwrite")
 dev=_json(development_result);devsha=sha256_path(development_result);selected=assert_holdout_may_start(dev,devsha,development_result);hold=_json(holdout_manifest);sources=validate_holdout_manifest(hold,holdout_manifest,development_sources=development_source_manifest,development_result=development_result)
 if hold.get("development_result_sha256")!=devsha or hold.get("schedule_sha256")!=dev.get("schedule_sha256") or hold.get("selected_target")!=selected: raise C7Error("holdout not bound to frozen development")
 resolver=CaptureResolver(holdout_capture_root,holdout_capture_manifest,"holdout",development_source_manifest=development_source_manifest,holdout_manifest=holdout_manifest,development_sha=devsha,schedule_sha=dev["schedule_sha256"]);access=TensorAccess(resolver,allowed_kind="holdout"); schedules={t:dev["schedules"][t]["table"] for t in TARGETS if dev["schedules"][t]["feasible"]};rows=c6._method_rows(access,validate_protocol(),schedules);expected=_expected_tensor_ids("holdout",hold)
 if access.opened!=expected: raise C7Error("holdout tensor firewall differs")
 identities=[_source_identity(x,"holdout") for x in sources]
 for a in ACTIONS: validate_observation_inventory(rows[a],action=a,role="holdout",expected_holdout_sources=identities)
 summaries={x:_summary(v) for x,v in rows.items()}
 if any(x["observation_count"]!=1440 for x in summaries.values()): raise C7Error("holdout observation count differs")
 q=validate_protocol()["quality_gates"];gates={};first=None
 for t in TARGETS:
  if t not in schedules:gates[t]={"feasible":False,"passes":False};continue
  m=summaries[t];r={"feasible":True,"mean_cosine_gte":m["mean_cosine"]>=q["mean_cosine_gte"],"mean_relative_l2_lte":m["mean_relative_l2"]<=q["mean_relative_l2_lte"],"relative_l2_lt_uniform10":m["mean_relative_l2"]<summaries["uniform10"]["mean_relative_l2"],"modeled_kv_traffic_lt_flat5":m["mean_total_kv_bytes"]<summaries["flat5"]["mean_total_kv_bytes"]};r["passes"]=all(v for k,v in r.items() if k not in {"feasible","passes"});gates[t]=r
  if first is None and r["passes"]:first=t
 before=dev["schedule_sha256"];audit=audit_prospective_test(opened_development=[],opened_holdout=expected,optimizer_rerun=False,schedule_before=before,schedule_after=schedule_digest(dev["schedules"]))
 result={"schema_version":"cascadekv-phi35-8k-c7-test-v1","protocol_sha256":sha256_path(PROTOCOL),"runtime_manifest_sha256":sha256_path(RUNTIME),"development_result_sha256":devsha,"holdout_manifest_sha256":sha256_path(holdout_manifest),"capture_manifest_sha256":sha256_path(holdout_capture_manifest),"selected_target":selected,"development_passing_targets":dev["development_passing_targets"],"baseline_test_metrics":{x:summaries[x] for x in c4.BASELINES},"target_test_metrics":{x:summaries[x] for x in TARGETS if x in summaries},"per_gate_pass_fail":gates,"first_passing_target_descriptive":first,"firewall_evidence":audit,"classification":prospective_verdict(selected,gates)};atomic_json(Path(output),result);return result

def capture_sources(*,kind,sources,output,qualification,development_source_sha=None,holdout_manifest=None,development_sha=None,schedule_sha=None):
 """C7 capture, exactly five C7-provenanced artifacts for each source."""
 if Path(output).exists() or Path(output).is_symlink(): raise C7Error("refusing overwrite capture evidence")
 Path(output).mkdir(parents=True,exist_ok=False)
 qrecord=c4._validate_c2_qualification(qualification)
 from cascadekv import phi35_kaggle_c2 as c2
 from cascadekv.phi35_8k_source_selection import load_dataset_for_spec,load_frozen_tokenizer,mechanical_proof
 import torch
 tokenizer=load_frozen_tokenizer();model=c2._load_pinned_model();records=[]
 try:
  c2.validate_qualification(qrecord);c2.assert_capture_backend(model,qrecord)
  for source in sources:
   spec={k:source[k] for k in ("dataset","config","split","revision","field")};dataset=load_dataset_for_spec(spec);proof=mechanical_proof(dataset,tokenizer,spec,int(source["dataset_index"]));value=dataset[int(source["dataset_index"])].get(source["field"])
   if not c4._proof_eligible(proof) or not isinstance(value,str):raise C7Error("captured source no longer eligible")
   ids=torch.tensor([tokenizer(value,add_special_tokens=True,truncation=True,max_length=8192).input_ids],dtype=torch.int64).contiguous();digest=c4._input_hash(ids)
   if digest!=source["input_ids_sha256"]:raise C7Error("captured input hash differs")
   identity=_source_identity(source,kind);captured=c2.capture_five_layers_post_rope_qkv(model,ids.to(c2._first_device(model)))
   for layer in c4.LAYERS:
    stem=f"{source['family']}_{kind}_index{source['dataset_index']}_layer{layer}";ar=f"artifacts/{stem}.safetensors";pr=f"artifacts/{stem}.provenance.json";artifact=Path(output)/ar;provenance=Path(output)/pr
    if artifact.exists() or artifact.is_symlink() or provenance.exists() or provenance.is_symlink():raise C7Error("never overwrite artifact")
    detail={"schema_version":"cascadekv-phi35-8k-c7-artifact-v1","protocol_sha256":sha256_path(PROTOCOL),"qualification_sha256":c4.QUALIFICATION_SHA,"backend_id":c4.BACKEND_ID,"identity":identity,"layer":layer,"input_ids_sha256":digest,"target":{"model":c4.MODEL,"revision":c4.REVISION,"tokenizer_revision":c4.REVISION},"source":{**{k:source[k] for k in ("family","dataset","config","split","revision","field","identity_kind","dataset_index")},"role":kind},"source_proof":proof,"q_shape":list(c4.SHAPE),"k_shape":list(c4.SHAPE),"v_shape":list(c4.SHAPE),"storage_dtype":"float16","model_compute_dtype":"float16","attention_implementation":"sdpa","use_cache":False,"quantization":"none","capture_adapter":"phi3-post-rope-qkv-v1"}
    written=c2.write_artifact(artifact,provenance,captured[layer],detail);ah=sha256_path(artifact)
    if written.get("artifact_sha256")!=ah or _json(provenance)!=written:raise C7Error("capture provenance chain differs")
    records.append({"identity":identity,"layer":layer,"artifact_relative_path":ar,"artifact_sha256":ah,"provenance_relative_path":pr,"provenance_sha256":sha256_path(provenance),"input_ids_sha256":digest,"source":{"family":source["family"],"dataset_index":source["dataset_index"],"role":kind}})
  expected=(90,18) if kind=="development" else (15,3)
  if (len(records),len(sources))!=expected:raise C7Error("C7 capture count differs")
  final={"schema_version":"cascadekv-phi35-8k-c7-capture-v1","kind":kind,"protocol_sha256":sha256_path(PROTOCOL),"qualification_sha256":c4.QUALIFICATION_SHA,"backend_id":c4.BACKEND_ID,"development_source_manifest_sha256":development_source_sha if kind=="development" else None,"holdout_manifest_sha256":sha256_path(holdout_manifest) if holdout_manifest else None,"development_result_sha256":development_sha,"schedule_sha256":schedule_sha,"artifact_count":len(records),"forwards":len(sources),"artifacts":records};atomic_json(Path(output)/"final_capture_manifest.json",final);return final
 finally: del model

def select_holdout(*,development_source_manifest,development_result,output):
 """Prospective-only selector.  Freeze checks intentionally precede source imports."""
 if Path(output).exists() or Path(output).is_symlink():raise C7Error("refusing overwrite holdout evidence")
 dev=_json(development_result);devsha=sha256_path(development_result);target=assert_holdout_may_start(dev,devsha,development_result);sources=validate_development_manifest(_json(development_source_manifest),development_source_manifest)
 from cascadekv.phi35_8k_source_selection import frozen_specs,load_dataset_for_spec,load_frozen_tokenizer,mechanical_proof
 import torch
 tokenizer,specs=load_frozen_tokenizer(),frozen_specs();inspections={f:[] for f in c4.FAMILIES};known={x["input_ids_sha256"] for x in sources}
 for family in c4.FAMILIES:
  dataset=load_dataset_for_spec(specs[family]);index=FRONTIER[family]
  while index<len(dataset):
   proof=mechanical_proof(dataset,tokenizer,specs[family],index);value=dataset[index].get(specs[family]["field"]);digest="0"*64
   if c4._proof_eligible(proof) and isinstance(value,str):digest=c4._input_hash(torch.tensor([tokenizer(value,add_special_tokens=True,truncation=True,max_length=8192).input_ids],dtype=torch.int64).contiguous())
   inspections[family].append({"dataset_index":index,"proof":proof,"input_ids_sha256":digest,"source":{"family":family,**{k:specs[family][k] for k in ("dataset","config","split","revision","field")},"identity_kind":"dataset_index","dataset_index":index}})
   if c4._proof_eligible(proof) and digest not in known:break
   index+=1
 result=select_holdout_from_inspections(inspections,[x["input_ids_sha256"] for x in sources],development_result_sha256=devsha,schedule_sha256=dev["schedule_sha256"],development_source_manifest_sha256=sha256_path(development_source_manifest));result["selected_target"]=target;validate_holdout_manifest(result,development_sources=development_source_manifest,development_result=development_result);atomic_json(Path(output),result);return result
