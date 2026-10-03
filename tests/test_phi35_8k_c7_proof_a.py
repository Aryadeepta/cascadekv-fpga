"""Proof-A synthetic matrices: resolver boundaries and family arithmetic only.

This module deliberately never imports torch, safetensors, a tokenizer, or a
dataset.  The resolver assertions run before ``TensorAccess.open``; patching
that method makes the fail-before-payload property observable.
"""
import copy
import hashlib
import json

import pytest

from cascadekv import phi35_8k_c7 as c7


def _digest(data): return hashlib.sha256(data).hexdigest()


def _dev_manifest(tmp_path):
    path=tmp_path / "development_sources.json"
    c7.build_development_manifest(output=path)
    return path, json.loads(path.read_text())


def _rows():
    """All 54 observations for one action/cell inventory, with visible means."""
    rows=[]
    value={"narrative":1.0,"report":2.0,"qa":3.0}
    for family in c7.c4.FAMILIES:
        for index in c7.DEVELOPMENT[family]:
            for layer in c7.c4.LAYERS:
                for position in c7.c4.POSITIONS:
                    for head in c7.c4.HEADS:
                        rows.append({"source":f"{family}:{index}:development","layer":layer,"position":position,"head":head,"metrics":{"relative_l2_error":value[family],"cosine_similarity":.9},"traffic":{"total_kv_bytes":10.0}})
    return rows


def test_family_aggregation_valid_exact_means_and_order_invariance():
    base=_rows(); by_action={action:copy.deepcopy(base) for action in c7.ACTIONS}
    stats=c7.family_cell_statistics(by_action)
    one=stats[f"{c7.c4.LAYERS[0]}:{c7.c4.HEADS[0]}:{c7.ACTIONS[0]}"]
    assert (one["narrative_mean_relative_l2"],one["report_mean_relative_l2"],one["qa_mean_relative_l2"],one["robust_relative_l2"],one["family_observation_count"],one["global_observation_count"]) == (1.0,2.0,3.0,3.0,18,54)
    shuffled={a:list(reversed(v)) for a,v in reversed(list(by_action.items()))}
    assert c7.family_cell_statistics(shuffled)==stats


@pytest.mark.parametrize("mutation",[
    "narrative17","narrative19","report17","report19","qa17","qa19",
    "duplicate","missing","unexpected","report20","qa14","layer","position","head",
    "nan_rel","inf_rel","nan_cos","inf_cos","nan_traffic","inf_traffic","spoof",
])
def test_family_aggregation_21_negative_cases(mutation):
    rows=_rows()
    if mutation.endswith("17"):
        family=mutation[:-2]; rows.pop(next(i for i,x in enumerate(rows) if x["source"].startswith(family+":")))
    elif mutation.endswith("19"):
        family=mutation[:-2]; rows.append(copy.deepcopy(next(x for x in rows if x["source"].startswith(family+":"))))
    elif mutation in {"duplicate","missing"}: rows.pop()
    elif mutation=="unexpected": rows[0]["source"]="narrative:999:development"
    elif mutation=="report20": rows[0]["source"]="report:20:development"
    elif mutation=="qa14": rows[0]["source"]="qa:14:development"
    elif mutation in {"layer","position","head"}: rows[0][mutation]=999
    elif mutation=="nan_rel": rows[0]["metrics"]["relative_l2_error"]=float("nan")
    elif mutation=="inf_rel": rows[0]["metrics"]["relative_l2_error"]=float("inf")
    elif mutation=="nan_cos": rows[0]["metrics"]["cosine_similarity"]=float("nan")
    elif mutation=="inf_cos": rows[0]["metrics"]["cosine_similarity"]=float("inf")
    elif mutation=="nan_traffic": rows[0]["traffic"]["total_kv_bytes"]=float("nan")
    elif mutation=="inf_traffic": rows[0]["traffic"]["total_kv_bytes"]=float("inf")
    elif mutation=="spoof": rows[0]["source"]="report:"+rows[0]["source"].split(":")[1]+":development"
    with pytest.raises(c7.C7Error): c7.family_cell_statistics({a:copy.deepcopy(rows) for a in c7.ACTIONS})


def test_family_aggregation_same_count_duplicate_replacement_and_exact_family_spoof_reject():
    rows=_rows(); rows[1]=copy.deepcopy(rows[0])
    with pytest.raises(c7.C7Error): c7.family_cell_statistics({a:copy.deepcopy(rows) for a in c7.ACTIONS})
    rows=_rows(); rows[0]["source"]="report:13:development"  # syntactic, but not a report-manifest identity
    with pytest.raises(c7.C7Error): c7.family_cell_statistics({a:copy.deepcopy(rows) for a in c7.ACTIONS})


def _capture(tmp_path, kind="development"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds,_=_dev_manifest(tmp_path); root=tmp_path/"capture"; root.mkdir(); artifacts=[]
    if kind=="development": sources=c7.validate_development_manifest(json.loads(ds.read_text()),ds); layers=c7.c4.LAYERS
    else: raise AssertionError("holdout fixture is covered by production binding tests")
    for source in sources:
        identity=f"{source['family']}:{source['dataset_index']}:development"
        for layer in layers:
            ar=f"a/{identity.replace(':','_')}_{layer}.bin"; pr=f"p/{identity.replace(':','_')}_{layer}.json"
            artifact=root/ar; provenance=root/pr; artifact.parent.mkdir(parents=True,exist_ok=True);provenance.parent.mkdir(parents=True,exist_ok=True)
            artifact.write_bytes(b"synthetic payload")
            detail={"schema_version":"cascadekv-phi35-8k-c7-artifact-v1","protocol_sha256":c7.sha256_path(c7.PROTOCOL),"qualification_sha256":c7.c4.QUALIFICATION_SHA,"backend_id":c7.c4.BACKEND_ID,"identity":identity,"layer":layer,"input_ids_sha256":source["input_ids_sha256"],"artifact_sha256":_digest(b"synthetic payload"),"source":{**{k:source[k] for k in ("family","dataset","config","split","revision","field","identity_kind","dataset_index")},"role":"development"},"target":{"model":c7.c4.MODEL,"revision":c7.c4.REVISION,"tokenizer_revision":c7.c4.REVISION},"q_shape":list(c7.c4.SHAPE),"k_shape":list(c7.c4.SHAPE),"v_shape":list(c7.c4.SHAPE),"storage_dtype":"float16","model_compute_dtype":"float16","attention_implementation":"sdpa","use_cache":False,"quantization":"none","capture_adapter":"phi3-post-rope-qkv-v1"}
            provenance.write_text(json.dumps(detail));artifacts.append({"identity":identity,"layer":layer,"artifact_relative_path":ar,"artifact_sha256":_digest(b"synthetic payload"),"provenance_relative_path":pr,"provenance_sha256":_digest(provenance.read_bytes()),"input_ids_sha256":source["input_ids_sha256"],"source":{"family":source["family"],"dataset_index":source["dataset_index"],"role":"development"}})
    manifest={"schema_version":"cascadekv-phi35-8k-c7-capture-v1","kind":"development","protocol_sha256":c7.sha256_path(c7.PROTOCOL),"qualification_sha256":c7.c4.QUALIFICATION_SHA,"backend_id":c7.c4.BACKEND_ID,"development_source_manifest_sha256":c7.sha256_path(ds),"holdout_manifest_sha256":None,"development_result_sha256":None,"schedule_sha256":None,"artifact_count":90,"forwards":18,"artifacts":artifacts}
    mp=root/"manifest.json";mp.write_text(json.dumps(manifest));return root,mp,ds,manifest


def test_development_resolver_valid_and_30_manifest_provenance_negatives_fail_before_payload(tmp_path,monkeypatch):
    root,mp,ds,manifest=_capture(tmp_path); opens=[]
    monkeypatch.setattr(c7.TensorAccess,"open",lambda self,record: opens.append(record) or (_ for _ in ()).throw(AssertionError("payload opened")))
    assert len(c7.CaptureResolver(root,mp,"development",development_source_manifest=ds).ordered())==90
    # Mutations span top-level identity/count/path/digest bindings and every
    # provenance field group.  Resolver construction cannot invoke open().
    cases=[("protocol_sha256","0"*64),("qualification_sha256","0"*64),("backend_id","0"*64),("forwards",17),("artifact_count",89),
           ("artifact_relative_path","/absolute"),("provenance_relative_path","../escape"),("artifact_sha256","0"*64),("provenance_sha256","0"*64),
           ("identity","narrative:999:development"),("layer",999),("input_ids_sha256","0"*64),("source",{}),
           ("detail:schema_version","bad"),("detail:protocol_sha256","0"*64),("detail:qualification_sha256","0"*64),("detail:backend_id","0"*64),
           ("detail:identity","bad"),("detail:layer",99),("detail:input_ids_sha256","0"*64),("detail:artifact_sha256","0"*64),
           ("detail:source",{}),("detail:target",{}),("detail:q_shape",[]),("detail:k_shape",[]),("detail:v_shape",[]),
           ("detail:storage_dtype","bad"),("detail:model_compute_dtype","bad"),("detail:attention_implementation","bad"),("detail:capture_adapter","bad")]
    for field,value in cases:
        candidate=copy.deepcopy(manifest); record=candidate["artifacts"][0]
        if field.startswith("detail:"):
            p=root/record["provenance_relative_path"]; detail=json.loads(p.read_text()); detail[field.split(":",1)[1]]=value; p.write_text(json.dumps(detail))
        elif field in candidate: candidate[field]=value
        else: record[field]=value
        bad=root/"bad.json";bad.write_text(json.dumps(candidate))
        with pytest.raises(c7.C7Error): c7.CaptureResolver(root,bad,"development",development_source_manifest=ds)
        assert opens==[]
        # restore fixture bytes/digest after each provenance mutation.
        if field.startswith("detail:"):
            root,mp,ds,manifest=_capture(tmp_path / ("fresh"+str(cases.index((field,value)))))


def _rewrite_provenance(root, manifest, record, mutate):
    path=root/record["provenance_relative_path"]
    detail=json.loads(path.read_text()); mutate(detail); path.write_text(json.dumps(detail))
    record["provenance_sha256"]=_digest(path.read_bytes())


@pytest.mark.parametrize("case", [
    "development_source_manifest_sha256", "forwards_19", "artifact_count_91",
    "missing_record", "extra_record", "duplicate_identity_layer", "wrong_role",
    "wrong_family", "wrong_dataset_index", "malformed_input_hash",
    "malformed_artifact_sha", "malformed_provenance_sha", "missing_artifact",
    "missing_provenance", "changed_artifact", "changed_provenance",
    "traversal_artifact", "absolute_provenance", "symlink_artifact", "symlink_provenance",
    "source_family", "source_dataset", "source_config", "source_split", "source_revision",
    "source_field", "source_identity_kind", "source_dataset_index", "source_role",
    "target_model", "target_revision", "target_tokenizer_revision", "use_cache", "quantization",
])
def test_development_resolver_additional_34_distinct_negatives_fail_before_payload(tmp_path, monkeypatch, case):
    root, mp, ds, manifest=_capture(tmp_path/case); opened=[]
    monkeypatch.setattr(c7.TensorAccess,"open",lambda self,record: opened.append(record))
    rec=manifest["artifacts"][0]
    if case=="development_source_manifest_sha256": manifest[case]="0"*64
    elif case=="forwards_19": manifest["forwards"]=19
    elif case=="artifact_count_91": manifest["artifact_count"]=91
    elif case=="missing_record": manifest["artifacts"].pop()
    elif case=="extra_record": manifest["artifacts"].append(copy.deepcopy(rec))
    elif case=="duplicate_identity_layer": manifest["artifacts"][1]["identity"]=rec["identity"];manifest["artifacts"][1]["layer"]=rec["layer"]
    elif case=="wrong_role": rec["source"]["role"]="holdout"
    elif case=="wrong_family": rec["source"]["family"]="report"
    elif case=="wrong_dataset_index": rec["source"]["dataset_index"]=999
    elif case=="malformed_input_hash": rec["input_ids_sha256"]="not-a-sha"
    elif case=="malformed_artifact_sha": rec["artifact_sha256"]="x"*64
    elif case=="malformed_provenance_sha": rec["provenance_sha256"]="x"*64
    elif case=="missing_artifact": (root/rec["artifact_relative_path"]).unlink()
    elif case=="missing_provenance": (root/rec["provenance_relative_path"]).unlink()
    elif case=="changed_artifact": (root/rec["artifact_relative_path"]).write_bytes(b"changed")
    elif case=="changed_provenance": (root/rec["provenance_relative_path"]).write_text("{}")
    elif case=="traversal_artifact": rec["artifact_relative_path"]="../outside"
    elif case=="absolute_provenance": rec["provenance_relative_path"]="/outside"
    elif case.startswith("symlink_"):
        field="artifact_relative_path" if case=="symlink_artifact" else "provenance_relative_path"; p=root/rec[field]; target=root/"linked"; target.write_bytes(p.read_bytes());p.unlink();p.symlink_to(target)
    else:
        mapping={"source_family":("source","family","bad"),"source_dataset":("source","dataset","bad"),"source_config":("source","config","bad"),"source_split":("source","split","bad"),"source_revision":("source","revision","bad"),"source_field":("source","field","bad"),"source_identity_kind":("source","identity_kind","bad"),"source_dataset_index":("source","dataset_index",999),"source_role":("source","role","bad"),"target_model":("target","model","bad"),"target_revision":("target","revision","bad"),"target_tokenizer_revision":("target","tokenizer_revision","bad"),"use_cache":(None,"use_cache",True),"quantization":(None,"quantization","bad")}
        outer,key,value=mapping[case]
        _rewrite_provenance(root,manifest,rec,lambda d: d.__setitem__(key,value) if outer is None else d[outer].__setitem__(key,value))
    bad=root/"bad.json";bad.write_text(json.dumps(manifest))
    with pytest.raises(c7.C7Error): c7.CaptureResolver(root,bad,"development",development_source_manifest=ds)
    assert opened==[]


def _frozen_go(tmp_path):
    ds,_=_dev_manifest(tmp_path)
    table={f"{l}:{h}":"A0" for l in c7.c4.LAYERS for h in c7.c4.HEADS}
    schedules={t:{"feasible":True,"target_mean_kv_bytes":1.,"strict":False,"integer_microbyte_cap":160000,"integer_microbytes_used":160000,"table":table} for t in c7.TARGETS}
    metrics={t:{"mean_cosine":.99,"mean_relative_l2":.1,"mean_total_kv_bytes":1.} for t in c7.TARGETS}
    dev={"classification":"C7-DEVELOPMENT-GO","selected_target":c7.TARGETS[0],"schedules":schedules,"schedule_sha256":c7.schedule_digest(schedules),"development_baseline_metrics":{"uniform10":{"mean_relative_l2":.2},"flat5":{"mean_total_kv_bytes":2.}},"development_target_metrics":metrics}
    dev["development_passing_targets"]=c7.development_passing_targets(dev)
    path=tmp_path/"development.json";path.write_text(json.dumps(dev));return ds,path,dev


def _holdout(tmp_path):
    ds,devpath,dev=_frozen_go(tmp_path); selected=[]
    for n,f in enumerate(c7.c4.FAMILIES):
        selected.append({"family":f,**c7.c4.SOURCE_SPECS[f],"identity_kind":"dataset_index","dataset_index":c7.FRONTIER[f],"role":"holdout","input_ids_sha256":f"{n+1:064x}"})
    x={"schema_version":"cascadekv-phi35-8k-c7-holdout-selection-v1","protocol_sha256":c7.sha256_path(c7.PROTOCOL),"development_source_manifest_sha256":c7.sha256_path(ds),"development_result_sha256":c7.sha256_path(devpath),"schedule_sha256":dev["schedule_sha256"],"selected_target":dev["selected_target"],"selection_rule":"first mechanically eligible globally-new exact input SHA in ascending order","selected_sources":selected,"rejected_sources":{f:[] for f in c7.c4.FAMILIES},"next_untouched_frontier":{x["family"]:x["dataset_index"]+1 for x in selected},"raw_text_persisted":False}
    path=tmp_path/"holdout.json";path.write_text(json.dumps(x));return ds,devpath,x,path


@pytest.mark.parametrize("case", ["missing_narrative","missing_report","missing_qa","duplicate_narrative","duplicate_report","duplicate_qa","extra","narrative_low","qa_low","report_low","malformed_hash","development_hash","duplicate_hash","wrong_family","wrong_dataset","wrong_config","wrong_split","wrong_revision","wrong_field","wrong_identity_kind","wrong_role","raw_text","dev_source_sha","dev_result_sha","schedule_sha","selected_missing","selected_different","selected_not_passing","frontier_keys","narrative_frontier","qa_frontier","report_frontier"])
def test_holdout_manifest_matrix_1_valid_plus_32_negatives(tmp_path, case):
    ds,dev,x,path=_holdout(tmp_path/case)
    assert len(c7.validate_holdout_manifest(x,path,development_sources=ds,development_result=dev))==3
    rows=x["selected_sources"]
    if case.startswith("missing_"): x["selected_sources"]=[r for r in rows if r["family"]!=case.split("_",1)[1]]
    elif case in {"duplicate_narrative","duplicate_report","duplicate_qa"}: x["selected_sources"].append(copy.deepcopy(next(r for r in rows if r["family"]==case.split("_",1)[1])))
    elif case=="extra": x["selected_sources"].append(copy.deepcopy(rows[0]))
    elif case.endswith("_low"): x["selected_sources"][next(i for i,r in enumerate(rows) if r["family"]==case.split("_",1)[0])]["dataset_index"]-=1
    elif case=="malformed_hash": rows[0]["input_ids_sha256"]="bad"
    elif case=="development_hash": rows[0]["input_ids_sha256"]=c7.validate_development_manifest(c7._json(ds),ds)[0]["input_ids_sha256"]
    elif case=="duplicate_hash": rows[1]["input_ids_sha256"]=rows[0]["input_ids_sha256"]
    elif case in {"wrong_family","wrong_dataset","wrong_config","wrong_split","wrong_revision","wrong_field","wrong_identity_kind","wrong_role"}:
        field=case.removeprefix("wrong_");rows[0][field]="bad"
    elif case=="raw_text": rows[0]["text"]=c7.RAW_TEXT_SENTINEL
    elif case=="dev_source_sha": x["development_source_manifest_sha256"]="0"*64
    elif case=="dev_result_sha": x["development_result_sha256"]="0"*64
    elif case=="schedule_sha": x["schedule_sha256"]="0"*64
    elif case=="selected_missing": x.pop("selected_target")
    elif case=="selected_different": x["selected_target"]=c7.TARGETS[1]
    elif case=="selected_not_passing":
        d=json.loads(dev.read_text());d["development_target_metrics"][d["selected_target"]]["mean_relative_l2"]=1.;d["development_passing_targets"]=c7.development_passing_targets(d);dev.write_text(json.dumps(d));x["development_result_sha256"]=c7.sha256_path(dev)
    elif case=="frontier_keys": x["next_untouched_frontier"].pop("qa")
    else: x["next_untouched_frontier"][case.split("_",1)[0]]+=1
    path.write_text(json.dumps(x))
    with pytest.raises(c7.C7Error): c7.validate_holdout_manifest(x,path,development_sources=ds,development_result=dev)


def test_rejected_to_accepted_frontier_is_auditable(tmp_path):
    ds,dev,x,path=_holdout(tmp_path); base=x["selected_sources"][0]; duplicate=c7.validate_development_manifest(c7._json(ds),ds)[0]["input_ids_sha256"]
    proof={"field_exists":True,"is_python_string":True,"at_least_8192":True}; bad={"field_exists":False,"is_python_string":True,"at_least_8192":True}
    inspections={}
    for n,f in enumerate(c7.c4.FAMILIES):
        i=c7.FRONTIER[f]; source={"family":f,**c7.c4.SOURCE_SPECS[f],"identity_kind":"dataset_index","dataset_index":i}
        inspections[f]=[{"dataset_index":i,"proof":bad,"input_ids_sha256":f"{40+n:064x}","source":source},{"dataset_index":i+1,"proof":proof,"input_ids_sha256":duplicate,"source":{**source,"dataset_index":i+1}},{"dataset_index":i+2,"proof":proof,"input_ids_sha256":f"{50+n:064x}","source":{**source,"dataset_index":i+2}}]
    result=c7.select_holdout_from_inspections(inspections,[r["input_ids_sha256"] for r in c7.validate_development_manifest(c7._json(ds),ds)],development_result_sha256=c7.sha256_path(dev),schedule_sha256=c7._json(dev)["schedule_sha256"],development_source_manifest_sha256=c7.sha256_path(ds));result["selected_target"]=c7._json(dev)["selected_target"]
    p=tmp_path/"sequenced.json";p.write_text(json.dumps(result));c7.validate_holdout_manifest(result,p,development_sources=ds,development_result=dev)
    assert all(len(result["rejected_sources"][f])==2 and result["next_untouched_frontier"][f]==c7.FRONTIER[f]+3 for f in c7.c4.FAMILIES)


def _holdout_capture(tmp_path):
    ds,dev,hold,hp=_holdout(tmp_path); root=tmp_path/"capture";root.mkdir();records=[]
    for source in hold["selected_sources"]:
        identity=f"{source['family']}:{source['dataset_index']}:holdout"
        for layer in c7.c4.LAYERS:
            ar=f"a/{identity.replace(':','_')}_{layer}.bin";pr=f"p/{identity.replace(':','_')}_{layer}.json";a=root/ar;p=root/pr;a.parent.mkdir(exist_ok=True);p.parent.mkdir(exist_ok=True);a.write_bytes(b"payload")
            detail={"schema_version":"cascadekv-phi35-8k-c7-artifact-v1","protocol_sha256":c7.sha256_path(c7.PROTOCOL),"qualification_sha256":c7.c4.QUALIFICATION_SHA,"backend_id":c7.c4.BACKEND_ID,"identity":identity,"layer":layer,"input_ids_sha256":source["input_ids_sha256"],"artifact_sha256":_digest(b"payload"),"source":{**{k:source[k] for k in ("family","dataset","config","split","revision","field","identity_kind","dataset_index")},"role":"holdout"},"target":{"model":c7.c4.MODEL,"revision":c7.c4.REVISION,"tokenizer_revision":c7.c4.REVISION},"q_shape":list(c7.c4.SHAPE),"k_shape":list(c7.c4.SHAPE),"v_shape":list(c7.c4.SHAPE),"storage_dtype":"float16","model_compute_dtype":"float16","attention_implementation":"sdpa","use_cache":False,"quantization":"none","capture_adapter":"phi3-post-rope-qkv-v1"};p.write_text(json.dumps(detail));records.append({"identity":identity,"layer":layer,"artifact_relative_path":ar,"artifact_sha256":_digest(b"payload"),"provenance_relative_path":pr,"provenance_sha256":_digest(p.read_bytes()),"input_ids_sha256":source["input_ids_sha256"],"source":{"family":source["family"],"dataset_index":source["dataset_index"],"role":"holdout"}})
    m={"schema_version":"cascadekv-phi35-8k-c7-capture-v1","kind":"holdout","protocol_sha256":c7.sha256_path(c7.PROTOCOL),"qualification_sha256":c7.c4.QUALIFICATION_SHA,"backend_id":c7.c4.BACKEND_ID,"development_source_manifest_sha256":None,"holdout_manifest_sha256":c7.sha256_path(hp),"development_result_sha256":c7.sha256_path(dev),"schedule_sha256":hold["schedule_sha256"],"artifact_count":15,"forwards":3,"artifacts":records};mp=root/"manifest.json";mp.write_text(json.dumps(m));return root,mp,ds,hp,dev,m


@pytest.mark.parametrize("case",[
    "protocol","qualification","backend","hold_sha","dev_sha","schedule","forwards2","forwards4","count14","count16",
    "missing_source","extra_source","missing_layer","extra_layer","duplicate","nonselected","family","role","index","hash","artifact_sha","provenance_sha",
    "missing_artifact","missing_provenance","changed_artifact","changed_provenance","traversal_artifact","absolute_provenance","symlink_artifact","symlink_provenance",
    # Every capture provenance field is independently bound.  In particular,
    # q/k/v are deliberately distinct cases, not one shape surrogate.
    "detail_schema_version","detail_protocol_sha256","detail_qualification_sha256","detail_backend_id","detail_identity","detail_layer","detail_input_ids_sha256","detail_artifact_sha256",
    "source_family","source_dataset","source_config","source_split","source_revision","source_field","source_identity_kind","source_dataset_index","source_role",
    "target_model","target_revision","target_tokenizer_revision",
    "q_shape","k_shape","v_shape",
    "storage_dtype","model_compute_dtype","attention_implementation","use_cache","quantization","capture_adapter",
])
def test_holdout_resolver_matrix_1_valid_plus_59_negatives_fail_before_payload(tmp_path,monkeypatch,case):
    root,mp,ds,hp,dev,m=_holdout_capture(tmp_path/case);opened=[];monkeypatch.setattr(c7.TensorAccess,"open",lambda self,record:opened.append(record))
    assert len(c7.CaptureResolver(root,mp,"holdout",development_source_manifest=ds,holdout_manifest=hp,development_sha=c7.sha256_path(dev),schedule_sha=c7._json(dev)["schedule_sha256"]).ordered())==15
    r=m["artifacts"][0]
    top={"protocol":"protocol_sha256","qualification":"qualification_sha256","backend":"backend_id","hold_sha":"holdout_manifest_sha256","dev_sha":"development_result_sha256","schedule":"schedule_sha256"}
    if case in top:m[top[case]]="0"*64
    elif case.startswith("forwards"):m["forwards"]=int(case[-1])
    elif case.startswith("count"):m["artifact_count"]=int(case[-2:])
    elif case=="missing_source":m["artifacts"]=[x for x in m["artifacts"] if not x["identity"].startswith("narrative")]
    elif case=="extra_source":m["artifacts"].append(copy.deepcopy(r))
    elif case=="missing_layer":m["artifacts"].pop()
    elif case=="extra_layer":m["artifacts"].append(copy.deepcopy(r))
    elif case=="duplicate":m["artifacts"][1]["identity"]=r["identity"];m["artifacts"][1]["layer"]=r["layer"]
    elif case=="nonselected":r["identity"]="narrative:999:holdout"
    elif case=="family":r["source"]["family"]="report"
    elif case=="role":r["source"]["role"]="development"
    elif case=="index":r["source"]["dataset_index"]=999
    elif case=="hash":r["input_ids_sha256"]="0"*64
    elif case in {"artifact_sha","provenance_sha"}:r["artifact_sha256" if case=="artifact_sha" else "provenance_sha256"]="x"*64
    elif case.startswith("missing_"):(root/r["artifact_relative_path" if case=="missing_artifact" else "provenance_relative_path"]).unlink()
    elif case.startswith("changed_"):(root/r["artifact_relative_path" if case=="changed_artifact" else "provenance_relative_path"]).write_bytes(b"changed")
    elif case=="traversal_artifact":r["artifact_relative_path"]="../x"
    elif case=="absolute_provenance":r["provenance_relative_path"]="/x"
    elif case.startswith("symlink_"):
        p=root/r["artifact_relative_path" if case=="symlink_artifact" else "provenance_relative_path"];q=root/"link";q.write_bytes(p.read_bytes());p.unlink();p.symlink_to(q)
    else:
        provenance={
            "detail_schema_version":(None,"schema_version","bad"),"detail_protocol_sha256":(None,"protocol_sha256","0"*64),"detail_qualification_sha256":(None,"qualification_sha256","0"*64),"detail_backend_id":(None,"backend_id","bad"),"detail_identity":(None,"identity","bad"),"detail_layer":(None,"layer",999),"detail_input_ids_sha256":(None,"input_ids_sha256","0"*64),"detail_artifact_sha256":(None,"artifact_sha256","0"*64),
            "source_family":("source","family","bad"),"source_dataset":("source","dataset","bad"),"source_config":("source","config","bad"),"source_split":("source","split","bad"),"source_revision":("source","revision","bad"),"source_field":("source","field","bad"),"source_identity_kind":("source","identity_kind","bad"),"source_dataset_index":("source","dataset_index",999),"source_role":("source","role","bad"),
            "target_model":("target","model","bad"),"target_revision":("target","revision","bad"),"target_tokenizer_revision":("target","tokenizer_revision","bad"),
            "q_shape":(None,"q_shape",[]),"k_shape":(None,"k_shape",[]),"v_shape":(None,"v_shape",[]),
            "storage_dtype":(None,"storage_dtype","bad"),"model_compute_dtype":(None,"model_compute_dtype","bad"),"attention_implementation":(None,"attention_implementation","bad"),"use_cache":(None,"use_cache",True),"quantization":(None,"quantization","bad"),"capture_adapter":(None,"capture_adapter","bad"),
        }
        outer,key,value=provenance[case]
        _rewrite_provenance(root,m,r,lambda d: d.__setitem__(key,value) if outer is None else d[outer].__setitem__(key,value))
    bad=root/"bad.json";bad.write_text(json.dumps(m))
    with pytest.raises(c7.C7Error):c7.CaptureResolver(root,bad,"holdout",development_source_manifest=ds,holdout_manifest=hp,development_sha=c7.sha256_path(dev),schedule_sha=c7._json(dev)["schedule_sha256"])
    assert opened==[]


def _audit_root(tmp_path, go):
    root=tmp_path/"root";root.mkdir(parents=True);(root/"protocol.json").write_bytes(c7.PROTOCOL.read_bytes());(root/"runtime_manifest.json").write_bytes(c7.RUNTIME.read_bytes());(root/"qualification.json").write_bytes((c7.ROOT/"artifacts/c5-v2/recovered/files/qualification.json").read_bytes());c7.build_development_manifest(output=root/"development_sources.json");(root/"development_capture_manifest.json").write_text("{}")
    _,_,d=_frozen_go(tmp_path/"seed")
    if not go:
        for value in d["development_target_metrics"].values(): value["mean_relative_l2"]=1.
        d["development_passing_targets"]=c7.development_passing_targets(d);d["classification"]="C7-DEVELOPMENT-NO-GO";d["selected_target"]=None
    d.update({"protocol_sha256":c7.sha256_path(c7.PROTOCOL),"runtime_manifest_sha256":c7.sha256_path(c7.RUNTIME),"qualification_sha256":c7.sha256_path(root/"qualification.json"),"development_source_manifest_sha256":c7.sha256_path(root/"development_sources.json"),"capture_manifest_sha256":c7.sha256_path(root/"development_capture_manifest.json")});(root/"development.json").write_text(json.dumps(d))
    if go:
        selected=[]
        for n,f in enumerate(c7.c4.FAMILIES):selected.append({"family":f,**c7.c4.SOURCE_SPECS[f],"identity_kind":"dataset_index","dataset_index":c7.FRONTIER[f],"role":"holdout","input_ids_sha256":f"{90+n:064x}"})
        h={"schema_version":"cascadekv-phi35-8k-c7-holdout-selection-v1","protocol_sha256":c7.sha256_path(c7.PROTOCOL),"development_source_manifest_sha256":c7.sha256_path(root/"development_sources.json"),"development_result_sha256":c7.sha256_path(root/"development.json"),"schedule_sha256":d["schedule_sha256"],"selected_target":d["selected_target"],"selection_rule":"first mechanically eligible globally-new exact input SHA in ascending order","selected_sources":selected,"rejected_sources":{f:[] for f in c7.c4.FAMILIES},"next_untouched_frontier":{x["family"]:x["dataset_index"]+1 for x in selected},"raw_text_persisted":False};(root/"holdout.json").write_text(json.dumps(h));(root/"holdout_capture_manifest.json").write_text("{}")
        gates={t:{"passes":False} for t in c7.TARGETS};gates[d["selected_target"]]={"passes":True};test={"protocol_sha256":c7.sha256_path(c7.PROTOCOL),"runtime_manifest_sha256":c7.sha256_path(c7.RUNTIME),"development_result_sha256":c7.sha256_path(root/"development.json"),"holdout_manifest_sha256":c7.sha256_path(root/"holdout.json"),"capture_manifest_sha256":c7.sha256_path(root/"holdout_capture_manifest.json"),"selected_target":d["selected_target"],"development_passing_targets":d["development_passing_targets"],"per_gate_pass_fail":gates,"first_passing_target_descriptive":c7.TARGETS[0],"firewall_evidence":{"opened_development_tensor_identities":[],"optimizer_rerun":False,"schedule_sha256_before":d["schedule_sha256"],"schedule_sha256_after":d["schedule_sha256"],"schedule_unchanged":True},"classification":"C7-PROSPECTIVE-PASS"};(root/"test.json").write_text(json.dumps(test))
    return root


def test_final_audit_valid_no_go_and_go_roots(tmp_path):
    assert c7.final_audit(_audit_root(tmp_path/"no",False))=="C7-DEVELOPMENT-NO-GO"
    assert c7.final_audit(_audit_root(tmp_path/"go",True))=="C7-PROSPECTIVE-PASS"


# Selected-target A--H is intentionally exercised through the frozen
# development/holdout/final-audit production chain, never via an isolated
# prospective_verdict call.
def _go_evidence(root):
    return (json.loads((root/"development.json").read_text()),
            json.loads((root/"holdout.json").read_text()),
            json.loads((root/"test.json").read_text()))


def _write_go_evidence(root, dev, hold, test):
    (root/"development.json").write_text(json.dumps(dev))
    hold["development_result_sha256"]=_digest((root/"development.json").read_bytes())
    (root/"holdout.json").write_text(json.dumps(hold))
    test["development_result_sha256"]=_digest((root/"development.json").read_bytes())
    test["holdout_manifest_sha256"]=_digest((root/"holdout.json").read_bytes())
    (root/"test.json").write_text(json.dumps(test))


def test_selected_target_a_frozen_development_to_final_audit_passes(tmp_path):
    root=_audit_root(tmp_path,True);dev,_,test=_go_evidence(root)
    assert dev["selected_target"]=="T0" and "T0" in dev["development_passing_targets"]
    assert test["per_gate_pass_fail"]["T0"]["passes"] is True
    assert c7.final_audit(root)=="C7-PROSPECTIVE-PASS"


def test_selected_target_b_and_h_selected_failure_wins_over_descriptive_pass(tmp_path):
    root=_audit_root(tmp_path,True);dev,hold,test=_go_evidence(root)
    assert dev["selected_target"]=="T0"
    test["per_gate_pass_fail"]["T0"]={"passes":False}
    test["per_gate_pass_fail"]["T1"]={"passes":True}
    test["first_passing_target_descriptive"]="T1"
    test["classification"]="C7-PROSPECTIVE-NO-PASS"
    _write_go_evidence(root,dev,hold,test)
    assert c7.final_audit(root)=="C7-PROSPECTIVE-NO-PASS"


def test_selected_target_c_missing_selected_metrics_rejects_in_final_audit(tmp_path):
    root=_audit_root(tmp_path,True);dev,hold,test=_go_evidence(root)
    test["per_gate_pass_fail"].pop("T0")
    _write_go_evidence(root,dev,hold,test)
    with pytest.raises(c7.C7Error): c7.final_audit(root)


def test_selected_target_d_holdout_selected_target_mismatch_rejects_production_binding(tmp_path):
    root=_audit_root(tmp_path,True);dev,hold,test=_go_evidence(root)
    hold["selected_target"]="T1"
    _write_go_evidence(root,dev,hold,test)
    with pytest.raises(c7.C7Error): c7.final_audit(root)


def test_selected_target_e_schedule_mismatch_rejects_holdout_and_final_bindings(tmp_path):
    root=_audit_root(tmp_path,True);dev,hold,test=_go_evidence(root)
    hold["schedule_sha256"]="0"*64
    _write_go_evidence(root,dev,hold,test)
    with pytest.raises(c7.C7Error): c7.validate_holdout_manifest(hold,root/"holdout.json",development_sources=root/"development_sources.json",development_result=root/"development.json")
    with pytest.raises(c7.C7Error): c7.final_audit(root)
    root=_audit_root(tmp_path/"firewall",True);dev,hold,test=_go_evidence(root)
    test["firewall_evidence"]["schedule_sha256_before"]="0"*64
    _write_go_evidence(root,dev,hold,test)
    with pytest.raises(c7.C7Error): c7.final_audit(root)


def test_selected_target_f_development_result_sha_mismatch_rejects(tmp_path):
    root=_audit_root(tmp_path,True);dev,hold,test=_go_evidence(root)
    _write_go_evidence(root,dev,hold,test)
    hold["development_result_sha256"]="0"*64
    (root/"holdout.json").write_text(json.dumps(hold))
    test["holdout_manifest_sha256"]=_digest((root/"holdout.json").read_bytes())
    (root/"test.json").write_text(json.dumps(test))
    with pytest.raises(c7.C7Error): c7.final_audit(root)


def test_selected_target_g_selected_target_absent_from_frozen_passing_targets_rejects(tmp_path):
    root=_audit_root(tmp_path,True);dev,hold,test=_go_evidence(root)
    dev["development_passing_targets"]=[x for x in dev["development_passing_targets"] if x!="T0"]
    _write_go_evidence(root,dev,hold,test)
    with pytest.raises(c7.C7Error): c7.final_audit(root)


def test_selected_target_h_selected_pass_ignores_descriptive_first_pass(tmp_path):
    root=_audit_root(tmp_path,True);dev,hold,test=_go_evidence(root)
    test["per_gate_pass_fail"]["T0"]={"passes":True}
    test["per_gate_pass_fail"]["T1"]={"passes":True}
    test["first_passing_target_descriptive"]="T1"
    test["classification"]="C7-PROSPECTIVE-PASS"
    _write_go_evidence(root,dev,hold,test)
    assert c7.final_audit(root)=="C7-PROSPECTIVE-PASS"


@pytest.mark.parametrize("case",["no_holdout","no_capture","no_test","protocol","runtime","qualification","dev_protocol","dev_runtime","dev_qualification","dev_sources","dev_capture","dev_passes","dev_class","test_class","test_target","hold_target","remove_passing","test_dev_sha","hold_dev_sha","test_hold_sha","hold_schedule","test_capture","test_protocol","test_runtime","before","after","rerun","changed","opened"])
def test_final_audit_29_corruptions_reject(tmp_path,case):
    root=_audit_root(tmp_path/case,False if case.startswith("no_") else True)
    if case=="no_holdout":(root/"holdout.json").write_text("{}")
    elif case=="no_capture":(root/"holdout_capture_manifest.json").write_text("{}")
    elif case=="no_test":(root/"test.json").write_text("{}")
    elif case in {"protocol","runtime","qualification"}:(root/{"protocol":"protocol.json","runtime":"runtime_manifest.json","qualification":"qualification.json"}[case]).write_bytes(b"bad")
    else:
        file="development.json" if case.startswith("dev_") or case=="remove_passing" else ("holdout.json" if case.startswith("hold_") else "test.json");x=json.loads((root/file).read_text())
        if case=="dev_protocol":x["protocol_sha256"]="0"*64
        elif case=="dev_runtime":x["runtime_manifest_sha256"]="0"*64
        elif case=="dev_qualification":x["qualification_sha256"]="0"*64
        elif case=="dev_sources":x["development_source_manifest_sha256"]="0"*64
        elif case=="dev_capture":x["capture_manifest_sha256"]="0"*64
        elif case=="dev_passes":x["development_passing_targets"]=[]
        elif case=="dev_class":x["classification"]="C7-DEVELOPMENT-NO-GO"
        elif case=="test_class":x["classification"]="C7-PROSPECTIVE-NO-PASS"
        elif case in {"test_target","hold_target"}:x["selected_target"]="T1"
        elif case=="remove_passing":x["development_passing_targets"]=[]
        elif case in {"test_dev_sha","hold_dev_sha"}:x["development_result_sha256"]="0"*64
        elif case=="test_hold_sha":x["holdout_manifest_sha256"]="0"*64
        elif case=="hold_schedule":x["schedule_sha256"]="0"*64
        elif case=="test_capture":x["capture_manifest_sha256"]="0"*64
        elif case=="test_protocol":x["protocol_sha256"]="0"*64
        elif case=="test_runtime":x["runtime_manifest_sha256"]="0"*64
        elif case in {"before","after","rerun","changed","opened"}:
            k={"before":"schedule_sha256_before","after":"schedule_sha256_after","rerun":"optimizer_rerun","changed":"schedule_unchanged","opened":"opened_development_tensor_identities"}[case];x["firewall_evidence"][k]=(["development"] if case=="opened" else (True if case=="rerun" else False if case=="changed" else "0"*64))
        (root/file).write_text(json.dumps(x))
    with pytest.raises((c7.C7Error,c7.c4.C4Error)):c7.final_audit(root)
