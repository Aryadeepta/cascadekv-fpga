"""Synthetic C5 protocol tests; intentionally no external scientific imports."""
import copy, hashlib, json
import random
import sys
import types
from pathlib import Path
import pytest
from cascadekv import phi35_8k_c4 as c4
from cascadekv import phi35_8k_c5 as c5

def test_parent_actions_profiles_and_inventory_are_frozen():
    p=c5.validate_protocol()
    assert p["c4_result_parent"] == c5.C4_PARENT
    assert list(p["methods"]["actions"]) == list(c5.ACTIONS)
    assert p["methods"]["actions"] == c5.action_definitions()
    assert {x:p["methods"]["actions"][x] for x in c4.ACTIONS} == c4._protocol_actions()
    assert p["methods"]["actions"]["A7"]["candidate_fraction"] == .10
    assert p["methods"]["actions"]["A8"]["candidate_fraction"] == .15
    assert p["methods"]["actions"]["A9"]["candidate_fraction"] == .20
    assert p["development_sources"]["identities"] == {family:list(indices) for family,indices in c5.DEVELOPMENT.items()}
    assert "qa:14" in p["development_sources"]["forbidden"]
    assert p["prospective_frontier"]["start_indices"] == c5.FRONTIER

def test_protocol_rejects_profile_tuning_action_or_frontier_change():
    for path,value in ((["methods","routing_profiles","phi35-8k-depth-transfer-qwen10-v1","reserve","sink"],33),(["methods","actions","A7","candidate_fraction"],.11),(["prospective_frontier","start_indices","qa"],17),(["quality_gates","mean_relative_l2_lte"],.121)):
        x=copy.deepcopy(c5._json(c5.PROTOCOL)); y=x
        for part in path[:-1]: y=y[part]
        y[path[-1]]=value
        with pytest.raises(c5.C5Error): c5.validate_protocol(x)

def _stat(t,e,c): return {"total_kv_bytes":t,"relative_l2":e,"cosine":c}
def test_ten_action_dp_and_old_seven_action_regression():
    rows=[{a:_stat(1,.2,.9) for a in c5.ACTIONS} for _ in range(2)]
    rows[0]["A9"]=_stat(1,.01,1); rows[1]["A8"]=_stat(1,.01,1)
    assert c5.optimize_action_cells(rows,2000)[3] == ("A9","A8")
    old=[{a:rows[i][a] for a in c4.ACTIONS} for i in range(2)]
    assert c5.optimize_action_cells(old,2000,c4.ACTIONS) == c5.optimize_action_cells(old,2000,tuple(c4.ACTIONS))

def _result(pass_gate):
    metric={"mean_cosine":.99,"mean_relative_l2":.1 if pass_gate else .13,"mean_total_kv_bytes":9}
    schedules={t:{"feasible":True,"target_mean_kv_bytes":1.,"strict":False,"integer_microbyte_cap":1000,"integer_microbytes_used":1000,"table":{}} for t in c5.TARGETS}
    return {"schedules":schedules,"development_target_metrics":{t:metric for t in c5.TARGETS},"development_baseline_metrics":{"flat5":{"mean_total_kv_bytes":10},"uniform10":{"mean_relative_l2":.2}},"schedule_sha256":c5.schedule_digest(schedules)}
def test_no_go_never_selects_and_go_requires_frozen_result_schedule():
    no=c5.freeze_development(_result(False)); assert no["classification"] == "C5-DEVELOPMENT-NO-GO"
    with pytest.raises(c5.C5Error): c5.assert_holdout_may_start(no,"b"*64)
    yes=c5.freeze_development(_result(True)); assert yes["selected_target"] == "T0"
    assert c5.assert_holdout_may_start(yes,"b"*64) == "T0"

def test_prospective_firewall_and_classifications_are_explicit():
    assert c5.audit_prospective_test(opened_development=[],opened_holdout=["x"],optimizer_rerun=False,schedule_before="a",schedule_after="a")["schedule_unchanged"]
    with pytest.raises(c5.C5Error): c5.audit_prospective_test(opened_development=["d"],opened_holdout=[],optimizer_rerun=False,schedule_before="a",schedule_after="a")

@pytest.mark.parametrize("selected_passes,classification,first",[(False,"C5-PROSPECTIVE-NO-PASS","T3"),(True,"C5-PROSPECTIVE-PASS","T2")])
def test_prospective_target_lock_and_optimizer_firewall(monkeypatch,tmp_path,selected_passes,classification,first):
    """The holdout evaluator may describe later winners but must decide T2 only."""
    result=_result(True)
    # Make T2 the frozen development choice, while retaining later feasible schedules.
    for target in ("T0","T1"): result["schedules"][target]["feasible"]=False
    result["schedule_sha256"]=c5.schedule_digest(result["schedules"])
    result=c5.freeze_development(result); result["selected_target"]="T2"
    dev=tmp_path/"development.json"; c5.atomic_json(dev,result); devsha=c5.sha256_path(dev)
    holdout=tmp_path/"holdout.json"; c5.atomic_json(holdout,{"development_result_sha256":devsha,"schedule_sha256":result["schedule_sha256"],"selected_target":"T2","selected_sources":[]})
    capture=tmp_path/"capture.json"; capture.write_text("{}")
    class Resolver:
        kind="holdout"
        def __init__(self,*a,**kw): pass
    class Access:
        def __init__(self,*a,**kw): self.opened=c5._expected_tensor_ids("holdout",json.loads(holdout.read_text()))
    monkeypatch.setattr(c5,"CaptureResolver",Resolver); monkeypatch.setattr(c5,"TensorAccess",Access)
    def rows(*args,**kwargs):
        def sample(ok): return {"metrics":{"cosine_similarity":.99,"relative_l2_error":.1 if ok else .3},"traffic":{"total_kv_bytes":9.}}
        out={baseline:[sample(True)]*1440 for baseline in c4.BASELINES}
        out["flat5"]=[{"metrics":{"cosine_similarity":.99,"relative_l2_error":.2},"traffic":{"total_kv_bytes":10.}}]*1440
        out["uniform10"]=[{"metrics":{"cosine_similarity":.99,"relative_l2_error":.2},"traffic":{"total_kv_bytes":9.}}]*1440
        for target in ("T2","T3","T4","T5"): out[target]=[sample(selected_passes if target=="T2" else True)]*1440
        return out
    monkeypatch.setattr(c5,"_method_rows",rows)
    for name in ("optimize_action_cells","optimize_static_schedule","_development_schedules"):
        monkeypatch.setattr(c5,name,lambda *a,**kw:(_ for _ in ()).throw(AssertionError("optimizer reached during test")))
    out=tmp_path/"test.json"; observed=c5.prospective_test(holdout_capture_root=tmp_path,holdout_capture_manifest=capture,holdout_manifest=holdout,development_result=dev,output=out)
    assert observed["classification"]==classification and observed["first_passing_target_descriptive"]==first
    evidence=observed["firewall_evidence"]
    assert evidence["opened_development_tensor_identities"]==[] and evidence["optimizer_rerun"] is False
    assert evidence["schedule_sha256_before"]==evidence["schedule_sha256_after"] and evidence["schedule_unchanged"] is True

def test_c5_seven_action_dp_is_bit_for_bit_c4_over_ties_caps_and_rounding():
    rng=random.Random(713)
    for case in range(40):
        groups=[]
        for _cell in range(1+rng.randrange(5)):
            row={}
            for action in c4.ACTIONS:
                # Deliberately use traffic near integer microbyte boundaries and
                # repeated error/cosine values to exercise lexical tie handling.
                row[action]=_stat(rng.choice([.9995,1.0005,2.4995,3.5005]),rng.choice([.01,.02,.02]),rng.choice([.9,.95,.95]))
            groups.append(row)
        for cap in (0,999,1000,2000,5000,10000):
            assert c5.optimize_action_cells(groups,cap,tuple(c4.ACTIONS)) == c4._optimize_action_cells(groups,cap)

def test_rows_for_schedule_selects_one_correct_action_per_observation_and_is_ordered():
    rows={a:[] for a in c5.ACTIONS}; table={f"{layer}:{head}":c5.ACTIONS[(layer+head)%len(c5.ACTIONS)] for layer in c4.LAYERS for head in c4.HEADS}
    for family in c4.FAMILIES:
        for index in c5.DEVELOPMENT[family]:
            for layer in c4.LAYERS:
                for position in c4.POSITIONS:
                    for head in c4.HEADS:
                        for action in c5.ACTIONS:
                            rows[action].append({"source":f"{family}:{index}:development","layer":layer,"position":position,"head":head,"metrics":{"cosine_similarity":float(c5.ACTIONS.index(action)),"relative_l2_error":0.},"traffic":{"total_kv_bytes":1}})
    for values in rows.values(): values.reverse()
    selected=c5.rows_for_schedule(rows,table)
    assert len(selected)==5760
    assert [r["source"] for r in selected] == sorted(r["source"] for r in selected)
    assert all(r["metrics"]["cosine_similarity"] == c5.ACTIONS.index(table[f"{r['layer']}:{r['head']}"]) for r in selected)
    with pytest.raises(c5.C5Error): c5.rows_for_schedule(rows,{**table,"bad":"A0"})
    bad=copy.deepcopy(rows); bad["A0"].append(dict(bad["A0"][0]))
    with pytest.raises(c5.C5Error): c5.rows_for_schedule(bad,table)


# Capture binding matrices deliberately construct manifests without payload files:
# CaptureResolver must reject every bad binding before TensorAccess can be used.
SENTINEL="SHOULD_NOT_PERSIST_RAW_SOURCE_TEXT_9F2C"
def _h(x): return hashlib.sha256(x.encode()).hexdigest()
def _source(family,index,role):
    return {"family":family,"dataset":"d","config":"c","split":"s","revision":"r","field":"f","identity_kind":"dataset_index","dataset_index":index,"role":role,"input_ids_sha256":_h(f"{family}:{index}")}
def _record(source,layer):
    identity=f"{source['family']}:{source['dataset_index']}:{source['role']}"
    return {"identity":identity,"layer":layer,"artifact_relative_path":f"artifacts/{identity}-{layer}.safetensors","artifact_sha256":_h("a"+identity+str(layer)),"provenance_relative_path":f"artifacts/{identity}-{layer}.json","provenance_sha256":_h("p"+identity+str(layer)),"input_ids_sha256":source["input_ids_sha256"],"source":{"family":source["family"],"dataset_index":source["dataset_index"],"role":source["role"]}}
def _write(path,value): path.write_text(json.dumps(value,sort_keys=True)); return path
def _capture_fixture(tmp_path,kind):
    if kind=="development":
        sources=[]
        for f in c4.FAMILIES:
            for i in c5.DEVELOPMENT[f]:
                row=_source(f,i,"development"); row.update({k:c4.SOURCE_SPECS[f][k] for k in ("dataset","config","split","revision","field")}); sources.append(row)
        persisted=[{k:v for k,v in source.items() if k!="role"} for source in sources]
        bound=_write(tmp_path/"development_sources.json",{"schema_version":"cascadekv-phi35-8k-c5-development-sources-v1","protocol_sha256":c5.sha256_path(c5.PROTOCOL),"target":{"model":c4.MODEL,"model_revision":c4.REVISION,"tokenizer_revision":c4.REVISION},"tokenization_contract":{"add_special_tokens":True,"truncation":True,"max_length":8192,"required_input_ids_length":8192,"canonicalization":"CPU contiguous int64 bytes"},"source_specs":c4.SOURCE_SPECS,"selected_sources":persisted,"rejected_sources":{f:[] for f in c4.FAMILIES},"global_input_ids_sha256_unique":True,"source_text_stored":False})
        assert len(c5.validate_development_manifest(c5._json(bound),bound))==12
        kw={"development_source_manifest":bound}; extra={"development_source_manifest_sha256":c5.sha256_path(bound),"holdout_manifest_sha256":None,"development_result_sha256":None,"schedule_sha256":None}
    else:
        sources=[_source(f,100+n,"holdout") for n,f in enumerate(c4.FAMILIES)]
        bound=_write(tmp_path/"holdout.json",{"selected_sources":sources})
        kw={"holdout_manifest":bound,"development_sha":_h("development"),"schedule_sha":_h("schedule")}; extra={"development_source_manifest_sha256":None,"holdout_manifest_sha256":c5.sha256_path(bound),"development_result_sha256":kw["development_sha"],"schedule_sha256":kw["schedule_sha"]}
    rows=[_record(s,l) for s in sources for l in c4.LAYERS]
    payload={"schema_version":"cascadekv-phi35-8k-c5-capture-v1","kind":kind,"protocol_sha256":c5.sha256_path(c5.PROTOCOL),"qualification_sha256":c4.QUALIFICATION_SHA,"backend_id":c4.BACKEND_ID,**extra,"artifact_count":len(rows),"forwards":len(sources),"artifacts":rows}
    manifest=_write(tmp_path/f"{kind}_capture.json",payload)
    return manifest,bound,payload,kw
def _resolver_never_opens(monkeypatch):
    called=[]
    monkeypatch.setattr(c5.TensorAccess,"open",lambda *_: called.append(True) or (_ for _ in ()).throw(AssertionError("payload opened")))
    return called
@pytest.mark.parametrize("mutation",[
    lambda p:p.__setitem__("development_source_manifest_sha256","0"*64), lambda p:p.__setitem__("protocol_sha256","0"*64), lambda p:p.__setitem__("qualification_sha256","0"*64), lambda p:p.__setitem__("backend_id","wrong"), lambda p:p.__setitem__("artifact_count",59), lambda p:p.__setitem__("forwards",11),
    lambda p:p["artifacts"].pop(), lambda p:p["artifacts"].append(copy.deepcopy(p["artifacts"][0])), lambda p:p["artifacts"].__setitem__(0,{**p["artifacts"][0],"identity":"qa:999:development"}), lambda p:p["artifacts"][0].__setitem__("artifact_sha256","x"*64), lambda p:p["artifacts"][0].__setitem__("provenance_sha256","x"*64), lambda p:p["artifacts"][0].__setitem__("input_ids_sha256","x"*64),
    lambda p:p["artifacts"].__setitem__(0,{**p["artifacts"][0],"layer":99}), lambda p:p["artifacts"].__setitem__(0,{**p["artifacts"][0],"identity":"qa:14:development"}),
])
def test_development_capture_binding_matrix(monkeypatch,tmp_path,mutation):
    manifest,bound,payload,kw=_capture_fixture(tmp_path,"development"); called=_resolver_never_opens(monkeypatch); mutation(payload); _write(manifest,payload)
    with pytest.raises(c5.C5Error): c5.CaptureResolver(tmp_path,manifest,"development",**kw)
    assert not called
def test_development_capture_binding_valid_missing_and_changed(monkeypatch,tmp_path):
    manifest,bound,payload,kw=_capture_fixture(tmp_path,"development"); called=_resolver_never_opens(monkeypatch)
    assert len(c5.CaptureResolver(tmp_path,manifest,"development",**kw).ordered())==60 and not called
    bound.unlink()
    with pytest.raises(c5.C5Error): c5.CaptureResolver(tmp_path,manifest,"development",**kw)
    _write(bound,{"changed":True})
    with pytest.raises(c5.C5Error): c5.CaptureResolver(tmp_path,manifest,"development",**kw)
@pytest.mark.parametrize("mutation",[
    lambda p:p.__setitem__("holdout_manifest_sha256","0"*64), lambda p:p.__setitem__("development_result_sha256","0"*64), lambda p:p.__setitem__("schedule_sha256","0"*64), lambda p:p.__setitem__("protocol_sha256","0"*64), lambda p:p.__setitem__("qualification_sha256","0"*64), lambda p:p.__setitem__("backend_id","wrong"), lambda p:p.__setitem__("development_source_manifest_sha256","0"*64), lambda p:p.__setitem__("artifact_count",14), lambda p:p.__setitem__("forwards",2),
    lambda p:p["artifacts"].pop(), lambda p:p["artifacts"].append(copy.deepcopy(p["artifacts"][0])), lambda p:p["artifacts"].__setitem__(0,{**p["artifacts"][0],"identity":"qa:999:holdout"}), lambda p:p["artifacts"][0].__setitem__("artifact_sha256","x"*64), lambda p:p["artifacts"][0].__setitem__("provenance_sha256","x"*64), lambda p:p["artifacts"][0].__setitem__("input_ids_sha256","x"*64),
])
def test_holdout_capture_binding_matrix(monkeypatch,tmp_path,mutation):
    manifest,bound,payload,kw=_capture_fixture(tmp_path,"holdout"); called=_resolver_never_opens(monkeypatch); mutation(payload); _write(manifest,payload)
    with pytest.raises(c5.C5Error): c5.CaptureResolver(tmp_path,manifest,"holdout",**kw)
    assert not called
def test_holdout_capture_binding_valid_missing_and_changed(monkeypatch,tmp_path):
    manifest,bound,payload,kw=_capture_fixture(tmp_path,"holdout"); called=_resolver_never_opens(monkeypatch)
    assert len(c5.CaptureResolver(tmp_path,manifest,"holdout",**kw).ordered())==15 and not called
    bound.unlink()
    with pytest.raises(c5.C5Error): c5.CaptureResolver(tmp_path,manifest,"holdout",**kw)
    _write(bound,{"changed":True})
    with pytest.raises(c5.C5Error): c5.CaptureResolver(tmp_path,manifest,"holdout",**kw)

def test_capture_finalization_binds_final_artifact_and_provenance(monkeypatch,tmp_path):
    source=_source("qa",13,"development"); source.update({k:c4.SOURCE_SPECS["qa"][k] for k in ("dataset","config","split","revision","field")})
    source["input_ids_sha256"]="input"; output=tmp_path/"capture"; qualification=tmp_path/"q.json"; qualification.write_text("{}"); proof={"field_exists":True,"is_python_string":True,"at_least_8192":True}
    class IDs:
        def contiguous(self): return self
        def to(self,*_): return self
    fake_torch=types.SimpleNamespace(int64=object(),tensor=lambda *_a,**_k:IDs())
    fake_c2=types.SimpleNamespace(_load_pinned_model=lambda:object(),validate_qualification=lambda *_:None,assert_capture_backend=lambda *_:None,_first_device=lambda _:None,capture_five_layers_post_rope_qkv=lambda *_:{l:object() for l in c4.LAYERS})
    def writer(a,p,_captured,detail):
        a.parent.mkdir(parents=True,exist_ok=True); a.write_bytes(b"artifact:"+str(detail["layer"]).encode()); written={**detail,"artifact_sha256":c5.sha256_path(a)}; p.write_text(json.dumps(written,sort_keys=True)); return written
    fake_c2.write_artifact=writer
    fake_select=types.SimpleNamespace(load_frozen_tokenizer=lambda:lambda *_a,**_k:types.SimpleNamespace(input_ids=[1]),load_dataset_for_spec=lambda _:{13:{source["field"]:"ephemeral "+SENTINEL}},mechanical_proof=lambda *_:proof)
    monkeypatch.setattr(c5.c4,"_validate_c2_qualification",lambda _:{"backend_id":c4.BACKEND_ID}); monkeypatch.setattr(c5.c4,"_input_hash",lambda _:"input")
    monkeypatch.setitem(sys.modules,"torch",fake_torch); monkeypatch.setitem(sys.modules,"cascadekv.phi35_kaggle_c2",fake_c2); monkeypatch.setitem(sys.modules,"cascadekv.phi35_8k_source_selection",fake_select)
    final=c5.capture_sources(kind="development",sources=[source],output=output,qualification=qualification,development_source_sha=_h("dev"))
    assert final["artifact_count"]==5 and SENTINEL not in json.dumps(final)
    for row in final["artifacts"]:
        assert row["artifact_sha256"]==c5.sha256_path(output/row["artifact_relative_path"])
        assert row["provenance_sha256"]==c5.sha256_path(output/row["provenance_relative_path"])
        p=json.loads((output/row["provenance_relative_path"]).read_text())
        assert p["artifact_sha256"]==row["artifact_sha256"] and p["identity"]==row["identity"] and p["layer"]==row["layer"] and p["source"]["role"]=="development"
    assert SENTINEL not in (output/"final_capture_manifest.json").read_text()

@pytest.mark.parametrize("bad",["returned","file"])
def test_capture_finalization_fails_closed_on_provenance_disagreement(monkeypatch,tmp_path,bad):
    # Reuse the successful boundary but corrupt exactly one side of C2's writer contract.
    base=globals()["test_capture_finalization_binds_final_artifact_and_provenance"]
    # A focused local reimplementation of its writer setup keeps the negative assertion explicit.
    source=_source("qa",13,"development"); source.update({k:c4.SOURCE_SPECS["qa"][k] for k in ("dataset","config","split","revision","field")}); source["input_ids_sha256"]="input"; q=tmp_path/"q"; q.write_text("{}"); proof={"field_exists":True,"is_python_string":True,"at_least_8192":True}
    ids=types.SimpleNamespace(contiguous=lambda:ids,to=lambda *_:ids); torch=types.SimpleNamespace(int64=object(),tensor=lambda *_a,**_k:ids)
    c2fake=types.SimpleNamespace(_load_pinned_model=lambda:object(),validate_qualification=lambda *_:None,assert_capture_backend=lambda *_:None,_first_device=lambda _:None,capture_five_layers_post_rope_qkv=lambda *_:{l:object() for l in c4.LAYERS})
    def writer(a,p,*args):
        a.parent.mkdir(parents=True,exist_ok=True); a.write_bytes(b"x"); detail=args[-1]; record={**detail,"artifact_sha256":c5.sha256_path(a)}; p.write_text(json.dumps({**record,"artifact_sha256":"0"*64} if bad=="file" else record)); return {**record,"artifact_sha256":"0"*64} if bad=="returned" else record
    c2fake.write_artifact=writer; select=types.SimpleNamespace(load_frozen_tokenizer=lambda:lambda *_a,**_k:types.SimpleNamespace(input_ids=[1]),load_dataset_for_spec=lambda _:{13:{source["field"]:"x"}},mechanical_proof=lambda *_:proof)
    monkeypatch.setattr(c5.c4,"_validate_c2_qualification",lambda _:{}); monkeypatch.setattr(c5.c4,"_input_hash",lambda _:"input"); monkeypatch.setitem(sys.modules,"torch",torch); monkeypatch.setitem(sys.modules,"cascadekv.phi35_kaggle_c2",c2fake); monkeypatch.setitem(sys.modules,"cascadekv.phi35_8k_source_selection",select)
    with pytest.raises(c5.C5Error,match="provenance chain"): c5.capture_sources(kind="development",sources=[source],output=tmp_path/"out",qualification=q,development_source_sha=_h("d"))

RUNTIME_MUTATIONS=("cascadekv/phi35_8k_c5.py","cascadekv/stdout_artifact_recovery.py","scripts/kaggle_phi35_8k_c5.py","cascadekv/evaluation_core.py","cascadekv/adaptive_lifting.py","cascadekv/phi35_kaggle_c2.py","configs/cascadekv_phi35_8k_kaggle_c2_protocol.json","configs/cascadekv_phi35_8k_phaseb_protocol.json","configs/cascadekv_phi35_8k_source_amendment_protocol.json","pyproject.toml","uv.lock")
def _runtime_payload(paths):
    return {"schema_version":1,"purpose":"synthetic","protocol":{"path":str(c5.PROTOCOL.relative_to(c5.ROOT)),"sha256":c5.sha256_path(c5.PROTOCOL)},"bound_files":[{"path":p,"sha256":c5.sha256_path(c5.ROOT/p)} for p in paths]}
@pytest.mark.parametrize("target",RUNTIME_MUTATIONS)
def test_runtime_closure_rejects_each_representative_bound_dependency(monkeypatch,tmp_path,target):
    manifest=tmp_path/"runtime.json"; _write(manifest,_runtime_payload(RUNTIME_MUTATIONS)); original=c5.sha256_path
    monkeypatch.setattr(c5,"RUNTIME",manifest)
    monkeypatch.setattr(c5,"sha256_path",lambda p:"0"*64 if Path(p)==c5.ROOT/target else original(Path(p)))
    with pytest.raises(c5.C5Error): c5.verify_runtime_closure()
@pytest.mark.parametrize("mutation",[
    lambda p:p["protocol"].__setitem__("sha256","0"*64),
    lambda p:p["bound_files"].append(dict(p["bound_files"][0])),
    lambda p:p["bound_files"].__setitem__(0,{"path":"missing.py","sha256":"0"*64}),
    lambda p:p["bound_files"][0].__setitem__("sha256","x"*64),
])
def test_runtime_closure_rejects_manifest_corruption(monkeypatch,tmp_path,mutation):
    manifest=tmp_path/"runtime.json"; payload=_runtime_payload(("cascadekv/phi35_8k_c5.py",)); mutation(payload); _write(manifest,payload); monkeypatch.setattr(c5,"RUNTIME",manifest)
    with pytest.raises(c5.C5Error): c5.verify_runtime_closure()
