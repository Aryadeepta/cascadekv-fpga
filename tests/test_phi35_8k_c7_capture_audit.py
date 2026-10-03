"""Offline adversarial C7 binding tests; no model, tokenizer, or data access."""
import copy
import hashlib
import json
import sys
import types

import pytest

from cascadekv import phi35_8k_c6 as c6
from cascadekv import phi35_8k_c7 as c7


def test_c7_v2_execution_tag_binds_exactly_to_head(monkeypatch):
    """v1 would reject the remediation commit; execution must bind only v2 to HEAD."""
    assert c7.C7_TAG == "cascadekv-phi35-8k-c7-freeze-v2"

    def git(ref):
        if ref == c7.C6_PARENT["result_tag"]: return c7.C6_PARENT["result_commit"]
        if ref == c7.C6_PARENT["freeze_tag"]: return c7.C6_PARENT["freeze_commit"]
        return "future-v2-commit" if ref in (c7.C7_TAG, "HEAD") else "unexpected"

    monkeypatch.setattr(c7, "_git", git)
    monkeypatch.setattr(c7, "validate_protocol", lambda: {})
    monkeypatch.setattr(c7, "verify_runtime_closure", lambda: ())
    assert not c7.preflight(execution=True)["local_only"]

    def stale_v1_git(ref):
        return "v1-commit" if ref == c7.C7_TAG else git(ref)

    monkeypatch.setattr(c7, "_git", stale_v1_git)
    with pytest.raises(c7.C7Error, match="matching freeze tag"):
        c7.preflight(execution=True)


def _tensor_access_fixture(tmp_path,monkeypatch):
    """Build synthetic C7-provenanced files; no model, source, or tensor payload."""
    root=tmp_path/"capture"; artifacts=root/"artifacts"; artifacts.mkdir(parents=True)
    sources=[]
    for family in c7.c4.FAMILIES:
        for index in c7.DEVELOPMENT[family]:
            sources.append({"family":family,**c7.c4.SOURCE_SPECS[family],"identity_kind":"dataset_index","dataset_index":index,"role":"development","input_ids_sha256":hashlib.sha256(f"{family}:{index}".encode()).hexdigest()})
    source_manifest=tmp_path/"development_sources.json"; source_manifest.write_text("{}")
    monkeypatch.setattr(c7,"validate_development_manifest",lambda payload,path: [dict(x) for x in sources])
    records=[]
    for source in sources:
        identity=f"{source['family']}:{source['dataset_index']}:development"
        expected_source={**{k:source[k] for k in ("family","dataset","config","split","revision","field","identity_kind","dataset_index")},"role":"development"}
        for layer in c7.c4.LAYERS:
            stem=f"{source['family']}_{source['dataset_index']}_{layer}"
            artifact_rel=f"artifacts/{stem}.safetensors"; provenance_rel=f"artifacts/{stem}.provenance.json"
            artifact=root/artifact_rel; artifact.write_bytes(f"synthetic:{stem}".encode())
            detail={"schema_version":"cascadekv-phi35-8k-c7-artifact-v1","protocol_sha256":c7.sha256_path(c7.PROTOCOL),"qualification_sha256":c7.c4.QUALIFICATION_SHA,"backend_id":c7.c4.BACKEND_ID,"identity":identity,"layer":layer,"input_ids_sha256":source["input_ids_sha256"],"artifact_sha256":c7.sha256_path(artifact),"source":expected_source,"target":{"model":c7.c4.MODEL,"revision":c7.c4.REVISION,"tokenizer_revision":c7.c4.REVISION},"q_shape":list(c7.c4.SHAPE),"k_shape":list(c7.c4.SHAPE),"v_shape":list(c7.c4.SHAPE),"storage_dtype":"float16","model_compute_dtype":"float16","attention_implementation":"sdpa","use_cache":False,"quantization":"none","capture_adapter":"phi3-post-rope-qkv-v1"}
            provenance=root/provenance_rel; provenance.write_text(json.dumps(detail,sort_keys=True))
            records.append({"identity":identity,"layer":layer,"artifact_relative_path":artifact_rel,"artifact_sha256":c7.sha256_path(artifact),"provenance_relative_path":provenance_rel,"provenance_sha256":c7.sha256_path(provenance),"input_ids_sha256":source["input_ids_sha256"],"source":{"family":source["family"],"dataset_index":source["dataset_index"],"role":"development"}})
    manifest={"schema_version":"cascadekv-phi35-8k-c7-capture-v1","kind":"development","protocol_sha256":c7.sha256_path(c7.PROTOCOL),"qualification_sha256":c7.c4.QUALIFICATION_SHA,"backend_id":c7.c4.BACKEND_ID,"development_source_manifest_sha256":c7.sha256_path(source_manifest),"holdout_manifest_sha256":None,"development_result_sha256":None,"schedule_sha256":None,"artifact_count":90,"forwards":18,"artifacts":records}
    manifest_path=root/"final_capture_manifest.json"; manifest_path.write_text(json.dumps(manifest,sort_keys=True))
    resolver=c7.CaptureResolver(root,manifest_path,"development",development_source_manifest=source_manifest)
    return root,resolver,resolver.ordered()[0]


def test_c7_tensor_access_uses_c7_provenance_and_rejects_post_resolution_tampering(tmp_path,monkeypatch):
    """C7 v1 inherited c6.TensorAccess.open, which compared against c6.PROTOCOL."""
    root,resolver,record=_tensor_access_fixture(tmp_path,monkeypatch)
    assert c7.sha256_path(c7.PROTOCOL)!=c7.sha256_path(c6.PROTOCOL)
    detail=json.loads((root/record["provenance_relative_path"]).read_text())
    source=resolver._authorized_sources[f"{record['identity']}:L{record['layer']}"]
    assert detail["protocol_sha256"]==c7.sha256_path(c7.PROTOCOL)
    assert not c7._c7_artifact_provenance_matches({**detail,"protocol_sha256":c7.sha256_path(c6.PROTOCOL)},record,source)
    assert c7._c7_artifact_provenance_matches(detail,record,source)
    payload_calls=[]
    class Tensor:
        dtype="float16"; shape=c7.c4.SHAPE
    class Handle:
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def keys(self): return ("q","k","v")
        def get_tensor(self,name): return Tensor()
    def safe_open(*args,**kwargs):
        payload_calls.append((args,kwargs)); return Handle()
    monkeypatch.setitem(sys.modules,"safetensors",types.SimpleNamespace(safe_open=safe_open))
    monkeypatch.setitem(sys.modules,"torch",types.SimpleNamespace(float16="float16"))
    access=c7.TensorAccess(resolver,allowed_kind="development")
    assert access.open(record) and access.opened==[f"{record['identity']}:L{record['layer']}"]
    assert len(payload_calls)==1

    provenance=root/record["provenance_relative_path"]
    original_provenance=provenance.read_text(); mutated=json.loads(original_provenance)
    mutated["protocol_sha256"]=c7.sha256_path(c6.PROTOCOL); provenance.write_text(json.dumps(mutated,sort_keys=True))
    # Updating a caller-visible manifest record cannot alter the resolver snapshot.
    resolver.records[f"{record['identity']}:L{record['layer']}"]["provenance_sha256"]=c7.sha256_path(provenance)
    with pytest.raises(c7.C7Error): access.open(resolver.ordered()[0])
    assert len(payload_calls)==1
    provenance.write_text(original_provenance)

    artifact=root/record["artifact_relative_path"]; original_artifact=artifact.read_bytes(); artifact.write_bytes(original_artifact+b"!")
    with pytest.raises(c7.C7Error): access.open(record)
    assert len(payload_calls)==1
    artifact.write_bytes(original_artifact)

    mutated_record=copy.deepcopy(record); mutated_record["layer"]=999
    with pytest.raises(c7.C7Error): access.open(mutated_record)
    assert len(payload_calls)==1

    artifact.unlink(); artifact.symlink_to(root/"artifacts"/"replacement.safetensors")
    (root/"artifacts"/"replacement.safetensors").write_bytes(original_artifact)
    with pytest.raises(c7.C7Error): access.open(record)
    assert len(payload_calls)==1
    artifact.unlink(); artifact.write_bytes(original_artifact)

    provenance.unlink(); provenance.symlink_to(root/"artifacts"/"replacement.provenance.json")
    (root/"artifacts"/"replacement.provenance.json").write_text(original_provenance)
    with pytest.raises(c7.C7Error): access.open(record)
    assert len(payload_calls)==1


def _fake_payload_boundary(monkeypatch):
    """A final-boundary-only stand-in: no tensor file is ever decoded."""
    calls=[]
    class Tensor:
        dtype="float16"; shape=c7.c4.SHAPE
    class Handle:
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def keys(self): return ("q","k","v")
        def get_tensor(self,name): return Tensor()
    def safe_open(*args,**kwargs):
        calls.append((args,kwargs)); return Handle()
    monkeypatch.setitem(sys.modules,"safetensors",types.SimpleNamespace(safe_open=safe_open))
    monkeypatch.setitem(sys.modules,"torch",types.SimpleNamespace(float16="float16"))
    return calls


def test_c7_tensor_access_authorization_snapshot_is_independent(tmp_path,monkeypatch):
    root,resolver,record=_tensor_access_fixture(tmp_path,monkeypatch); calls=_fake_payload_boundary(monkeypatch)
    key=f"{record['identity']}:L{record['layer']}"; access=c7.TensorAccess(resolver,allowed_kind="development")
    resolver.records[key]["source"]["family"]="attacker"
    visible=resolver.ordered()[0]; visible["artifact_sha256"]="0"*64
    assert resolver._authorized_records[key]["source"]["family"] != "attacker"
    with pytest.raises(TypeError): resolver._authorized_records[key]["source"]["family"]="attacker"
    with pytest.raises(TypeError): resolver._authorized_sources[key]["family"]="attacker"
    with pytest.raises(c7.C7Error): access.open(visible)
    assert calls == []
    assert access.open(record) and len(calls)==1


def test_c7_tensor_access_rechecks_artifact_and_provenance_digests(tmp_path,monkeypatch):
    root,resolver,record=_tensor_access_fixture(tmp_path,monkeypatch); calls=_fake_payload_boundary(monkeypatch)
    access=c7.TensorAccess(resolver,allowed_kind="development")
    artifact=root/record["artifact_relative_path"]; original_artifact=artifact.read_bytes()
    artifact.write_bytes(original_artifact+b"!")
    with pytest.raises(c7.C7Error): access.open(record)
    assert calls == []
    artifact.write_bytes(original_artifact)
    assert access.open(record) and len(calls)==1
    provenance=root/record["provenance_relative_path"]; original_provenance=provenance.read_text()
    provenance.write_text(original_provenance+" ")
    with pytest.raises(c7.C7Error): access.open(record)
    assert len(calls)==1
    provenance.write_text(original_provenance)
    assert access.open(record) and len(calls)==2


def test_c7_provenance_predicate_binds_every_c7_field(tmp_path,monkeypatch):
    root,resolver,record=_tensor_access_fixture(tmp_path,monkeypatch)
    detail=json.loads((root/record["provenance_relative_path"]).read_text())
    source=resolver._authorized_sources[f"{record['identity']}:L{record['layer']}"]
    assert c7._c7_artifact_provenance_matches(detail,record,source)
    mutations={
        "protocol_sha256":c7.sha256_path(c6.PROTOCOL), "qualification_sha256":"0"*64,
        "backend_id":"other", "identity":"other", "layer":999,
        "input_ids_sha256":"0"*64, "artifact_sha256":"0"*64,
        "source":{**source,"role":"holdout"}, "target":{"model":"other"},
        "q_shape":[0], "k_shape":[0], "v_shape":[0], "storage_dtype":"float32",
        "model_compute_dtype":"float32", "attention_implementation":"eager",
        "use_cache":True, "quantization":"int8", "capture_adapter":"other",
        "schema_version":"other",
    }
    for field,value in mutations.items():
        assert not c7._c7_artifact_provenance_matches({**detail,field:value},record,source), field


def test_c7_tensor_access_rejects_consistent_visible_c6_substitution_and_path_edges(tmp_path,monkeypatch):
    root,resolver,record=_tensor_access_fixture(tmp_path,monkeypatch); calls=_fake_payload_boundary(monkeypatch)
    access=c7.TensorAccess(resolver,allowed_kind="development"); key=f"{record['identity']}:L{record['layer']}"
    provenance=root/record["provenance_relative_path"]; original=provenance.read_text(); detail=json.loads(original)
    detail["protocol_sha256"]=c7.sha256_path(c6.PROTOCOL); provenance.write_text(json.dumps(detail,sort_keys=True))
    resolver.records[key]["provenance_sha256"]=c7.sha256_path(provenance)
    with pytest.raises(c7.C7Error): access.open(resolver.ordered()[0])
    with pytest.raises(c7.C7Error): access.open(record)
    assert calls == []
    provenance.write_text(original)
    for field,value in (
        ("artifact_relative_path","/tmp/evil"), ("provenance_relative_path","/tmp/evil"),
        ("artifact_relative_path","../evil"), ("provenance_relative_path","../evil"),
        ("identity","other:0:development"), ("layer",999),
        ("artifact_sha256","0"*64), ("provenance_sha256","0"*64),
    ):
        bad=copy.deepcopy(record); bad[field]=value
        with pytest.raises(c7.C7Error): access.open(bad)
    assert calls == []


def test_c7_tensor_access_leaf_symlinks_fail_before_payload(tmp_path,monkeypatch):
    root,resolver,record=_tensor_access_fixture(tmp_path,monkeypatch); calls=_fake_payload_boundary(monkeypatch)
    access=c7.TensorAccess(resolver,allowed_kind="development")
    for field in ("artifact_relative_path","provenance_relative_path"):
        leaf=root/record[field]; saved=leaf.read_bytes(); replacement=leaf.with_name("replacement-"+leaf.name)
        replacement.write_bytes(saved); leaf.unlink(); leaf.symlink_to(replacement)
        with pytest.raises(c7.C7Error): access.open(record)
        assert calls == []


def _manifest(tmp_path):
    path=tmp_path/"development_sources.json"
    c7.build_development_manifest(output=path)
    return path,json.loads(path.read_text())


def test_archived_identity_hash_authority_has_exact_18_pairs(tmp_path):
    path,payload=_manifest(tmp_path)
    rows=c7.validate_development_manifest(payload,path)
    assert len(rows)==18
    assert {(x["family"],x["dataset_index"]):x["input_ids_sha256"] for x in rows}==c7.archived_development_hashes()


@pytest.mark.parametrize("family,index",[(f,i) for f in c7.c4.FAMILIES for i in c7.DEVELOPMENT[f]])
def test_every_archived_identity_rejects_wrong_hash(tmp_path,family,index):
    path,payload=_manifest(tmp_path)
    row=next(x for x in payload["selected_sources"] if (x["family"],x["dataset_index"])==(family,index))
    row["input_ids_sha256"]="0"*64 if row["input_ids_sha256"]!="0"*64 else "1"*64
    with pytest.raises(c7.C7Error): c7.validate_development_manifest(payload,path)


def test_archived_hash_swap_duplicate_and_identity_metadata_reject(tmp_path):
    path,payload=_manifest(tmp_path)
    for mutation in ("swap","duplicate","omitted","extra","wrong_metadata"):
        value=copy.deepcopy(payload)
        if mutation=="swap": value["selected_sources"][0]["input_ids_sha256"],value["selected_sources"][1]["input_ids_sha256"]=value["selected_sources"][1]["input_ids_sha256"],value["selected_sources"][0]["input_ids_sha256"]
        elif mutation=="duplicate": value["selected_sources"][1]["input_ids_sha256"]=value["selected_sources"][0]["input_ids_sha256"]
        elif mutation=="omitted": value["selected_sources"].pop()
        elif mutation=="extra": value["selected_sources"].append(copy.deepcopy(value["selected_sources"][0]))
        else: value["selected_sources"][0]["field"]="spoof"
        with pytest.raises(c7.C7Error): c7.validate_development_manifest(value,path)


def test_parent_tar_and_stdout_are_material_runtime_bindings():
    bound={x["path"] for x in c7.verify_runtime_closure()}
    assert "artifacts/c6-v1/cascadekv-c6-result.tar.gz" in bound
    assert "artifacts/c6-v1/evidence/c6_v1_kaggle_stdout.md" in bound


def _stat(loss,traffic,cosine):
    return {"robust_relative_l2":loss,"relative_l2":loss,"total_kv_bytes":traffic,"cosine":cosine}


@pytest.mark.parametrize("case",range(12))
def test_c6_direct_reuse_parity_matrix(case):
    groups=[]
    for cell in range(160):
        row={a:_stat(1.0+(cell%3)*.01,10.0+(cell%7),.99) for a in c7.ACTIONS}
        row["A0"]=_stat(.1,10,.98);row["A1"]=_stat(.1 if case%2 else .2,9 if case%3 else 10,.99)
        row["A2"]=_stat(.1 if case==9 else .3,9,.99)
        groups.append(row)
    domains=c7.action_domain_by_cell() if case in (10,11) else (c7.BASE_ACTIONS,)*160
    cap=(160*9 if case in (1,2,3) else 160*10)*1000
    projected=[{a:{"relative_l2":x["robust_relative_l2"],"total_kv_bytes":x["total_kv_bytes"],"cosine":x["cosine"]} for a,x in row.items()} for row in groups]
    assert c7.optimize_action_cells(groups,cap,domains=domains)==c6.optimize_action_cells(projected,cap,c7.ACTIONS,domains)


def test_global_and_robust_are_semantically_independent():
    result={"development_baseline_metrics":{"uniform10":{"mean_relative_l2":.2},"flat5":{"mean_total_kv_bytes":10}},"schedules":{"T0":{"feasible":True}},"development_target_metrics":{"T0":{"mean_cosine":.99,"mean_relative_l2":.121,"mean_total_kv_bytes":9}}}
    assert c7.development_passing_targets(result)==[] # robust diagnostics cannot pass a global gate
    result["development_target_metrics"]["T0"]["mean_relative_l2"]=.119
    assert c7.development_passing_targets(result)==["T0"]


def test_global_gate_two_way_proof_uses_real_development_result_shape():
    # The robust scalar is a DP diagnostic only. It cannot waive a global gate.
    result={"development_baseline_metrics":{"uniform10":{"mean_relative_l2":.2},"flat5":{"mean_total_kv_bytes":10}},"schedules":{"T0":{"feasible":True}},"development_target_metrics":{"T0":{"mean_cosine":.99,"mean_relative_l2":.121,"mean_total_kv_bytes":9}},"action_cell_statistics":{"0:0:A0":{"robust_relative_l2":.001}}}
    assert c7.development_passing_targets(result)==[]
    result["development_target_metrics"]["T0"]={"mean_cosine":.985,"mean_relative_l2":.119,"mean_total_kv_bytes":9}
    result["action_cell_statistics"]["0:0:A0"]["robust_relative_l2"]=.121
    assert c7.development_passing_targets(result)==["T0"]


def test_t5_is_exactly_one_microbyte_below_current_flat5():
    protocol=c7.validate_protocol()
    mean,strict,cap=c7._target_cap(protocol,"T5",123.25,7.125)
    assert (mean,strict,cap)==(7.125,True,1_140_000-1)
    assert 1_140_000>cap and 1_139_999<=cap
    _,_,changed=c7._target_cap(protocol,"T5",999,8.001)
    assert changed==1_280_160-1 and changed!=cap


def test_selected_target_controls_prospective_verdict():
    gates={"T0":{"passes":False},"T1":{"passes":True}}
    assert c7.prospective_verdict("T0",gates)=="C7-PROSPECTIVE-NO-PASS"
    with pytest.raises(c7.C7Error): c7.prospective_verdict("T2",gates)


def test_selected_target_lock_missing_metrics_rejects():
    with pytest.raises(c7.C7Error): c7.prospective_verdict("T0",{})


def test_final_audit_binds_root_protocol_runtime_and_qualification_bytes(tmp_path, monkeypatch):
    """Root copies are immutable evidence, independently of downstream JSON."""
    root=tmp_path/"result"; root.mkdir()
    for name, source in (("protocol.json",c7.PROTOCOL),("runtime_manifest.json",c7.RUNTIME)):
        (root/name).write_bytes(source.read_bytes())
    qualification=root/"qualification.json"; qualification.write_bytes(b"qualification")
    for name in ("development_sources.json","development_capture_manifest.json"):
        (root/name).write_text("{}")
    dev={"protocol_sha256":c7.sha256_path(c7.PROTOCOL),"runtime_manifest_sha256":c7.sha256_path(c7.RUNTIME),"qualification_sha256":c7.sha256_path(qualification),"development_source_manifest_sha256":c7.sha256_path(root/"development_sources.json"),"capture_manifest_sha256":c7.sha256_path(root/"development_capture_manifest.json"),"development_passing_targets":[],"classification":"C7-DEVELOPMENT-NO-GO"}
    (root/"development.json").write_text(json.dumps(dev))
    monkeypatch.setattr(c7.c4,"_validate_c2_qualification",lambda _: {"backend_id":c7.c4.BACKEND_ID})
    monkeypatch.setattr(c7,"development_passing_targets",lambda _: [])
    monkeypatch.setattr(c7.c4,"QUALIFICATION_SHA",hashlib.sha256(b"qualification").hexdigest())
    assert c7.final_audit(root)=="C7-DEVELOPMENT-NO-GO"
    for name in ("protocol.json","runtime_manifest.json","qualification.json"):
        original=(root/name).read_bytes(); (root/name).write_bytes(original+b"x")
        with pytest.raises(c7.C7Error): c7.final_audit(root)
        (root/name).write_bytes(original)
