"""Offline adversarial C7 binding tests; no model, tokenizer, or data access."""
import copy
import hashlib
import json

import pytest

from cascadekv import phi35_8k_c6 as c6
from cascadekv import phi35_8k_c7 as c7


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
