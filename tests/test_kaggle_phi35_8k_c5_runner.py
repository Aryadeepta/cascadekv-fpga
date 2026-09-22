"""Synthetic end-to-end proof for the C5 executor (no torch or data access)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

from cascadekv import phi35_8k_c5 as c5
from cascadekv.stdout_artifact_recovery import recover


RUNNER = Path(__file__).parents[1] / "scripts/kaggle_phi35_8k_c5.py"

def load_runner():
    spec=importlib.util.spec_from_file_location("synthetic_c5_runner", RUNNER)
    module=importlib.util.module_from_spec(spec); assert spec.loader
    spec.loader.exec_module(module)
    return module

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def development(go):
    metric={"mean_cosine":.99,"mean_relative_l2":.1 if go else .3,"mean_total_kv_bytes":9.}
    schedules={t:{"feasible":True,"target_mean_kv_bytes":1.,"strict":False,"integer_microbyte_cap":1000,"integer_microbytes_used":1000,"table":{}} for t in c5.TARGETS}
    row={"schedules":schedules,"development_target_metrics":{t:metric for t in c5.TARGETS},"development_baseline_metrics":{"flat5":{"mean_total_kv_bytes":10.},"uniform10":{"mean_relative_l2":.2}},"schedule_sha256":c5.schedule_digest(schedules)}
    return c5.freeze_development(row)

def install_synthetic_lifecycle(monkeypatch, tmp_path, *, go):
    runner=load_runner(); log=[]; snapshots={}
    monkeypatch.setattr(runner, "_audit_checkout", lambda:log.append("checkout_audit"))
    monkeypatch.setattr(c5, "preflight", lambda **kw:log.append("execution_preflight"))
    monkeypatch.setattr(runner, "_clean_root", lambda p:(log.append("clean_output_root"), p.mkdir(parents=True,exist_ok=True)))
    monkeypatch.setattr(runner, "_copy_new", lambda s,d:(log.append("frozen_copy") if d.name=="protocol.json" else None, d.write_bytes(b'{"frozen":true}\n')))
    # Lifecycle tests inject only the canonical qualification boundary.
    monkeypatch.setattr(runner,"_qualification",lambda root:(log.append("qualification"),(root/"qualification.json").write_bytes(b'{"qualification":true}\n'),root/"qualification.json")[2])
    monkeypatch.setattr(c5,"_git",lambda ref:"synthetic-commit")
    def build(*,output): log.append("development_manifest"); output.write_bytes(b'{"development_sources":true}\n'); return {}
    def capture(*,kind,output,**kw):
        log.append(f"{kind}_capture"); output.mkdir(); (output/"final_capture_manifest.json").write_bytes(("{"+json.dumps({"kind":kind})[1:]+"\n").encode()); return {}
    def develop(**kw):
        log.append("development_evaluation"); result=development(go); kw["output"].write_text(json.dumps(result,sort_keys=True)+"\n"); snapshots["development_sha"]=digest(kw["output"]); return result
    monkeypatch.setattr(c5,"build_development_manifest",build); monkeypatch.setattr(c5,"capture_sources",capture); monkeypatch.setattr(c5,"develop",develop)
    monkeypatch.setattr(c5,"validate_development_manifest",lambda payload,path: [])
    if go:
        original=c5.assert_holdout_may_start
        def selection(**kw):
            log.append("select_holdout"); p=kw["development_result"]; assert digest(p)==snapshots["development_sha"]
            data=json.loads(p.read_text()); assert data["schedule_sha256"]==development(True)["schedule_sha256"] and data["selected_target"]=="T0"
            result={"selected_sources":[],"next_untouched_frontier":{"narrative":18,"qa":19,"report":21}}
            kw["output"].write_text(json.dumps(result)+"\n"); return result
        monkeypatch.setattr(c5,"select_holdout",selection)
        def prospective(**kw):
            log.append("prospective_test"); assert digest(kw["development_result"])==snapshots["development_sha"]
            result={"classification":"C5-PROSPECTIVE-PASS"}; kw["output"].write_text(json.dumps(result)+"\n"); return result
        monkeypatch.setattr(c5,"prospective_test",prospective)
    else:
        monkeypatch.setattr(c5,"select_holdout",lambda **kw:(_ for _ in ()).throw(AssertionError("prospective selection called")))
        monkeypatch.setattr(c5,"prospective_test",lambda **kw:(_ for _ in ()).throw(AssertionError("prospective test called")))
    args=types.SimpleNamespace(root=tmp_path/"run",result_dir=tmp_path/"results")
    return runner,args,log,snapshots

def assert_bundle_and_recovery(tmp_path, args, expected):
    tar=args.result_dir/"cascadekv-c5-result.tar.gz"
    with tarfile.open(tar) as t:
        assert set(t.getnames())==set(expected)
        values={m:t.extractfile(m).read() for m in expected}
    sums={name:value for value,name in (line.split("  ",1) for line in values["SHA256SUMS"].decode().splitlines())}
    assert set(sums)==set(expected)-{"SHA256SUMS"}
    assert all(hashlib.sha256(values[name]).hexdigest()==value for name,value in sums.items())
    # Recovery frames are stdout after audit; no tensor is present in either form.
    stream=(tmp_path/"stdout.txt").read_text()
    restored=recover(stream, tmp_path/"restored")
    assert {k:v.read_bytes() for k,v in restored.items()}==values

def test_runner_import_has_no_c2_heavy_import():
    sys.modules.pop("cascadekv.phi35_kaggle_c2",None)
    runner=load_runner()
    assert "cascadekv.phi35_kaggle_c2" not in sys.modules
    assert callable(runner.run)

def test_actual_runner_no_go_order_bundle_recovery_and_banners(monkeypatch,tmp_path,capsys):
    runner,args,log,_=install_synthetic_lifecycle(monkeypatch,tmp_path,go=False)
    assert runner.run(args)==0; (tmp_path/"stdout.txt").write_text(capsys.readouterr().out)
    assert log==["checkout_audit","execution_preflight","clean_output_root","frozen_copy","qualification","development_manifest","development_capture","development_evaluation"]
    assert not (args.root/"development_capture").exists()
    expected=["protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json","SHA256SUMS"]
    assert_bundle_and_recovery(tmp_path,args,expected)
    stream=(tmp_path/"stdout.txt").read_text(); assert "C5-DEVELOPMENT-NO-GO" in stream and "PROSPECTIVE SELECTION" not in stream

def test_actual_runner_go_order_immutable_development_and_bundle(monkeypatch,tmp_path,capsys):
    runner,args,log,snapshots=install_synthetic_lifecycle(monkeypatch,tmp_path,go=True)
    assert runner.run(args)==0; (tmp_path/"stdout.txt").write_text(capsys.readouterr().out)
    assert log==["checkout_audit","execution_preflight","clean_output_root","frozen_copy","qualification","development_manifest","development_capture","development_evaluation","select_holdout","holdout_capture","prospective_test"]
    assert digest(args.root/"development.json")==snapshots["development_sha"]
    assert not (args.root/"development_capture").exists() and not (args.root/"holdout_capture").exists()
    expected=["protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json","holdout.json","holdout_capture_manifest.json","test.json","SHA256SUMS"]
    assert_bundle_and_recovery(tmp_path,args,expected)
    assert "next_untouched_frontier" in (tmp_path/"stdout.txt").read_text()

@pytest.mark.parametrize("name",["protocol.json","runtime_manifest.json","qualification.json","development.json","holdout.json","unrelated"])
def test_runner_rejects_any_stale_output(monkeypatch,tmp_path,name):
    runner=load_runner(); root=tmp_path/"run"; root.mkdir(); (root/name).write_text("old")
    with pytest.raises(c5.C5Error): runner._clean_root(root)

def test_bundle_refuses_existing_result_tarball(tmp_path):
    root=tmp_path/"root"; root.mkdir(); (root/"protocol.json").write_text("x")
    result=tmp_path/"result"; result.mkdir(); (result/"cascadekv-c5-result.tar.gz").write_bytes(b"old")
    with pytest.raises(c5.C5Error): load_runner().bundle(root,result)

def test_lazy_canonical_qualification_success_and_failure_matrix(monkeypatch,tmp_path):
    runner=load_runner(); assert "--qualification" not in RUNNER.read_text()
    canonical=b'{"canonical":true}\n'; digest=hashlib.sha256(canonical).hexdigest(); calls=[]
    fake=types.SimpleNamespace(qualify=lambda p:(p.write_bytes(canonical),calls.append("qualify")))
    monkeypatch.setitem(sys.modules,"cascadekv.phi35_kaggle_c2",fake)
    monkeypatch.setattr(c5.c4,"QUALIFICATION_SHA",digest); monkeypatch.setattr(c5.c4,"BACKEND_ID","synthetic-backend")
    monkeypatch.setattr(c5.c4,"_validate_c2_qualification",lambda p:{"backend_id":"synthetic-backend"})
    path=runner._qualification(tmp_path)
    assert path==tmp_path/"qualification.json" and path.read_bytes()==canonical and calls==["qualify"]
    variants={
        "raises":lambda p:(_ for _ in ()).throw(RuntimeError("no")),
        "absent":lambda p:None,
        "wrong_bytes":lambda p:p.write_bytes(b"wrong"),
    }
    for name,qualify in variants.items():
        root=tmp_path/name; root.mkdir(); monkeypatch.setattr(fake,"qualify",qualify)
        with pytest.raises((c5.C5Error,RuntimeError,FileNotFoundError)): runner._qualification(root)
    root=tmp_path/"validation"; root.mkdir(); fake.qualify=lambda p:p.write_bytes(canonical)
    monkeypatch.setattr(c5.c4,"_validate_c2_qualification",lambda p:(_ for _ in ()).throw(c5.C5Error("bad validation")))
    with pytest.raises(c5.C5Error): runner._qualification(root)
    root=tmp_path/"backend"; root.mkdir(); monkeypatch.setattr(c5.c4,"_validate_c2_qualification",lambda p:{"backend_id":"other"})
    with pytest.raises(c5.C5Error): runner._qualification(root)
