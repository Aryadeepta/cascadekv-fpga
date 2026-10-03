"""C2 orchestration proof: import the small runner only, never C2/torch."""
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def _runner():
    spec=importlib.util.spec_from_file_location("c7_qualification_mock",ROOT/"scripts/kaggle_phi35_8k_c7.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _setup(tmp_path,monkeypatch,fail=None):
    r=_runner(); work=tmp_path/"worktree"; durable=tmp_path/"durable"/"qualification.json"; root=tmp_path/"root";root.mkdir()
    monkeypatch.setattr(r,"C2_WORKTREE",work);monkeypatch.setattr(r,"C2_DURABLE_QUALIFICATION",durable)
    monkeypatch.setattr(r.c7.c4,"QUALIFICATION_SHA",hashlib.sha256(b"q").hexdigest())
    monkeypatch.setattr(r.c7.c4,"_validate_c2_qualification",lambda _: {"backend_id":r.c7.c4.BACKEND_ID})
    calls=[]
    def check(command,**kwargs):
        calls.append(("check",command,kwargs))
        if command[-2:]==["rev-parse","HEAD"]: return "bad\n" if fail=="wrong_commit" else r.C2_COMMIT+"\n"
        return "active\n"
    def run(command,**kwargs):
        calls.append(("run",command,kwargs))
        if "add" in command:
            if fail=="add": raise subprocess.CalledProcessError(1,command)
            work.mkdir()
        elif command[0]==sys.executable:
            if fail=="qualifier": raise subprocess.CalledProcessError(1,command)
            if fail!="absent": durable.write_bytes(b"q")
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(r.subprocess,"check_output",check);monkeypatch.setattr(r.subprocess,"run",run)
    if fail=="stale_worktree": work.mkdir()
    if fail=="stale_durable": durable.parent.mkdir();durable.write_bytes(b"old")
    if fail=="copy": monkeypatch.setattr(r,"_copy_new",lambda *_: (_ for _ in ()).throw(r.c7.C7Error("copy bytes differ")))
    if fail=="destination": (root/"qualification.json").write_bytes(b"old")
    if fail=="validator": monkeypatch.setattr(r.c7.c4,"_validate_c2_qualification",lambda _: (_ for _ in ()).throw(r.c7.C7Error("validator rejects")))
    if fail=="backend": monkeypatch.setattr(r.c7.c4,"_validate_c2_qualification",lambda _: {"backend_id":"bad"})
    if fail=="sha": monkeypatch.setattr(r.c7.c4,"QUALIFICATION_SHA","0"*64)
    if fail=="head":
        n=[0]
        def git(_): n[0]+=1; return "active" if n[0]==1 else "changed"
        monkeypatch.setattr(r.c7,"_git",git)
    else: monkeypatch.setattr(r.c7,"_git",lambda _:"active")
    return r,root,calls


def test_qualification_valid_mock_has_exact_isolation_contract(tmp_path,monkeypatch):
    r,root,calls=_setup(tmp_path,monkeypatch)
    assert r._qualification(root)==root/"qualification.json"
    add=next(x for x in calls if x[0]=="run" and "add" in x[1]); qualifier=next(x for x in calls if x[0]=="run" and x[1][0]==sys.executable)
    assert add[1]==["git","-C",str(r.c7.ROOT),"worktree","add","--detach",str(r.C2_WORKTREE),r.C2_TAG]
    assert r.C2_TAG=="cascadekv-phi35-8k-kaggle-c2-prep-v2" and r.C2_COMMIT=="fa54b581ee5ee581dd51b8085db8b5233467450a"
    assert qualifier[2]["cwd"]==r.C2_WORKTREE and qualifier[2]["env"]["PYTHONPATH"].split(":")[0]==str(r.C2_WORKTREE)
    assert r.C2_DURABLE_QUALIFICATION.name=="qualification.json" and (root/"qualification.json").read_bytes()==b"q"


@pytest.mark.parametrize("failure",["stale_worktree","stale_durable","add","wrong_commit","qualifier","absent","validator","sha","backend","destination","copy","head"])
def test_qualification_12_negative_cases_and_post_add_cleanup(tmp_path,monkeypatch,failure):
    r,root,calls=_setup(tmp_path,monkeypatch,failure)
    with pytest.raises((r.c7.C7Error,subprocess.CalledProcessError)):
        r._qualification(root)
    added=r.C2_WORKTREE.exists() and failure not in {"stale_worktree","stale_durable","add"}
    removed=any(x[0]=="run" and "remove" in x[1] for x in calls)
    assert removed is added


def test_qualification_mock_module_does_not_import_torch():
    before=set(sys.modules); _runner()
    assert "torch" not in set(sys.modules)-before
