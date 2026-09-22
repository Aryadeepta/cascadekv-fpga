"""Final local, mock-only C6 freeze proofs (no torch or scientific inputs)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascadekv import phi35_8k_c6 as c6
from cascadekv.stdout_artifact_recovery import BEGIN, recover

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = b"SHOULD_NOT_PERSIST_RAW_SOURCE_TEXT_C6_4D71"


def runner_module():
    spec = importlib.util.spec_from_file_location("c6_final_runner", ROOT / "scripts/kaggle_phi35_8k_c6.py")
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(runner)
    return runner


def test_c2_qualification_worktree_matrix_without_importing_c2(tmp_path, monkeypatch):
    """Exercise the real subprocess/worktree adapter only; C2 is never imported here."""
    runner = runner_module(); worktree = tmp_path / "worktree"; durable = tmp_path / "durable" / "qualification.json"
    assert (runner.C2_TAG, runner.C2_COMMIT, str(runner.C2_DURABLE_QUALIFICATION)) == ("cascadekv-phi35-8k-kaggle-c2-prep-v2", "fa54b581ee5ee581dd51b8085db8b5233467450a", "/kaggle/working/cascadekv_phi35_8k_capture/c6_v1/qualification.json")
    monkeypatch.setattr(runner, "C2_WORKTREE", worktree); monkeypatch.setattr(runner, "C2_DURABLE_QUALIFICATION", durable)
    monkeypatch.setattr(runner.c6, "ROOT", tmp_path / "active")
    calls = []; removals = []; state = {"case": "success", "head": "active-head"}
    def check_output(args, **_):
        calls.append(("check_output", args))
        if "worktree" in str(args): return runner.C2_COMMIT + "\n"
        return state["head"] + "\n"
    def run(args, **kwargs):
        calls.append(("run", args, kwargs))
        if "add" in args:
            if state["case"] == "add_failure": raise subprocess.CalledProcessError(1, args)
            worktree.mkdir()
        elif args[-2:] == ["-c", args[-1]]:  # unreachable; retained as an explicit no-C2 import guard
            raise AssertionError("unexpected qualifier command shape")
        elif "-c" in args:
            if state["case"] == "subprocess_failure": raise subprocess.CalledProcessError(1, args)
            durable.parent.mkdir(parents=True, exist_ok=True)
            if state["case"] != "absent": durable.write_bytes(b"qualification")
        elif "remove" in args:
            removals.append(tuple(args))
            shutil.rmtree(worktree, ignore_errors=True)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "check_output", check_output); monkeypatch.setattr(runner.subprocess, "run", run)
    def validate(_):
        if state["case"] == "validator": raise c6.C6Error("rejected")
        return {"backend_id": "wrong"} if state["case"] == "backend" else {"backend_id": runner.c6.c4.BACKEND_ID}
    monkeypatch.setattr(runner.c6.c4, "_validate_c2_qualification", validate)
    real_sha = runner.sha
    monkeypatch.setattr(runner, "sha", lambda p: "0" * 64 if state["case"] == "sha" else real_sha(p))
    # Success proves exact detached tag/commit, interpreter, cwd, PYTHONPATH,
    # exact-byte copy, digest recheck, unchanged active HEAD, and cleanup.
    dest = tmp_path / "root"; dest.mkdir()
    monkeypatch.setattr(runner.c6.c4, "QUALIFICATION_SHA", hashlib.sha256(b"qualification").hexdigest())
    assert runner._qualification(dest) == dest / "qualification.json"
    qualifier = next(row for row in calls if row[0] == "run" and "-c" in row[1])
    assert qualifier[1][0] == sys.executable and qualifier[2]["cwd"] == worktree
    assert qualifier[2]["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(worktree)
    assert any(row[0] == "run" and row[1][-1] == runner.C2_TAG for row in calls)
    assert not worktree.exists() and (dest / "qualification.json").read_bytes() == b"qualification"
    # Each post-add failure removes the worktree.  Pre-add stale states fail closed.
    for case in ("wrong_commit", "subprocess_failure", "absent", "validator", "backend", "sha"):
        calls.clear(); state["case"] = case; state["head"] = "active-head"; dest2 = tmp_path / case; dest2.mkdir()
        if case == "wrong_commit":
            monkeypatch.setattr(runner.subprocess, "check_output", lambda args, **_: "bad\n" if "worktree" in str(args) else "active-head\n")
        with pytest.raises(c6.C6Error): runner._qualification(dest2)
        assert removals
        monkeypatch.setattr(runner.subprocess, "check_output", check_output)
        shutil.rmtree(durable.parent, ignore_errors=True); shutil.rmtree(worktree, ignore_errors=True)
    state["case"] = "success"; worktree.mkdir()
    with pytest.raises(c6.C6Error): runner._qualification(tmp_path / "stale-worktree")
    shutil.rmtree(worktree); durable.parent.mkdir(); durable.write_bytes(b"old")
    with pytest.raises(c6.C6Error): runner._qualification(tmp_path / "stale-durable")
    # Remaining immutable-output and active-checkout adversaries are injected
    # after worktree creation, so removal is mandatory for each.
    shutil.rmtree(durable.parent); dest3=tmp_path/"destination"; dest3.mkdir(); (dest3/"qualification.json").write_bytes(b"old")
    calls.clear()
    with pytest.raises(c6.C6Error): runner._qualification(dest3)
    assert len(removals) >= 2
    shutil.rmtree(durable.parent, ignore_errors=True); calls.clear()
    original_copy=runner._copy_new
    monkeypatch.setattr(runner,"_copy_new",lambda src,dst: (dst.write_bytes(b"corrupt")))
    with pytest.raises(c6.C6Error): runner._qualification(tmp_path/"copy-mismatch")
    assert len(removals) >= 3
    monkeypatch.setattr(runner,"_copy_new",original_copy); shutil.rmtree(durable.parent, ignore_errors=True); calls.clear()
    seen_heads=iter(["active-head\n", "changed-head\n"])
    monkeypatch.setattr(runner.subprocess,"check_output",lambda args, **_: runner.C2_COMMIT+"\n" if "worktree" in str(args) else next(seen_heads))
    with pytest.raises(c6.C6Error): runner._qualification(tmp_path/"head-mutation")
    assert len(removals) >= 4


def _schedule():
    table = {f"{l}:{h}": "A0" for l in c6.c4.LAYERS for h in c6.c4.HEADS}
    return {t: {"feasible": True, "target_mean_kv_bytes": 1., "strict": False, "integer_microbyte_cap": 160000,
                "integer_microbytes_used": 1, "table": table} for t in c6.TARGETS}


def _go_result():
    schedules = _schedule(); result = {"schedules": schedules,
        "development_target_metrics": {t: {"mean_cosine": .99, "mean_relative_l2": .1, "mean_total_kv_bytes": 1.} for t in c6.TARGETS},
        "development_baseline_metrics": {"flat5": {"mean_total_kv_bytes": 2.}, "uniform10": {"mean_relative_l2": .2}}}
    result.update(classification="C6-DEVELOPMENT-GO", development_passing_targets=list(c6.TARGETS), selected_target="T0", schedule_sha256=c6.schedule_digest(schedules))
    return result


def _runner_mocks(runner, monkeypatch, *, go):
    monkeypatch.setattr(runner, "_audit_checkout", lambda: None); monkeypatch.setattr(runner.c6, "preflight", lambda **_: None); monkeypatch.setattr(runner.c6, "_git", lambda _: "frozen")
    monkeypatch.setattr(runner, "_qualification", lambda root: (root / "qualification.json").write_bytes(b"q") or root / "qualification.json")
    def sources(*, output):
        # Raw source exists only in this mock's local variable and is never serialized.
        raw = SENTINEL.decode(); assert raw.endswith("4D71"); output.write_text("{}")
    def capture(*, output, **_): output.mkdir(); (output / "final_capture_manifest.json").write_text("capture")
    def develop(*, output, **_):
        result = _go_result() if go else {"classification":"C6-DEVELOPMENT-NO-GO", "schedule_sha256":"s" * 64, "development_passing_targets":[], "selected_target":None}
        output.write_text(json.dumps(result)); return result
    monkeypatch.setattr(runner.c6, "build_development_manifest", sources); monkeypatch.setattr(runner.c6, "validate_development_manifest", lambda *_: [])
    monkeypatch.setattr(runner.c6, "capture_sources", capture); monkeypatch.setattr(runner.c6, "develop", develop)


def test_actual_go_runner_proves_order_freeze_firewall_bundle_and_raw_text(tmp_path, monkeypatch, capsys):
    runner = runner_module(); _runner_mocks(runner, monkeypatch, go=True); root=tmp_path/"root"; results=tmp_path/"results"; seen=[]
    original_assert = runner.c6.assert_holdout_may_start
    def frozen(result, digest, path):
        assert path.read_bytes() and digest == hashlib.sha256(path.read_bytes()).hexdigest(); assert result["selected_target"] == "T0"; seen.append("frozen")
        return original_assert(result, digest, path)
    def select(**kwargs):
        assert seen == ["frozen"] and kwargs["development_result"].is_file()
        payload={"next_untouched_frontier":{"narrative":18,"qa":19,"report":22}}
        kwargs["output"].write_text(json.dumps(payload)); return {"selected_sources": []}
    def test(**kwargs):
        assert not (root/"development_capture").exists()
        kwargs["output"].write_text("test"); return {"classification":"C6-PROSPECTIVE-PASS"}
    monkeypatch.setattr(runner.c6, "assert_holdout_may_start", frozen); monkeypatch.setattr(runner.c6, "select_holdout", select); monkeypatch.setattr(runner.c6, "prospective_test", test)
    for name in ("optimize_action_cells", "optimize_static_schedule"):
        monkeypatch.setattr(runner.c6, name, lambda *a, **k: pytest.fail("development optimizer used prospectively"))
    assert runner.run(SimpleNamespace(root=root, result_dir=results)) == 0
    stdout=capsys.readouterr().out; phases=[x.split(": ",1)[1].split(" =====")[0] for x in stdout.splitlines() if x.startswith("===== C6 PHASE")]
    assert phases == ["PREFLIGHT / CHECKOUT","C2 QUALIFICATION","DEVELOPMENT SOURCE MANIFEST","DEVELOPMENT CAPTURE","DEVELOPMENT EVALUATION + DP","DEVELOPMENT GO/NO-GO","PROSPECTIVE SELECTION","DEVELOPMENT CLEANUP","HOLDOUT CAPTURE","PROSPECTIVE TEST","FINAL AUDIT","RESULT BUNDLE","STDOUT RECOVERY","HOLDOUT CLEANUP"]
    expected=set(runner.SMALL); archive=results/"cascadekv-c6-result.tar.gz"
    with tarfile.open(archive) as tar: members={n:tar.extractfile(n).read() for n in tar.getnames()}
    recovered=recover(stdout[stdout.index(BEGIN):], tmp_path/"recovered")
    assert set(members)==expected==set(recovered) and all(recovered[n].read_bytes()==b for n,b in members.items())
    checks={line.split("  ")[1]: line.split("  ")[0] for line in members["SHA256SUMS"].decode().splitlines()}
    assert all(hashlib.sha256(data).hexdigest()==checks[name] for name,data in members.items() if name != "SHA256SUMS")
    surfaces=list(members.values())+[stdout.encode()]+[p.read_bytes() for p in recovered.values()]
    assert all(SENTINEL not in raw for raw in surfaces)


def test_no_go_raw_text_sentinel_absent_from_all_emitted_bytes(tmp_path, monkeypatch, capsys):
    runner=runner_module(); _runner_mocks(runner, monkeypatch, go=False); root=tmp_path/"root"; results=tmp_path/"results"
    assert runner.run(SimpleNamespace(root=root,result_dir=results)) == 0
    stdout=capsys.readouterr().out; archive=results/"cascadekv-c6-result.tar.gz"
    with tarfile.open(archive) as tar: members={n:tar.extractfile(n).read() for n in tar.getnames()}
    recovered=recover(stdout[stdout.index(BEGIN):],tmp_path/"recovered")
    persistent=[p.read_bytes() for p in root.iterdir() if p.is_file()]
    assert all(SENTINEL not in raw for raw in [*members.values(), *persistent, stdout.encode(), *(p.read_bytes() for p in recovered.values())])


def test_go_stdout_failure_preserves_holdout_capture(tmp_path, monkeypatch):
    runner=runner_module(); _runner_mocks(runner, monkeypatch, go=True); root=tmp_path/"root"
    monkeypatch.setattr(runner.c6, "assert_holdout_may_start", lambda *_: "T0")
    monkeypatch.setattr(runner.c6, "select_holdout", lambda **k: (k["output"].write_text(json.dumps({"next_untouched_frontier":{}})), {"selected_sources":[]})[1])
    monkeypatch.setattr(runner.c6, "prospective_test", lambda **k: (k["output"].write_text("x"), {"classification":"C6-PROSPECTIVE-PASS"})[1])
    monkeypatch.setattr(runner, "_recover_stdout", lambda *_: (_ for _ in ()).throw(RuntimeError("stdout")))
    with pytest.raises(RuntimeError): runner.run(SimpleNamespace(root=root,result_dir=tmp_path/"results"))
    assert (root/"holdout_capture").is_dir()


@pytest.mark.parametrize("selected,gates,expected", [("T0", {"T0":{"passes":False},"T1":{"passes":True}}, "C6-PROSPECTIVE-NO-PASS"), ("T0", {"T0":{"passes":True},"T1":{"passes":False}}, "C6-PROSPECTIVE-PASS"), ("T0", {"T0":{"passes":False},"T1":{"passes":True},"first_passing_target_descriptive":"T1"}, "C6-PROSPECTIVE-NO-PASS")])
def test_selected_target_exclusively_controls_verdict(selected,gates,expected): assert c6.prospective_verdict(selected,gates)==expected


@pytest.mark.parametrize("field,value", [("selected_target", "T1"), ("schedule_sha256", "0" * 64), ("development_result_sha256", "0" * 64)])
def test_holdout_binding_rejects_selected_target_schedule_and_development_bytes(tmp_path, monkeypatch, field, value):
    """These are the three prospective lock fields, checked before any capture opens."""
    development=tmp_path/"development.json"; development.write_text(json.dumps(_go_result()))
    holdout=tmp_path/"holdout.json"; payload={"selected_target":"T0", "schedule_sha256":_go_result()["schedule_sha256"], "development_result_sha256":hashlib.sha256(development.read_bytes()).hexdigest()}; payload[field]=value; holdout.write_text(json.dumps(payload))
    monkeypatch.setattr(c6,"validate_holdout_manifest",lambda *_: [])
    with pytest.raises(c6.C6Error): c6.prospective_test(holdout_capture_root=tmp_path/"capture",holdout_capture_manifest=tmp_path/"capture.json",holdout_manifest=holdout,development_result=development,output=tmp_path/"test.json")


def test_runtime_each_bound_file_is_material(tmp_path, monkeypatch):
    manifest=json.loads(c6.RUNTIME.read_text()); fixture=tmp_path/"fixture"
    protocol_source = ROOT / manifest["protocol"]["path"]
    protocol_target = fixture / manifest["protocol"]["path"]
    protocol_target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(protocol_source, protocol_target)
    for row in manifest["bound_files"]:
        source=ROOT/row["path"]; target=fixture/row["path"]; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    protocol=protocol_target; runtime=fixture/"runtime.json"; manifest["protocol"]["sha256"]=hashlib.sha256(protocol.read_bytes()).hexdigest(); runtime.write_text(json.dumps(manifest))
    monkeypatch.setattr(c6,"ROOT",fixture); monkeypatch.setattr(c6,"PROTOCOL",protocol); monkeypatch.setattr(c6,"RUNTIME",runtime)
    assert len(manifest["bound_files"]) == 27
    for row in manifest["bound_files"]:
        target=fixture/row["path"]; original=target.read_bytes(); target.write_bytes(original+b"\n# mutation")
        with pytest.raises(c6.C6Error): c6.verify_runtime_closure()
        target.write_bytes(original)


def test_granular_protocol_leaves_reject_independently():
    base=c6.validate_protocol(); paths=[]
    for action, fields in (("A0",("kind","candidate_fraction","routing_profile")),("A9",("candidate_fraction","routing_profile")),("A10",("kind","candidate_fraction","routing_profile")),("A11",("kind","candidate_fraction","routing_profile"))): paths += [("methods","actions",action,f) for f in fields]
    paths += [("targets","fractions",t) for t in c6.TARGETS] + [("holdout_selection","requires_frozen_development_result_and_schedule"),("holdout_selection","persist_raw_text"),("firewall","development_and_holdout_tensor_roots_distinct"),("scientific_state","local_preparation_only"),("scientific_state","prospective_frontier_accessed")]
    for path in paths:
        changed=copy.deepcopy(base); node=changed
        for key in path[:-1]: node=node[key]
        node[path[-1]]="MUTATED"
        with pytest.raises(c6.C6Error): c6.validate_protocol(changed)
    for key, value in (("layer_0", [x for x in c6.ACTIONS if x != "A10"]),("layer_0", [x for x in c6.ACTIONS if x != "A11"]),("nonzero_sampled_layers", list(c6.BASE_ACTIONS)+["A10"]),("nonzero_sampled_layers", list(c6.BASE_ACTIONS)+["A11"])):
        changed=copy.deepcopy(base); changed["action_validity"][key]=value
        with pytest.raises(c6.C6Error): c6.validate_protocol(changed)
