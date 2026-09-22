import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from cascadekv.stdout_artifact_recovery import BEGIN, recover
ROOT=Path(__file__).resolve().parents[1]
def test_direct_script_import_without_pythonpath():
    result=subprocess.run([sys.executable,"scripts/kaggle_phi35_8k_c6.py","--help"],cwd=ROOT,text=True,capture_output=True,env={})
    assert result.returncode==0 and "--root" in result.stdout


def runner_module():
    spec=importlib.util.spec_from_file_location("c6_runner_under_test", ROOT/"scripts/kaggle_phi35_8k_c6.py")
    module=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module); return module


def test_actual_runner_no_go_lifecycle_bundle_and_stdout_recovery(tmp_path, monkeypatch, capsys):
    runner=runner_module(); root=tmp_path/"run"; results=tmp_path/"results"; calls=[]
    monkeypatch.setattr(runner, "_audit_checkout", lambda: calls.append("checkout"))
    monkeypatch.setattr(runner.c6, "preflight", lambda **_: calls.append("preflight"))
    monkeypatch.setattr(runner.c6, "_git", lambda _: "frozen-head")
    def qualification(where): (where/"qualification.json").write_text("qualification"); calls.append("qualification"); return where/"qualification.json"
    def sources(*, output): output.write_text("{}"); calls.append("sources"); return {}
    def capture(*, output, **_): output.mkdir(); (output/"final_capture_manifest.json").write_text("capture"); calls.append("capture"); return {}
    def develop(*, output, **_):
        output.write_text("development"); calls.append("develop")
        return {"classification":"C6-DEVELOPMENT-NO-GO", "schedule_sha256":"s"*64, "development_passing_targets":[], "selected_target":None}
    monkeypatch.setattr(runner, "_qualification", qualification); monkeypatch.setattr(runner.c6, "build_development_manifest", sources)
    monkeypatch.setattr(runner.c6, "validate_development_manifest", lambda *_: [])
    monkeypatch.setattr(runner.c6, "capture_sources", capture); monkeypatch.setattr(runner.c6, "develop", develop)
    monkeypatch.setattr(runner.c6, "select_holdout", lambda **_: pytest.fail("holdout selection must not run"))
    monkeypatch.setattr(runner.c6, "prospective_test", lambda **_: pytest.fail("prospective test must not run"))
    assert runner.run(SimpleNamespace(root=root, result_dir=results)) == 0
    text=capsys.readouterr().out; phases=[line for line in text.splitlines() if line.startswith("===== C6 PHASE")]
    assert phases == [f"===== C6 PHASE: {x} =====" for x in ("PREFLIGHT / CHECKOUT", "C2 QUALIFICATION", "DEVELOPMENT SOURCE MANIFEST", "DEVELOPMENT CAPTURE", "DEVELOPMENT EVALUATION + DP", "DEVELOPMENT GO/NO-GO", "FINAL AUDIT", "RESULT BUNDLE", "STDOUT RECOVERY", "DEVELOPMENT CLEANUP")]
    assert calls == ["checkout","preflight","qualification","sources","capture","develop"]
    with tarfile.open(results/"cascadekv-c6-result.tar.gz") as archive:
        names=set(archive.getnames()); assert names == {"protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json","SHA256SUMS"}
        tar_bytes={name: archive.extractfile(name).read() for name in names}
    recovered=recover(text[text.index(BEGIN):], tmp_path/"recovered")
    assert set(recovered) == names and all(path.read_bytes() == tar_bytes[name] for name,path in recovered.items())
    assert not (root / "development_capture").exists()


@pytest.mark.parametrize("failure", ["bundle", "stdout"])
def test_no_go_preserves_capture_when_evidence_closure_fails(tmp_path, monkeypatch, failure):
    runner=runner_module(); root=tmp_path/"run"; results=tmp_path/"results"
    monkeypatch.setattr(runner, "_audit_checkout", lambda: None)
    monkeypatch.setattr(runner.c6, "preflight", lambda **_: None)
    monkeypatch.setattr(runner.c6, "_git", lambda _: "frozen-head")
    monkeypatch.setattr(runner, "_qualification", lambda where: (where/"qualification.json"))
    monkeypatch.setattr(runner.c6, "build_development_manifest", lambda *, output: output.write_text("{}"))
    monkeypatch.setattr(runner.c6, "validate_development_manifest", lambda *_: [])
    def capture(*, output, **_): output.mkdir(); (output/"final_capture_manifest.json").write_text("capture")
    monkeypatch.setattr(runner.c6, "capture_sources", capture)
    monkeypatch.setattr(runner.c6, "develop", lambda *, output, **_: (output.write_text("development"), {"classification":"C6-DEVELOPMENT-NO-GO", "schedule_sha256":"s"*64, "development_passing_targets":[], "selected_target":None})[1])
    # qualification is normally a durable copy; create its destination here.
    def qualification(where):
        p=where/"qualification.json"; p.write_text("qualification"); return p
    monkeypatch.setattr(runner, "_qualification", qualification)
    if failure == "bundle":
        monkeypatch.setattr(runner, "bundle", lambda *_: (_ for _ in ()).throw(RuntimeError("bundle failed")))
    else:
        monkeypatch.setattr(runner, "bundle", lambda *_: "recovery payload")
        monkeypatch.setattr(runner, "_recover_stdout", lambda *_: (_ for _ in ()).throw(RuntimeError("stdout failed")))
    with pytest.raises(RuntimeError): runner.run(SimpleNamespace(root=root, result_dir=results))
    assert (root/"development_capture").is_dir()


@pytest.mark.parametrize("name", ["protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json","holdout.json","holdout_capture_manifest.json","test.json","SHA256SUMS"])
def test_bundle_refuses_stale_evidence_member(tmp_path, name):
    runner=runner_module(); root=tmp_path/"run"; root.mkdir(); (root/name).write_text("stale")
    with pytest.raises(runner.c6.C6Error): runner._clean_root(root)
