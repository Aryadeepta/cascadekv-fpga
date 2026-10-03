import hashlib,importlib.util,json,subprocess,sys,tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from cascadekv.stdout_artifact_recovery import BEGIN,recover
ROOT=Path(__file__).resolve().parents[1]
def test_direct_script_import_without_pythonpath():
 r=subprocess.run([sys.executable,"scripts/kaggle_phi35_8k_c7.py","--help"],cwd=ROOT,text=True,capture_output=True,env={})
 assert r.returncode==0 and "--root" in r.stdout

def runner_module():
 spec=importlib.util.spec_from_file_location("c7_runner_under_test",ROOT/"scripts/kaggle_phi35_8k_c7.py")
 module=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(module);return module

def _light(runner,monkeypatch,go):
 monkeypatch.setattr(runner,"_audit_checkout",lambda:None);monkeypatch.setattr(runner.c7,"preflight",lambda **_:None)
 monkeypatch.setattr(runner,"_qualification",lambda root:(root/"qualification.json").write_bytes(b"q") or root/"qualification.json")
 monkeypatch.setattr(runner.c7,"build_development_manifest",lambda *,output:output.write_text("{}"));monkeypatch.setattr(runner.c7,"validate_development_manifest",lambda *_:[])
 monkeypatch.setattr(runner.c7,"capture_sources",lambda *,output,**_:(output.mkdir(),(output/"final_capture_manifest.json").write_text("capture")))
 result={"classification":"C7-DEVELOPMENT-GO" if go else "C7-DEVELOPMENT-NO-GO","schedule_sha256":"s"*64,"development_passing_targets":["T0"] if go else [],"selected_target":"T0" if go else None}
 monkeypatch.setattr(runner.c7,"develop",lambda *,output,**_:(output.write_text(json.dumps(result)),result)[1]);monkeypatch.setattr(runner.c7,"final_audit",lambda _:"ok")

def test_actual_no_go_runner_orders_bundle_and_stdout_recovery(tmp_path,monkeypatch,capsys):
 runner=runner_module();_light(runner,monkeypatch,False);root=tmp_path/"root"
 monkeypatch.setattr(runner.c7,"select_holdout",lambda **_:pytest.fail("NO-GO must not select"));monkeypatch.setattr(runner.c7,"prospective_test",lambda **_:pytest.fail("NO-GO must not test"))
 assert runner.run(SimpleNamespace(root=root,result_dir=tmp_path/"results"))==0
 output=capsys.readouterr().out;phases=[x.split(": ",1)[1].split(" =====")[0] for x in output.splitlines() if x.startswith("===== C7 PHASE")]
 assert phases==["PREFLIGHT / CHECKOUT","C2 QUALIFICATION","DEVELOPMENT SOURCE MANIFEST","DEVELOPMENT CAPTURE","DEVELOPMENT EVALUATION + ROBUST DP","DEVELOPMENT GO/NO-GO","FINAL AUDIT","RESULT BUNDLE","STDOUT RECOVERY","DEVELOPMENT CLEANUP"]
 with tarfile.open(tmp_path/"results/cascadekv-c7-result.tar.gz") as tar:members={n:tar.extractfile(n).read() for n in tar.getnames()}
 recovered=recover(output[output.index(BEGIN):],tmp_path/"recovered")
 assert set(members)=={"protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json","SHA256SUMS"}==set(recovered)
 assert all(recovered[n].read_bytes()==b for n,b in members.items()) and not (root/"development_capture").exists()

def test_actual_go_runner_freezes_development_before_selection_and_cleans_in_order(tmp_path,monkeypatch,capsys):
 runner=runner_module();_light(runner,monkeypatch,True);root=tmp_path/"root";seen=[]
 def frozen(result,digest,path):assert digest==hashlib.sha256(path.read_bytes()).hexdigest() and result["selected_target"]=="T0";seen.append("frozen");return "T0"
 def select(**kw):assert seen==["frozen"] and kw["development_result"].is_file();kw["output"].write_text("{}");return {"selected_sources":[]}
 def test(**kw):assert not (root/"development_capture").exists();kw["output"].write_text("{}");return {"classification":"C7-PROSPECTIVE-PASS"}
 monkeypatch.setattr(runner.c7,"assert_holdout_may_start",frozen);monkeypatch.setattr(runner.c7,"select_holdout",select);monkeypatch.setattr(runner.c7,"prospective_test",test)
 assert runner.run(SimpleNamespace(root=root,result_dir=tmp_path/"results"))==0
 phases=[x.split(": ",1)[1].split(" =====")[0] for x in capsys.readouterr().out.splitlines() if x.startswith("===== C7 PHASE")]
 assert phases==["PREFLIGHT / CHECKOUT","C2 QUALIFICATION","DEVELOPMENT SOURCE MANIFEST","DEVELOPMENT CAPTURE","DEVELOPMENT EVALUATION + ROBUST DP","DEVELOPMENT GO/NO-GO","PROSPECTIVE SELECTION","DEVELOPMENT CLEANUP","HOLDOUT CAPTURE","PROSPECTIVE TEST","FINAL AUDIT","RESULT BUNDLE","STDOUT RECOVERY","HOLDOUT CLEANUP"]
 assert not (root/"holdout_capture").exists()
