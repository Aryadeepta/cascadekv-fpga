#!/usr/bin/env python3
"""One-shot C7 executor with immutable, C7-specific evidence."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from cascadekv import phi35_8k_c7 as c7
from cascadekv.stdout_artifact_recovery import emit
SMALL=("protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json","holdout.json","holdout_capture_manifest.json","test.json","SHA256SUMS")
C2_TAG="cascadekv-phi35-8k-kaggle-c2-prep-v2";C2_COMMIT="fa54b581ee5ee581dd51b8085db8b5233467450a";C2_DURABLE_QUALIFICATION=Path("/kaggle/working/cascadekv_phi35_8k_capture/c7_v1/qualification.json");C2_WORKTREE=Path("/kaggle/working/cascadekv_phi35_8k_c2_worktree_c7_v1")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _copy_new(src,dst):
 if Path(dst).exists() or Path(dst).is_symlink():raise c7.C7Error("stale output artifact")
 shutil.copy2(src,dst)
def _clean_root(root):
 if Path(root).exists() and any(Path(root).iterdir()):raise c7.C7Error("output root must be absent or empty")
 Path(root).mkdir(parents=True,exist_ok=True)
def _audit_checkout():
 if subprocess.check_output(["git","-C",str(c7.ROOT),"status","--porcelain"],text=True):raise c7.C7Error("frozen executor requires a clean checkout")
def _phase(name):print(f"===== C7 PHASE: {name} =====")
def bundle(root,result_dir):
 root=Path(root);items={n:root/n for n in SMALL[:-1] if (root/n).is_file()};sums=root/"SHA256SUMS"
 if sums.exists() or sums.is_symlink():raise c7.C7Error("stale SHA256SUMS")
 with sums.open("x",encoding="ascii") as h:h.write("".join(f"{sha(p)}  {n}\n" for n,p in sorted(items.items())))
 items["SHA256SUMS"]=sums;Path(result_dir).mkdir(parents=True,exist_ok=True);tar=Path(result_dir)/"cascadekv-c7-result.tar.gz"
 if tar.exists() or tar.is_symlink():raise c7.C7Error("stale result tarball")
 with tarfile.open(tar,"x:gz") as t:
  for n,p in sorted(items.items()):t.add(p,arcname=n,recursive=False)
 return emit(dict(sorted(items.items())))
def _qualification(root):
 if C2_WORKTREE.exists() or C2_WORKTREE.is_symlink() or C2_DURABLE_QUALIFICATION.exists() or C2_DURABLE_QUALIFICATION.is_symlink():raise c7.C7Error("stale C2 evidence")
 active=c7._git("HEAD");added=False
 try:
  subprocess.run(["git","-C",str(c7.ROOT),"worktree","add","--detach",str(C2_WORKTREE),C2_TAG],check=True);added=True
  if subprocess.check_output(["git","-C",str(C2_WORKTREE),"rev-parse","HEAD"],text=True).strip()!=C2_COMMIT:raise c7.C7Error("C2 commit differs")
  C2_DURABLE_QUALIFICATION.parent.mkdir(parents=True,exist_ok=False);env=os.environ.copy();env["PYTHONPATH"]=str(C2_WORKTREE)+os.pathsep+env.get("PYTHONPATH","")
  subprocess.run([sys.executable,"-c","from pathlib import Path;import sys;from cascadekv import phi35_kaggle_c2 as c2;c2.qualify(Path(sys.argv[1]))",str(C2_DURABLE_QUALIFICATION)],cwd=C2_WORKTREE,env=env,check=True)
  if not C2_DURABLE_QUALIFICATION.is_file() or C2_DURABLE_QUALIFICATION.is_symlink():raise c7.C7Error("durable qualification absent")
  record=c7.c4._validate_c2_qualification(C2_DURABLE_QUALIFICATION)
  if sha(C2_DURABLE_QUALIFICATION)!=c7.c4.QUALIFICATION_SHA or record.get("backend_id")!=c7.c4.BACKEND_ID:raise c7.C7Error("qualification differs")
  dst=Path(root)/"qualification.json";_copy_new(C2_DURABLE_QUALIFICATION,dst)
  if sha(dst)!=sha(C2_DURABLE_QUALIFICATION) or c7._git("HEAD")!=active:raise c7.C7Error("qualification copy/head differs")
  return dst
 finally:
  if added:subprocess.run(["git","-C",str(c7.ROOT),"worktree","remove","--force",str(C2_WORKTREE)],check=False)
def run(a):
 _phase("PREFLIGHT / CHECKOUT");_audit_checkout();c7.preflight(execution=True);_clean_root(a.root);root=Path(a.root);_copy_new(c7.PROTOCOL,root/"protocol.json");_copy_new(c7.RUNTIME,root/"runtime_manifest.json")
 _phase("C2 QUALIFICATION");q=_qualification(root)
 _phase("DEVELOPMENT SOURCE MANIFEST");ds=root/"development_sources.json";c7.build_development_manifest(output=ds)
 _phase("DEVELOPMENT CAPTURE");dc=root/"development_capture";c7.capture_sources(kind="development",sources=c7.validate_development_manifest(json.loads(ds.read_text()),ds),output=dc,qualification=q,development_source_sha=sha(ds));_copy_new(dc/"final_capture_manifest.json",root/"development_capture_manifest.json")
 _phase("DEVELOPMENT EVALUATION + ROBUST DP");dev=root/"development.json";r=c7.develop(capture_root=dc,capture_manifest=dc/"final_capture_manifest.json",development_source_manifest=ds,qualification=q,output=dev)
 _phase("DEVELOPMENT GO/NO-GO")
 if r["classification"]=="C7-DEVELOPMENT-NO-GO":
  _phase("FINAL AUDIT");c7.final_audit(root);_phase("RESULT BUNDLE");frame=bundle(root,a.result_dir);_phase("STDOUT RECOVERY");print(frame,end="");_phase("DEVELOPMENT CLEANUP");shutil.rmtree(dc);return 0
 devsha=sha(dev);c7.assert_holdout_may_start(r,devsha,dev);_phase("PROSPECTIVE SELECTION");hold=root/"holdout.json";selected=c7.select_holdout(development_source_manifest=ds,development_result=dev,output=hold)
 _phase("DEVELOPMENT CLEANUP");shutil.rmtree(dc);_phase("HOLDOUT CAPTURE");hc=root/"holdout_capture";c7.capture_sources(kind="holdout",sources=selected["selected_sources"],output=hc,qualification=q,holdout_manifest=hold,development_sha=devsha,schedule_sha=r["schedule_sha256"]);_copy_new(hc/"final_capture_manifest.json",root/"holdout_capture_manifest.json")
 _phase("PROSPECTIVE TEST");c7.prospective_test(holdout_capture_root=hc,holdout_capture_manifest=hc/"final_capture_manifest.json",holdout_manifest=hold,development_result=dev,development_source_manifest=ds,output=root/"test.json")
 _phase("FINAL AUDIT");c7.final_audit(root);_phase("RESULT BUNDLE");frame=bundle(root,a.result_dir);_phase("STDOUT RECOVERY");print(frame,end="");_phase("HOLDOUT CLEANUP");shutil.rmtree(hc);return 0
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--result-dir",type=Path,required=True);a=p.parse_args()
 try:return run(a)
 except Exception as e:print(f"PHI35-8K-C7-FAILED: {e}");return 1
if __name__=="__main__":raise SystemExit(main())
