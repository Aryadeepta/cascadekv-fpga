#!/usr/bin/env python3
"""Audited, one-shot Kaggle executor for the frozen C5 lifecycle."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path

# This file is deliberately runnable by path (``python scripts/...``), where
# Python otherwise places only ``scripts/`` on sys.path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cascadekv import phi35_8k_c5 as c5
from cascadekv.stdout_artifact_recovery import emit

SMALL=("protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json","holdout.json","holdout_capture_manifest.json","test.json","SHA256SUMS")
C2_TAG = "cascadekv-phi35-8k-kaggle-c2-prep-v2"
C2_COMMIT = "fa54b581ee5ee581dd51b8085db8b5233467450a"
C2_DURABLE_ROOT = Path("/kaggle/working/cascadekv_phi35_8k_capture")
C2_DURABLE_QUALIFICATION = C2_DURABLE_ROOT / "c5_v2" / "qualification.json"
C2_WORKTREE = Path("/kaggle/working/cascadekv_phi35_8k_c2_worktree_v2")
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _copy_new(src:Path,dst:Path)->None:
    if dst.exists(): raise c5.C5Error(f"stale output artifact: {dst.name}")
    shutil.copy2(src,dst)
def _clean_root(root:Path)->None:
    if root.exists() and any(root.iterdir()): raise c5.C5Error("output root must be absent or empty")
    root.mkdir(parents=True,exist_ok=True)
def _audit_checkout()->None:
    dirty=subprocess.check_output(["git","-C",str(c5.ROOT),"status","--porcelain"],text=True)
    if dirty: raise c5.C5Error("frozen executor requires a clean checkout")
def _phase(name:str)->None:
    """Stable, terse progress markers for the one-shot Kaggle transcript."""
    print(f"===== C5 PHASE: {name} =====")

def bundle(root:Path,result_dir:Path)->str:
    present={n:root/n for n in SMALL if n != "SHA256SUMS" and (root/n).is_file()}
    (root/"SHA256SUMS").write_text("".join(f"{sha(p)}  {n}\n" for n,p in sorted(present.items())))
    present["SHA256SUMS"]=root/"SHA256SUMS"; result_dir.mkdir(parents=True,exist_ok=True)
    target=result_dir/"cascadekv-c5-result.tar.gz"
    if target.exists() or target.is_symlink(): raise c5.C5Error("refusing to overwrite existing scientific result tarball")
    with tarfile.open(target,"x:gz") as archive:
        for n,p in sorted(present.items()): archive.add(p,arcname=n,recursive=False)
    return emit(dict(sorted(present.items())))
def _qualification(root:Path)->Path:
    """Run the unmodified C2 qualifier from its frozen detached checkout."""
    if C2_WORKTREE.exists() or C2_WORKTREE.is_symlink():
        raise c5.C5Error(f"stale C2 worktree path: {C2_WORKTREE}")
    if C2_DURABLE_QUALIFICATION.exists() or C2_DURABLE_QUALIFICATION.is_symlink():
        raise c5.C5Error(f"stale durable C2 qualification: {C2_DURABLE_QUALIFICATION}")
    active_head=c5._git("HEAD")
    added=False
    try:
        subprocess.run(["git", "-C", str(c5.ROOT), "worktree", "add", "--detach", str(C2_WORKTREE), C2_TAG], check=True)
        added=True
        resolved=subprocess.check_output(["git", "-C", str(C2_WORKTREE), "rev-parse", "HEAD"], text=True).strip()
        if resolved != C2_COMMIT:
            raise c5.C5Error("isolated C2 worktree commit differs from frozen C2 tag")
        C2_DURABLE_QUALIFICATION.parent.mkdir(parents=True, exist_ok=False)
        environment=os.environ.copy()
        environment["PYTHONPATH"]=str(C2_WORKTREE)+os.pathsep+environment.get("PYTHONPATH", "")
        program="from pathlib import Path; import sys; from cascadekv import phi35_kaggle_c2 as c2; c2.qualify(Path(sys.argv[1]))"
        subprocess.run([sys.executable, "-c", program, str(C2_DURABLE_QUALIFICATION)], cwd=C2_WORKTREE, env=environment, check=True)
        if not C2_DURABLE_QUALIFICATION.is_file():
            raise c5.C5Error("canonical C2 qualification output is missing")
        record=c5.c4._validate_c2_qualification(C2_DURABLE_QUALIFICATION)
        digest=sha(C2_DURABLE_QUALIFICATION)
        if digest != c5.c4.QUALIFICATION_SHA or record.get("backend_id") != c5.c4.BACKEND_ID:
            raise c5.C5Error("canonical C2 qualification differs")
        destination=root/"qualification.json"
        _copy_new(C2_DURABLE_QUALIFICATION, destination)
        if sha(destination) != digest:
            raise c5.C5Error("copied C2 qualification SHA differs")
        if c5._git("HEAD") != active_head:
            raise c5.C5Error("active C5 checkout changed during C2 qualification")
        print("C5-C2-QUALIFICATION "+json.dumps({"c2_worktree_tag":C2_TAG,"c2_worktree_commit":resolved,"c2_durable_qualification_path":str(C2_DURABLE_QUALIFICATION),"qualification_sha256":digest,"backend_id":record["backend_id"]},sort_keys=True))
        return destination
    except (OSError, subprocess.CalledProcessError) as exc:
        raise c5.C5Error("isolated canonical C2 qualification failed") from exc
    finally:
        if added:
            subprocess.run(["git", "-C", str(c5.ROOT), "worktree", "remove", "--force", str(C2_WORKTREE)], check=False)
def _print_audit(root:Path,result:dict,test:dict|None=None,result_dir:Path|None=None)->None:
    fields={"c5_tag":c5.C5_TAG,"c5_commit":c5._git("HEAD"),"protocol_sha256":sha(c5.PROTOCOL),"runtime_manifest_sha256":sha(c5.RUNTIME),"qualification_sha256":sha(root/"qualification.json"),"development_source_sha256":sha(root/"development_sources.json"),"development_capture_sha256":sha(root/"development_capture_manifest.json"),"development_result_sha256":sha(root/"development.json"),"schedule_sha256":result["schedule_sha256"],"development_classification":result["classification"],"development_passing_targets":result["development_passing_targets"],"selected_target":result["selected_target"],"optimizer_rerun":False}
    if (root/"holdout.json").is_file(): fields["holdout_sha256"]=sha(root/"holdout.json")
    if (root/"holdout_capture_manifest.json").is_file(): fields["holdout_capture_sha256"]=sha(root/"holdout_capture_manifest.json")
    if test: fields.update(test_sha256=sha(root/"test.json"),prospective_classification=test["classification"],schedule_unchanged=True,next_untouched_frontier=json.loads((root/"holdout.json").read_text())["next_untouched_frontier"])
    if result_dir is not None: fields["result_bundle_path"]=str(result_dir/"cascadekv-c5-result.tar.gz")
    print("C5-AUDIT "+json.dumps(fields,sort_keys=True))
def run(a):
    _phase("PREFLIGHT / CHECKOUT"); _audit_checkout(); c5.preflight(execution=True); _clean_root(a.root); root=a.root
    _copy_new(c5.PROTOCOL,root/"protocol.json"); _copy_new(c5.RUNTIME,root/"runtime_manifest.json")
    _phase("C2 QUALIFICATION"); qualification=_qualification(root)
    _phase("DEVELOPMENT SOURCE MANIFEST"); devsrc=root/"development_sources.json"; c5.build_development_manifest(output=devsrc)
    _phase("DEVELOPMENT CAPTURE"); devcap=root/"development_capture"; c5.capture_sources(kind="development",sources=c5.validate_development_manifest(json.loads(devsrc.read_text()),devsrc),output=devcap,qualification=qualification,development_source_sha=sha(devsrc)); _copy_new(devcap/"final_capture_manifest.json",root/"development_capture_manifest.json")
    _phase("DEVELOPMENT EVALUATION + DP"); development=root/"development.json"; result=c5.develop(capture_root=devcap,capture_manifest=devcap/"final_capture_manifest.json",development_source_manifest=devsrc,qualification=qualification,output=development)
    _phase("DEVELOPMENT GO/NO-GO")
    if result["classification"]=="C5-DEVELOPMENT-NO-GO":
        _phase("FINAL AUDIT"); _print_audit(root,result,result_dir=a.result_dir)
        _phase("RESULT BUNDLE"); framed=bundle(root,a.result_dir)
        _phase("STDOUT RECOVERY"); print(framed,end=""); _phase("DEVELOPMENT CLEANUP"); shutil.rmtree(devcap); return 0
    devsha=sha(development); c5.assert_holdout_may_start(result,devsha,development)
    _phase("PROSPECTIVE SELECTION"); holdout=root/"holdout.json"; selected=c5.select_holdout(development_source_manifest=devsrc,development_result=development,output=holdout)
    _phase("DEVELOPMENT CLEANUP"); shutil.rmtree(devcap)
    _phase("HOLDOUT CAPTURE"); holdcap=root/"holdout_capture"; c5.capture_sources(kind="holdout",sources=selected["selected_sources"],output=holdcap,qualification=qualification,holdout_manifest=holdout,development_sha=devsha,schedule_sha=result["schedule_sha256"]); _copy_new(holdcap/"final_capture_manifest.json",root/"holdout_capture_manifest.json")
    _phase("PROSPECTIVE TEST"); test=c5.prospective_test(holdout_capture_root=holdcap,holdout_capture_manifest=holdcap/"final_capture_manifest.json",holdout_manifest=holdout,development_result=development,output=root/"test.json")
    _phase("FINAL AUDIT"); _print_audit(root,result,test,a.result_dir)
    _phase("RESULT BUNDLE"); framed=bundle(root,a.result_dir)
    _phase("STDOUT RECOVERY"); print(framed,end=""); _phase("HOLDOUT CLEANUP"); shutil.rmtree(holdcap); return 0
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,required=True); parser.add_argument("--result-dir",type=Path,required=True); args=parser.parse_args()
    try: return run(args)
    except Exception as exc: print(f"PHI35-8K-C5-FAILED: {exc}"); return 1
if __name__=="__main__": raise SystemExit(main())
