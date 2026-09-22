#!/usr/bin/env python3
"""Audited, one-shot Kaggle executor for the frozen C5 lifecycle."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, tarfile
from pathlib import Path
from cascadekv import phi35_8k_c5 as c5
from cascadekv.stdout_artifact_recovery import emit

SMALL=("protocol.json","runtime_manifest.json","qualification.json","development_sources.json","development_capture_manifest.json","development.json","holdout.json","holdout_capture_manifest.json","test.json","SHA256SUMS")
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
    # C2 imports torch/model dependencies.  Keep them out of local runner
    # import so the lifecycle can be tested with an injected canonical C2.
    from cascadekv import phi35_kaggle_c2 as c2
    path=root/"qualification.json"; c2.qualify(path)
    record=c5.c4._validate_c2_qualification(path)
    if sha(path)!=c5.c4.QUALIFICATION_SHA or record["backend_id"]!=c5.c4.BACKEND_ID: raise c5.C5Error("canonical C2 qualification differs")
    return path
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
