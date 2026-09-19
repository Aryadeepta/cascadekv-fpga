"""Fail-closed verification for prospective CascadeKV runtime source closure."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


class RuntimeProvenanceError(RuntimeError):
    """The declared runtime closure is absent, untracked, or byte-different."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runtime_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeProvenanceError(f"cannot load runtime provenance manifest: {path}") from exc
    required = {"schema_version", "purpose", "target_family", "target_model", "phase_a_scientific_status", "routing_profile_status", "bound_files"}
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise RuntimeProvenanceError("runtime provenance manifest has an incomplete schema")
    if manifest["schema_version"] != 1 or manifest["target_family"] != "phi3":
        raise RuntimeProvenanceError("runtime provenance manifest has an unsupported schema/family")
    if manifest["phase_a_scientific_status"] != "NO_PHI_DATA_CONSUMED":
        raise RuntimeProvenanceError("runtime provenance manifest does not preserve Phase-A status")
    target = manifest["target_model"]
    if (not isinstance(target, dict) or not isinstance(target.get("name"), str)
            or not isinstance(target.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", target["revision"])):
        raise RuntimeProvenanceError("runtime provenance manifest lacks a pinned target model")
    if not isinstance(manifest["routing_profile_status"], str) or not manifest["routing_profile_status"].startswith("UNFROZEN"):
        raise RuntimeProvenanceError("runtime provenance manifest must leave Phi routing unfrozen")
    if not isinstance(manifest["bound_files"], list) or not manifest["bound_files"]:
        raise RuntimeProvenanceError("runtime provenance manifest binds no files")
    return manifest


def verify_runtime_manifest(manifest_path: Path, repository_root: Path, *, require_tracked: bool = True) -> tuple[dict[str, str], ...]:
    """Verify every declared source.  Tracking is required by default.

    ``require_tracked=False`` exists solely for isolated hash tests; callers
    establishing execution closure must use the fail-closed default.
    """
    manifest = load_runtime_manifest(manifest_path)
    root = repository_root.resolve()
    seen: set[str] = set()
    verified: list[dict[str, str]] = []
    for entry in manifest["bound_files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise RuntimeProvenanceError("each bound runtime entry must contain only path and sha256")
        relative, expected = entry["path"], entry["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeProvenanceError("invalid bound runtime entry")
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeProvenanceError("duplicate or unsafe bound runtime path")
        seen.add(relative)
        source = root / relative
        if not source.is_file():
            raise RuntimeProvenanceError(f"bound runtime source is absent: {relative}")
        if require_tracked:
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
                text=True, capture_output=True, check=False,
            )
            if tracked.returncode != 0:
                raise RuntimeProvenanceError(f"bound runtime source is untracked: {relative}")
        actual = _sha256(source)
        if actual != expected:
            raise RuntimeProvenanceError(f"bound runtime source hash mismatch: {relative}")
        verified.append({"path": relative, "sha256": actual})
    return tuple(verified)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a CascadeKV runtime-source manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        verified = verify_runtime_manifest(args.manifest, args.repository_root)
    except RuntimeProvenanceError as exc:
        print(f"runtime provenance closure FAILED: {exc}")
        return 1
    print(f"runtime provenance closure verified: {len(verified)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
