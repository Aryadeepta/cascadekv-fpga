"""Deterministic, text-free recovery framing for small CascadeKV artifacts."""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

BEGIN = "===== BEGIN CASCADEKV C5 RECOVERY MANIFEST ====="
END = "===== END CASCADEKV C5 RECOVERY MANIFEST ====="
ARTIFACT_BEGIN = "===== BEGIN CASCADEKV C5 ARTIFACT "
ARTIFACT_END = "===== END CASCADEKV C5 ARTIFACT "


class RecoveryError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def emit(artifacts: Mapping[str, Path]) -> str:
    """Return a framed stream.  Callers print it only after terminal packaging."""
    rows = []
    for name, path in artifacts.items():
        if not name or "/" in name or ".." in name or not path.is_file():
            raise RecoveryError("unsafe or missing recovery artifact")
        data = path.read_bytes()
        rows.append({"filename": name, "byte_count": len(data), "sha256": sha256_bytes(data),
                     "base64": base64.b64encode(data).decode("ascii")})
    manifest = {"schema_version": "cascadekv-c5-stdout-recovery-v1", "artifacts": [{key: row[key] for key in ("filename", "byte_count", "sha256")} for row in rows]}
    lines = [BEGIN, canonical_json(manifest), END]
    for row in rows:
        lines += [f"{ARTIFACT_BEGIN}{row['filename']} =====", canonical_json(row),
                  f"{ARTIFACT_END}{row['filename']} ====="]
    return "\n".join(lines) + "\n"


def recover(stream: str, output: Path) -> dict[str, Path]:
    """Recover exact bytes, requiring each per-artifact row to match the manifest."""
    lines = stream.splitlines()
    try:
        if lines.count(BEGIN) != 1 or lines.count(END) != 1: raise ValueError
        start = lines.index(BEGIN); end = lines.index(END, start + 1)
        manifest = json.loads(lines[start + 1])
    except (ValueError, IndexError, json.JSONDecodeError) as exc:
        raise RecoveryError("missing canonical recovery manifest") from exc
    if end != start + 2 or set(manifest) != {"schema_version", "artifacts"} or manifest["schema_version"] != "cascadekv-c5-stdout-recovery-v1":
        raise RecoveryError("invalid recovery manifest")
    if not isinstance(manifest["artifacts"], list): raise RecoveryError("invalid artifact manifest")
    output.mkdir(parents=True, exist_ok=True); recovered: dict[str, Path] = {}; declared=set()
    for row in manifest["artifacts"]:
        if not isinstance(row, dict) or set(row) != {"filename", "byte_count", "sha256"}:
            raise RecoveryError("invalid artifact row")
        name = row["filename"]
        if not isinstance(name, str) or not name or "/" in name or ".." in name or name in recovered:
            raise RecoveryError("unsafe artifact name")
        if not isinstance(row["byte_count"], int) or row["byte_count"] < 0 or not isinstance(row["sha256"],str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise RecoveryError("invalid artifact metadata")
        declared.add(name)
        begin, end_marker = f"{ARTIFACT_BEGIN}{name} =====", f"{ARTIFACT_END}{name} ====="
        if lines.count(begin) != 1 or lines.count(end_marker) != 1: raise RecoveryError("duplicate or missing artifact framing")
        left,right=lines.index(begin),lines.index(end_marker)
        if right != left+2: raise RecoveryError("truncated artifact framing")
        try: block=json.loads(lines[left+1]); data = base64.b64decode(block.get("base64"), validate=True)
        except (ValueError, TypeError) as exc: raise RecoveryError("invalid artifact base64") from exc
        if not isinstance(block,dict) or set(block)!={"filename","byte_count","sha256","base64"} or any(block[key]!=row[key] for key in row): raise RecoveryError("artifact block metadata mismatch")
        if len(data) != row["byte_count"] or sha256_bytes(data) != row["sha256"]:
            raise RecoveryError("artifact byte/hash mismatch")
        path = output / name; path.write_bytes(data); recovered[name] = path
    # No unreferenced block may smuggle an artifact outside manifest review.
    blocks=[]; endings=[]
    for line in lines:
        if line.startswith(ARTIFACT_BEGIN) and line.endswith(" ====="):
            blocks.append(line[len(ARTIFACT_BEGIN):-len(" =====")])
        if line.startswith(ARTIFACT_END) and line.endswith(" ====="):
            endings.append(line[len(ARTIFACT_END):-len(" =====")])
        if line.startswith(ARTIFACT_BEGIN) and not line.endswith(" ====="):
            raise RecoveryError("malformed artifact begin framing")
        if line.startswith(ARTIFACT_END) and not line.endswith(" ====="):
            raise RecoveryError("malformed artifact end framing")
    if set(blocks) != declared or len(blocks) != len(declared) or set(endings) != declared or len(endings) != len(declared):
        raise RecoveryError("extra or duplicate artifact block")
    return recovered
