"""Adversarial manifest-first tests for the C6 capture firewall.

These fixtures intentionally contain arbitrary bytes instead of tensors.  A
resolver must finish all structural/provenance validation before a caller is
ever given an opportunity to open such a payload.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cascadekv import phi35_8k_c4 as c4
from cascadekv import phi35_8k_c6 as c6


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: object | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())


def _source(family: str, index: int, role: str, number: int) -> dict:
    return {"family": family, **{key: c4.SOURCE_SPECS[family][key] for key in ("dataset", "config", "split", "revision", "field")},
            "identity_kind": "dataset_index", "dataset_index": index,
            "input_ids_sha256": f"{number:064x}", "role": role}


def _development_manifest() -> dict:
    rows = [_source(family, index, "development", n + 1)
            for n, (family, index) in enumerate((f, i) for f in c4.FAMILIES for i in c6.DEVELOPMENT[f])]
    for row in rows:
        row.pop("role")
    return {"schema_version": "cascadekv-phi35-8k-c6-development-sources-v1", "protocol_sha256": c6.sha256_path(c6.PROTOCOL),
            "target": {"model": c4.MODEL, "model_revision": c4.REVISION, "tokenizer_revision": c4.REVISION},
            "tokenization_contract": {"add_special_tokens": True, "truncation": True, "max_length": 8192,
                                      "required_input_ids_length": 8192, "canonicalization": "CPU contiguous int64 bytes"},
            "source_specs": c4.SOURCE_SPECS, "selected_sources": rows, "rejected_sources": {f: [] for f in c4.FAMILIES},
            "global_input_ids_sha256_unique": True, "source_text_stored": False}


def _holdout_manifest() -> dict:
    rows = [_source(family, c6.FRONTIER[family], "holdout", n + 41) for n, family in enumerate(c4.FAMILIES)]
    return {"schema_version": "cascadekv-phi35-8k-c6-holdout-selection-v1", "protocol_sha256": c6.sha256_path(c6.PROTOCOL),
            "development_source_manifest_sha256": "a" * 64, "development_result_sha256": "b" * 64,
            "schedule_sha256": "c" * 64, "selected_target": "T0",
            "selection_rule": "first mechanically eligible globally-new exact input SHA in ascending order",
            "selected_sources": rows, "rejected_sources": {f: [] for f in c4.FAMILIES},
            "next_untouched_frontier": {f: c6.FRONTIER[f] + 1 for f in c4.FAMILIES}, "raw_text_persisted": False}


def fixture(tmp_path: Path, kind: str) -> tuple[Path, Path, Path, dict]:
    root = tmp_path / "capture"; manifest_path = root / "final_capture_manifest.json"
    source_path = tmp_path / ("development_sources.json" if kind == "development" else "holdout.json")
    source = _development_manifest() if kind == "development" else _holdout_manifest()
    _write(source_path, source)
    sources = c6.validate_development_manifest(source) if kind == "development" else c6.validate_holdout_manifest(source)
    rows = []
    for source_number, item in enumerate(sources):
        identity = f"{item['family']}:{item['dataset_index']}:{kind}"
        for layer in c4.LAYERS:
            artifact_rel = f"artifacts/{identity.replace(':', '_')}_{layer}.bin"
            provenance_rel = f"artifacts/{identity.replace(':', '_')}_{layer}.json"
            artifact = f"payload-{identity}-{layer}".encode()
            _write(root / artifact_rel, artifact)
            detail = {"protocol_sha256": c6.sha256_path(c6.PROTOCOL), "identity": identity, "layer": layer,
                      "input_ids_sha256": item["input_ids_sha256"], "artifact_sha256": _digest(artifact)}
            _write(root / provenance_rel, detail)
            rows.append({"identity": identity, "layer": layer, "artifact_relative_path": artifact_rel,
                         "artifact_sha256": _digest(artifact), "provenance_relative_path": provenance_rel,
                         "provenance_sha256": _digest((root / provenance_rel).read_bytes()),
                         "input_ids_sha256": item["input_ids_sha256"],
                         "source": {"family": item["family"], "dataset_index": item["dataset_index"], "role": kind}})
    payload = {"schema_version": "cascadekv-phi35-8k-c6-capture-v1", "kind": kind,
               "protocol_sha256": c6.sha256_path(c6.PROTOCOL), "qualification_sha256": c4.QUALIFICATION_SHA,
               "backend_id": c4.BACKEND_ID,
               "development_source_manifest_sha256": c6.sha256_path(source_path) if kind == "development" else None,
               "holdout_manifest_sha256": c6.sha256_path(source_path) if kind == "holdout" else None,
               "development_result_sha256": "b" * 64 if kind == "holdout" else None,
               "schedule_sha256": "c" * 64 if kind == "holdout" else None,
               "artifact_count": len(rows), "forwards": len(sources), "artifacts": rows}
    _write(manifest_path, payload)
    return root, manifest_path, source_path, payload


def resolve(root, manifest, source, kind):
    args = {"development_source_manifest": source} if kind == "development" else {"holdout_manifest": source, "development_sha": "b" * 64, "schedule_sha": "c" * 64}
    return c6.CaptureResolver(root, manifest, kind, **args)


def test_valid_development_capture_fixture_has_15_forwards_75_artifacts(tmp_path):
    root, manifest, source, _ = fixture(tmp_path, "development")
    assert len(resolve(root, manifest, source, "development").ordered()) == 75


def test_valid_holdout_capture_fixture_has_3_forwards_15_artifacts(tmp_path):
    root, manifest, source, _ = fixture(tmp_path, "holdout")
    assert len(resolve(root, manifest, source, "holdout").ordered()) == 15


@pytest.mark.parametrize("mutation", [
    "protocol", "qualification", "backend", "source_sha", "forwards_low", "forwards_high", "count_low", "count_high",
    "missing_source", "extra_source", "report20", "missing_layer", "unexpected_layer", "duplicate", "wrong_family",
    "wrong_index", "wrong_role", "input_mismatch", "input_malformed", "artifact_malformed", "provenance_malformed",
    "missing_artifact", "missing_provenance", "artifact_changed", "provenance_changed", "provenance_protocol",
    "provenance_identity", "provenance_layer", "provenance_input", "provenance_artifact", "absolute_artifact",
    "absolute_provenance", "traversal_artifact", "traversal_provenance", "symlink_artifact", "symlink_provenance", "source_bytes",
])
def test_development_capture_resolver_rejects_before_payload_open(tmp_path, mutation):
    root, manifest, source, payload = fixture(tmp_path, "development")
    row = payload["artifacts"][0]
    if mutation == "protocol": payload["protocol_sha256"] = "0" * 64
    elif mutation == "qualification": payload["qualification_sha256"] = "0" * 64
    elif mutation == "backend": payload["backend_id"] = "0" * 64
    elif mutation == "source_sha": payload["development_source_manifest_sha256"] = "0" * 64
    elif mutation == "forwards_low": payload["forwards"] = 14
    elif mutation == "forwards_high": payload["forwards"] = 16
    elif mutation == "count_low": payload["artifact_count"] = 74
    elif mutation == "count_high": payload["artifact_count"] = 76
    elif mutation == "missing_source": payload["artifacts"] = [x for x in payload["artifacts"] if not x["identity"].startswith("narrative:13:")]
    elif mutation == "extra_source": row["identity"] = "narrative:99:development"
    elif mutation == "report20": row["identity"] = "report:20:development"
    elif mutation == "missing_layer": payload["artifacts"] = [x for x in payload["artifacts"] if x["layer"] != 0]
    elif mutation == "unexpected_layer": row["layer"] = 1
    elif mutation == "duplicate": payload["artifacts"][-1] = copy.deepcopy(row)
    elif mutation == "wrong_family": row["source"]["family"] = "qa"
    elif mutation == "wrong_index": row["source"]["dataset_index"] = 999
    elif mutation == "wrong_role": row["source"]["role"] = "holdout"
    elif mutation == "input_mismatch": row["input_ids_sha256"] = "f" * 64
    elif mutation == "input_malformed": row["input_ids_sha256"] = "bad"
    elif mutation == "artifact_malformed": row["artifact_sha256"] = "bad"
    elif mutation == "provenance_malformed": row["provenance_sha256"] = "bad"
    elif mutation == "missing_artifact": (root / row["artifact_relative_path"]).unlink()
    elif mutation == "missing_provenance": (root / row["provenance_relative_path"]).unlink()
    elif mutation == "artifact_changed": _write(root / row["artifact_relative_path"], b"changed")
    elif mutation == "provenance_changed": _write(root / row["provenance_relative_path"], {"changed": True})
    elif mutation.startswith("provenance_"):
        path = root / row["provenance_relative_path"]; detail = json.loads(path.read_text())
        field = {"provenance_protocol": "protocol_sha256", "provenance_identity": "identity", "provenance_layer": "layer", "provenance_input": "input_ids_sha256", "provenance_artifact": "artifact_sha256"}[mutation]
        detail[field] = "bad" if field != "layer" else 1; _write(path, detail); row["provenance_sha256"] = _digest(path.read_bytes())
    elif mutation == "absolute_artifact": row["artifact_relative_path"] = "/tmp/nope"
    elif mutation == "absolute_provenance": row["provenance_relative_path"] = "/tmp/nope"
    elif mutation == "traversal_artifact": row["artifact_relative_path"] = "../nope"
    elif mutation == "traversal_provenance": row["provenance_relative_path"] = "../nope"
    elif mutation in {"symlink_artifact", "symlink_provenance"}:
        field = "artifact_relative_path" if mutation == "symlink_artifact" else "provenance_relative_path"; path = root / row[field]; path.unlink(); path.symlink_to("/etc/hosts")
    elif mutation == "source_bytes": source.write_text(source.read_text() + " ")
    _write(manifest, payload)
    opens = 0
    with pytest.raises(c6.C6Error):
        resolve(root, manifest, source, "development")
    assert opens == 0


@pytest.mark.parametrize("mutation", [
    "protocol", "qualification", "backend", "source_sha", "forwards_low", "forwards_high", "count_low", "count_high",
    "extra_source", "duplicate_family", "missing_source", "wrong_role", "missing_layer", "unexpected_layer", "duplicate",
    "input_mismatch", "input_malformed", "artifact_malformed", "provenance_malformed", "missing_artifact", "missing_provenance",
    "artifact_changed", "provenance_changed", "provenance_protocol", "absolute_artifact", "traversal_provenance", "symlink_artifact",
    "source_bytes", "development_result_bytes",
])
def test_holdout_capture_resolver_rejects_before_payload_open(tmp_path, mutation):
    root, manifest, source, payload = fixture(tmp_path, "holdout")
    row = payload["artifacts"][0]
    if mutation == "protocol": payload["protocol_sha256"] = "0" * 64
    elif mutation == "qualification": payload["qualification_sha256"] = "0" * 64
    elif mutation == "backend": payload["backend_id"] = "0" * 64
    elif mutation == "source_sha": payload["holdout_manifest_sha256"] = "0" * 64
    elif mutation == "forwards_low": payload["forwards"] = 2
    elif mutation == "forwards_high": payload["forwards"] = 4
    elif mutation == "count_low": payload["artifact_count"] = 14
    elif mutation == "count_high": payload["artifact_count"] = 16
    elif mutation == "extra_source": row["identity"] = "narrative:99:holdout"
    elif mutation == "duplicate_family": row["identity"] = "qa:19:holdout"
    elif mutation == "missing_source": payload["artifacts"] = payload["artifacts"][5:]
    elif mutation == "wrong_role": row["source"]["role"] = "development"
    elif mutation == "missing_layer": payload["artifacts"] = payload["artifacts"][1:]
    elif mutation == "unexpected_layer": row["layer"] = 1
    elif mutation == "duplicate": payload["artifacts"][-1] = copy.deepcopy(row)
    elif mutation == "input_mismatch": row["input_ids_sha256"] = "f" * 64
    elif mutation == "input_malformed": row["input_ids_sha256"] = "bad"
    elif mutation == "artifact_malformed": row["artifact_sha256"] = "bad"
    elif mutation == "provenance_malformed": row["provenance_sha256"] = "bad"
    elif mutation == "missing_artifact": (root / row["artifact_relative_path"]).unlink()
    elif mutation == "missing_provenance": (root / row["provenance_relative_path"]).unlink()
    elif mutation == "artifact_changed": _write(root / row["artifact_relative_path"], b"changed")
    elif mutation == "provenance_changed": _write(root / row["provenance_relative_path"], {"changed": True})
    elif mutation == "provenance_protocol":
        path = root / row["provenance_relative_path"]; detail = json.loads(path.read_text()); detail["protocol_sha256"] = "0" * 64; _write(path, detail); row["provenance_sha256"] = _digest(path.read_bytes())
    elif mutation == "absolute_artifact": row["artifact_relative_path"] = "/tmp/nope"
    elif mutation == "traversal_provenance": row["provenance_relative_path"] = "../nope"
    elif mutation == "symlink_artifact":
        path = root / row["artifact_relative_path"]; path.unlink(); path.symlink_to("/etc/hosts")
    elif mutation in {"source_bytes", "development_result_bytes"}:
        source.write_text(source.read_text() + " ")
    _write(manifest, payload)
    opens = 0
    with pytest.raises(c6.C6Error):
        resolve(root, manifest, source, "holdout")
    assert opens == 0
