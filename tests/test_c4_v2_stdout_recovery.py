import hashlib
import json
from pathlib import Path

import pytest

from cascadekv import c4_v2_stdout_recovery as r


def digest(x): return hashlib.sha256(x).hexdigest()

def transcript(rows):
    return b"SHA256SUMS:\n" + b"\n".join(f"{h}  {n}".encode() for n, h in rows.items()) + b"\n"

def with_frozen(rows):
    root = Path(__file__).parents[1]
    rows["c4_protocol.json"] = digest((root / "configs/cascadekv_phi35_8k_c4_protocol.json").read_bytes())
    rows["c4_runtime_manifest.json"] = digest((root / "configs/cascadekv_phi35_8k_c4_runtime_manifest.json").read_bytes())
    return rows

def test_sums_and_one_line_extraction():
    rows = {name: "a" * 64 for name in r.bundle.REQUIRED}
    assert r.parse_sha256s(transcript(rows).decode()) == rows
    found = r.one_line_json_candidates(b"x\n{\"schema_version\":\"x\"}\n")
    assert found[0][0] == 2 and found[0][1] == b'{"schema_version":"x"}'

def test_literal_hash_and_newline_variants(tmp_path, monkeypatch):
    payload = b'{"schema_version":"cascadekv-phi35-8k-c4-test-v1"}'
    rows = with_frozen({name: "0" * 64 for name in r.bundle.REQUIRED}); rows["test.json"] = digest(payload + b"\n")
    # Keep all other objects intentionally missing; test the narrowly justified LF variant.
    path = tmp_path / "stdout"; path.write_bytes(transcript(rows) + payload + b"\n")
    m = r.recover(path, tmp_path / "out")
    assert m["members"]["test.json"]["classification"] == r.CLASS_BYTE
    assert (tmp_path / "out" / "test.json").read_bytes() == payload + b"\n"

def test_valid_json_wrong_hash_is_not_byte_recovered(tmp_path):
    payload = b'{"schema_version":"cascadekv-phi35-8k-c4-test-v1"}'
    rows = with_frozen({name: "0" * 64 for name in r.bundle.REQUIRED}); rows["test.json"] = "f" * 64
    path = tmp_path / "stdout"; path.write_bytes(transcript(rows) + payload + b"\n")
    m = r.recover(path, tmp_path / "out")
    assert m["members"]["test.json"]["classification"] == r.CLASS_ATTESTED

def test_missing_and_quantitative_refusal(tmp_path):
    rows = with_frozen({name: "0" * 64 for name in r.bundle.REQUIRED})
    path = tmp_path / "stdout"; path.write_bytes(transcript(rows))
    m = r.recover(path, tmp_path / "out")
    assert m["members"]["development.json"]["classification"] == r.CLASS_MISSING
    with pytest.raises(r.RecoveryError, match="authenticated development.json"):
        r.postmortem(m)

def test_frozen_git_recovery_and_deterministic_manifest(tmp_path):
    root = Path(__file__).parents[1]
    data = (root / "configs/cascadekv_phi35_8k_c4_protocol.json").read_bytes()
    runtime = (root / "configs/cascadekv_phi35_8k_c4_runtime_manifest.json").read_bytes()
    rows = {name: "0" * 64 for name in r.bundle.REQUIRED}; rows["c4_protocol.json"] = digest(data); rows["c4_runtime_manifest.json"] = digest(runtime)
    path = tmp_path / "stdout"; path.write_bytes(transcript(rows))
    m1 = r.recover(path, tmp_path / "out"); m2 = r.recover(path, tmp_path / "out")
    assert m1["members"]["c4_protocol.json"]["classification"] == r.CLASS_FROZEN
    assert m1["members"]["c4_protocol.json"]["verified_recovered_sha256"] == digest(data)
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
