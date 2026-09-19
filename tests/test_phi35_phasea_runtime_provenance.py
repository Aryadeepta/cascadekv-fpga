import json
from pathlib import Path

import pytest

from cascadekv.runtime_provenance import RuntimeProvenanceError, verify_runtime_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/cascadekv_phi35_phasea_runtime_manifest.json"


def _manifest_with(tmp_path, mutate):
    payload = json.loads(MANIFEST.read_text())
    mutate(payload)
    path = tmp_path / "runtime-manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_phasea_manifest_hashes_bind_the_complete_local_runtime_path():
    verified = verify_runtime_manifest(MANIFEST, ROOT, require_tracked=False)
    payload = json.loads(MANIFEST.read_text())
    assert payload["phase_a_scientific_status"] == "NO_PHI_DATA_CONSUMED"
    assert payload["target_model"]["name"] == "microsoft/Phi-3.5-mini-instruct"
    assert payload["target_model"]["revision"] == "2fe192450127e6a83f7441aef6e3ca586c338b77"
    assert payload["routing_profile_status"].startswith("UNFROZEN")
    assert not any("lambda" in key or "schedule" in key for key in payload)
    assert [entry["path"] for entry in verified] == [
        "cascadekv/model_agnostic.py",
        "cascadekv/evaluation_core.py",
        "cascadekv/adaptive_lifting.py",
        "cascadekv/quantize.py",
        "pyproject.toml",
        "uv.lock",
    ]


def test_phasea_runtime_verifier_fails_closed_for_absent_or_hash_changed_source(tmp_path):
    absent = _manifest_with(tmp_path, lambda payload: payload["bound_files"][0].update(path="cascadekv/absent.py"))
    with pytest.raises(RuntimeProvenanceError, match="absent"):
        verify_runtime_manifest(absent, ROOT, require_tracked=False)
    changed = _manifest_with(tmp_path, lambda payload: payload["bound_files"][0].update(sha256="0" * 64))
    with pytest.raises(RuntimeProvenanceError, match="hash mismatch"):
        verify_runtime_manifest(changed, ROOT, require_tracked=False)


def test_phasea_runtime_verifier_fails_closed_for_untracked_source(monkeypatch):
    class Result:
        returncode = 1

    monkeypatch.setattr("cascadekv.runtime_provenance.subprocess.run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeProvenanceError, match="untracked"):
        verify_runtime_manifest(MANIFEST, ROOT)
