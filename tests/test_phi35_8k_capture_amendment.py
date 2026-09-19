import builtins
import copy
from types import SimpleNamespace

import pytest

import cascadekv.phi35_8k_capture_amendment as supplement


def _base_entries():
    entries = []
    for family, roles, indexes in (
        ("narrative", ("calibration_1", "calibration_2", "validation"), (13, 14, 15)),
        ("report", ("calibration_1", "calibration_2", "validation"), (16, 17, 18)),
        ("qa", ("calibration_1", "calibration_2", "validation"), (13, 14, 15)),
    ):
        for role, index in zip(roles, indexes, strict=True):
            for layer in supplement.LAYERS:
                entries.append({"path": f"artifacts/{family}_{role}_index{index}_layer{layer}.safetensors", "sha256": f"{len(entries):064x}"})
    return entries


def test_protocol_and_local_preflight_are_data_free(monkeypatch):
    supplement.validate_protocol()
    real_import = builtins.__import__
    forbidden = {"torch", "transformers", "datasets", "huggingface_hub"}
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"preflight imported forbidden stack: {name}")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = supplement.preflight(require_supplement_tag=False)
    assert result["protocol_sha256"] == supplement.sha256_path(supplement.PROTOCOL)
    assert {entry["path"] for entry in result["bound_files"]} >= {
        "cascadekv/phi35_8k_capture_amendment.py",
        "docs/phi35_8k_capture_amendment.md",
    }


def test_protocol_rejects_qa16_or_logical_accounting_mutation():
    protocol = supplement._json(supplement.PROTOCOL)
    broken = copy.deepcopy(protocol)
    broken["capture"]["qa16"]["dataset_index"] = 17
    with pytest.raises(supplement.SupplementError, match="capture"):
        supplement.validate_protocol(broken)
    broken = copy.deepcopy(protocol)
    broken["logical_capture"]["new_tensors"] = 6
    with pytest.raises(supplement.SupplementError, match="logical"):
        supplement.validate_protocol(broken)


@pytest.mark.parametrize(
    ("identifier", "mutate"),
    [
        ("attention", lambda value: value["capture"].update(attention="eager")),
        ("compute_dtype", lambda value: value["capture"].update(compute_dtype="bfloat16")),
        ("storage_dtype", lambda value: value["capture"].update(storage_dtype="bfloat16")),
        ("adapter", lambda value: value["capture"].update(adapter="different-adapter")),
        ("target_revision", lambda value: value["capture"]["target"].update(revision="0" * 40)),
        ("use_cache", lambda value: value["capture"].update(use_cache=True)),
        ("quantization", lambda value: value["capture"].update(quantization="int8")),
        ("qa15_tensor_sha", lambda value: value["qa15_historical_artifacts"]["0"].update(tensor_sha256="0" * 64)),
        ("qa15_provenance_sha", lambda value: value["qa15_historical_artifacts"]["0"].update(provenance_sha256="0" * 64)),
    ],
)
def test_protocol_rejects_every_material_capture_binding(identifier, mutate):
    broken = copy.deepcopy(supplement._json(supplement.PROTOCOL))
    mutate(broken)
    with pytest.raises(supplement.SupplementError):
        supplement.validate_protocol(broken)


def test_role_reference_is_explicitly_historical_validation_to_calibration_2(tmp_path):
    details = {layer: {"tensor_path": f"artifacts/qa_validation_index15_layer{layer}.safetensors", "provenance_path": f"artifacts/qa_validation_index15_layer{layer}.provenance.json", "tensor_sha256": supplement.QA15_TENSORS[layer], "provenance_sha256": supplement.QA15_PROVENANCE[layer]} for layer in supplement.LAYERS}
    references = supplement.write_references(tmp_path, details, "a" * 64, "b" * 64)
    assert len(references) == 5
    value = supplement._json(tmp_path / references[0]["path"])
    assert value["historical_role"] == "validation"
    assert value["amended_role"] == "calibration_2"
    assert value["tensor_bytes_reused_unchanged"] is True
    assert value["tensor_bytes_recaptured"] is False
    changed = copy.deepcopy(value); changed["historical_role"] = "calibration_2"
    (tmp_path / references[0]["path"]).write_text(supplement.canonical_json(changed))
    with pytest.raises(supplement.SupplementError, match="reference mismatch"):
        supplement.write_references(tmp_path, details, "a" * 64, "b" * 64)


def test_logical_inventory_excludes_qa14_and_has_40_reused_plus_five_new(monkeypatch):
    monkeypatch.setattr(supplement, "git_commit", lambda _ref: "a" * 40)
    refs = {layer: {"path": f"artifacts/ref{layer}.json", "sha256": f"a{layer:063x}"} for layer in supplement.LAYERS}
    qa16 = {layer: {"path": f"artifacts/new{layer}.safetensors", "sha256": f"b{layer:063x}", "provenance_path": f"artifacts/new{layer}.json", "provenance_sha256": f"c{layer:063x}"} for layer in supplement.LAYERS}
    manifest = supplement._logical_manifest({"complete_artifacts": _base_entries()}, refs, qa16, "d" * 64, "e" * 64)
    assert manifest["logical_artifact_count"] == 45
    assert manifest["logical_tensor_payload_bytes"] == 6794772480
    assert manifest["new_tensor_payload_bytes"] == 754974720
    assert sum(item["origin"] == "historical_unchanged" for item in manifest["artifacts"]) == 35
    assert sum(item["origin"] == "historical_role_rebound" for item in manifest["artifacts"]) == 5
    assert sum(item["origin"] == "supplement_capture" for item in manifest["artifacts"]) == 5
    assert not any("qa_calibration_2_index14" in str(item) for item in manifest["artifacts"])


@pytest.mark.parametrize("relative", (".", "child", ".."), ids=("exact", "descendant", "ancestor"))
def test_output_isolation_rejects_all_base_overlap_cases(tmp_path, monkeypatch, relative):
    base = tmp_path / "base-parent" / "base"; base.mkdir(parents=True)
    amendment = tmp_path / "separate-parent" / "amendment"; amendment.mkdir(parents=True)
    monkeypatch.setattr(supplement, "SOURCE_AMENDMENT_ROOT", amendment)
    with pytest.raises(supplement.SupplementError):
        supplement._output_isolated(base / relative, base)


@pytest.mark.parametrize("relative", (".", "child", ".."), ids=("exact", "descendant", "ancestor"))
def test_output_isolation_rejects_all_source_amendment_overlap_cases(tmp_path, monkeypatch, relative):
    base = tmp_path / "separate-parent" / "base"; base.mkdir(parents=True)
    amendment = tmp_path / "amendment-parent" / "amendment"; amendment.mkdir(parents=True)
    monkeypatch.setattr(supplement, "SOURCE_AMENDMENT_ROOT", amendment)
    with pytest.raises(supplement.SupplementError):
        supplement._output_isolated(amendment / relative, base)


def test_output_isolation_accepts_intended_separate_sibling(tmp_path, monkeypatch):
    base = tmp_path / "cascadekv_phi35_8k_capture"; base.mkdir()
    amendment = tmp_path / "cascadekv_phi35_8k_source_amendment"; amendment.mkdir()
    output = tmp_path / "cascadekv_phi35_8k_capture_amendment"
    monkeypatch.setattr(supplement, "SOURCE_AMENDMENT_ROOT", amendment)
    assert supplement._output_isolated(output, base) == output.resolve()


class _FakeIds:
    def to(self, _device):
        return self


def _mock_capture_dependencies(monkeypatch, tmp_path, *, reject_validation=False, resumed_layer=None):
    """Exercise only synthetic tiny files; no source/model/tensor payload is used."""
    import cascadekv

    c2 = SimpleNamespace()
    monkeypatch.setattr(cascadekv, "phi35_kaggle_c2", c2, raising=False)

    output = tmp_path / "output"
    base = tmp_path / "base"; base.mkdir()
    qualification = base / "qualification.json"; qualification.write_text("{}")
    source_manifest = tmp_path / "source.json"; source_manifest.write_text("{}")
    calls = []
    monkeypatch.setattr(supplement, "preflight", lambda: {"protocol_sha256": "a" * 64, "runtime_manifest_sha256": "b" * 64})
    monkeypatch.setattr(supplement, "_amended_sources", lambda _path: ({}, {"family": "qa"}))
    monkeypatch.setattr(supplement, "verify_base_capture", lambda *_args: ({"complete_artifacts": _base_entries()}, {}))
    monkeypatch.setattr(supplement, "write_references", lambda *_args: {layer: {"path": f"artifacts/ref{layer}.json", "sha256": f"{layer + 1:064x}"} for layer in supplement.LAYERS})
    monkeypatch.setattr(supplement, "_qa16_input", lambda _source: (_FakeIds(), {}))
    monkeypatch.setattr(supplement, "_supplement_provenance", lambda _source, _proof, layer, *_args: {"layer": layer})
    monkeypatch.setattr(supplement, "git_commit", lambda _ref: "f" * 40)
    c2.load_approved_qualification = lambda *_args: ({"backend_id": supplement.BACKEND_ID}, supplement.QUALIFICATION_SHA256)
    c2._load_pinned_model = lambda: object()
    c2.assert_capture_backend = lambda *_args: None
    c2._first_device = lambda _model: "synthetic"

    def capture_once(_model, _ids):
        calls.append(("forward", None))
        return {layer: object() for layer in supplement.LAYERS}

    def write_synthetic(artifact, provenance, _triplet, expected):
        layer = expected["layer"]
        calls.append(("write", layer))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"synthetic-{layer}".encode())
        payload = {**expected, "artifact_sha256": supplement.sha256_path(artifact)}
        provenance.write_text(supplement.canonical_json(payload))

    def validate_synthetic(artifact, provenance, expected):
        layer = expected["layer"]
        calls.append(("validate", layer))
        if reject_validation:
            raise RuntimeError("synthetic validator rejection")
        assert supplement._json(provenance) == expected
        return expected

    c2.capture_five_layers_post_rope_qkv = capture_once
    c2.write_artifact = write_synthetic
    c2.validate_artifact = validate_synthetic
    if resumed_layer is not None:
        artifact = output / "artifacts" / f"qa_validation_index16_layer{resumed_layer}.safetensors"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"synthetic-{resumed_layer}".encode())
        provenance = artifact.with_suffix(".provenance.json")
        provenance.write_text(supplement.canonical_json({"layer": resumed_layer, "artifact_sha256": supplement.sha256_path(artifact)}))
    return output, base, qualification, source_manifest, calls


def test_fresh_and_resumed_artifacts_share_validated_completed_records(monkeypatch, tmp_path):
    output, base, qualification, source_manifest, calls = _mock_capture_dependencies(monkeypatch, tmp_path, resumed_layer=0)
    supplement.capture(source_manifest=source_manifest, base_capture_root=base, qualification=qualification, output=output)
    assert calls.count(("forward", None)) == 1  # One forward when any layer is absent.
    assert calls[0] == ("validate", 0)  # Resumed record is validated before fresh writes.
    for layer in supplement.LAYERS:
        assert ("validate", layer) in calls
    for layer in supplement.LAYERS[1:]:
        assert calls.index(("write", layer)) < calls.index(("validate", layer))
    progress = supplement._json(output / "supplement_progress_manifest.json")
    assert len(progress["complete_new_artifacts"]) == 5
    assert all(set(record) == {"path", "sha256", "provenance_path", "provenance_sha256"} for record in progress["complete_new_artifacts"])
    assert (output / "final_amended_capture_manifest.json").is_file()


def test_fresh_artifact_validator_failure_blocks_final_manifest(monkeypatch, tmp_path):
    output, base, qualification, source_manifest, calls = _mock_capture_dependencies(monkeypatch, tmp_path, reject_validation=True)
    with pytest.raises(RuntimeError, match="validator rejection"):
        supplement.capture(source_manifest=source_manifest, base_capture_root=base, qualification=qualification, output=output)
    assert calls[:3] == [("forward", None), ("write", 0), ("validate", 0)]
    assert not (output / "supplement_progress_manifest.json").exists()
    assert not (output / "final_amended_capture_manifest.json").exists()


def test_cli_has_no_default_capture_arguments():
    script = (supplement.ROOT / "scripts/kaggle_phi35_8k_capture_amendment.py").read_text()
    assert "uv run python3" not in script
    parser_text = supplement.main.__code__.co_consts
    assert "--source-manifest" in parser_text and "--base-capture-root" in parser_text
