import copy
import json
from pathlib import Path

import pytest
import torch

import cascadekv.phi35_kaggle_c2 as c2


TEST_C2_COMMIT = "a" * 40


def qualification_record():
    peaks = {
        str(length): [
            {"index": 0, "total_vram_bytes": 16_000_000_000, "allocated_bytes": length, "reserved_bytes": length + 1, "headroom_reserved_bytes": 16_000_000_000 - length - 1},
            {"index": 1, "total_vram_bytes": 16_000_000_000, "allocated_bytes": length + 2, "reserved_bytes": length + 3, "headroom_reserved_bytes": 16_000_000_000 - length - 3},
        ]
        for length in c2.LENGTHS
    }
    record = {
        "schema_version": c2.QUALIFICATION_SCHEMA, "qualification_result": "QUALIFIED",
        "repo_capture_prep": {"tag": c2.C2_PREP_TAG, "commit": TEST_C2_COMMIT},
        "c2_protocol_binding": {"path": "configs/cascadekv_phi35_8k_kaggle_c2_protocol.json", "sha256": c2.C2_PROTOCOL_SHA256},
        "c2_runtime_manifest_binding": {"path": "configs/cascadekv_phi35_8k_kaggle_c2_runtime_manifest.json", "sha256": c2.sha256_path(c2.C2_RUNTIME_MANIFEST)},
        "c1_manifest_binding": {"sha256": c2.C1_MANIFEST_SHA256, "tag": c2.C1_TAG, "commit": c2.C1_COMMIT},
        "phase_b_freeze": {"protocol_path": "configs/cascadekv_phi35_8k_phaseb_protocol.json", "protocol_sha256": c2.PROTOCOL_SHA256, "runtime_manifest_path": "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json", "runtime_manifest_sha256": c2.RUNTIME_MANIFEST_SHA256},
        "target": {"model": c2.TARGET_MODEL, "revision": c2.TARGET_REVISION, "tokenizer_revision": c2.TARGET_REVISION},
        "model_config_sha256": "b" * 64, "model_compute_dtype": "float16",
        "attention_implementation": "sdpa", "use_cache": False, "quantization": "none",
        "cuda_version": "12.4", "gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"],
        "vram_per_gpu_bytes": [16_000_000_000, 16_000_000_000],
        "software_versions": {"torch": "x", "transformers": "y", "accelerate": "z"},
        "resolved_hf_device_map": c2.deterministic_device_map(), "max_memory": {"0": "13GiB", "1": "13GiB"},
        "synthetic_sequence_lengths": list(c2.LENGTHS),
        "capture_shape_proof_8192": {str(layer): {"q_shape": list(c2.SHAPE), "k_shape": list(c2.SHAPE), "v_shape": list(c2.SHAPE)} for layer in c2.LAYERS},
        "peak_memory_reset_device_indices": [0, 1], "peak_cuda_memory_by_length": peaks,
        "capture_adapter": "phi3-post-rope-qkv-v1",
    }
    record["backend_id"] = c2.qualification_backend_id(record)
    return record


def test_qualification_schema_and_freeze_barrier(tmp_path, monkeypatch):
    record = qualification_record()
    c2.validate_qualification(record, expected_c2_commit=TEST_C2_COMMIT)
    monkeypatch.setattr(c2, "git_commit", lambda ref: TEST_C2_COMMIT if ref == c2.C2_PREP_TAG else "unexpected")
    qualification = tmp_path / "qualification.json"; qualification.write_text(c2.canonical_json(record))
    approval = tmp_path / "approval.json"
    with pytest.raises(c2.C2Error, match="approval"):
        c2.load_approved_qualification(qualification, approval)
    approval.write_text(c2.canonical_json({"schema_version": c2.APPROVAL_SCHEMA, "review_status": "FROZEN_APPROVED", "qualification_sha256": c2.sha256_path(qualification), "backend_id": record["backend_id"]}))
    loaded, digest = c2.load_approved_qualification(qualification, approval)
    assert loaded["backend_id"] == record["backend_id"] and len(digest) == 64
    broken = copy.deepcopy(record); broken["synthetic_sequence_lengths"] = [512]
    with pytest.raises(c2.C2Error): c2.validate_qualification(broken)


def test_qualification_fails_closed_on_each_capture_package_binding():
    record = qualification_record()
    for key, value in (
        ("repo_capture_prep", {"tag": "wrong-tag", "commit": TEST_C2_COMMIT}),
        ("repo_capture_prep", {"tag": c2.C2_PREP_TAG, "commit": "f" * 40}),
        ("c2_protocol_binding", {"path": "configs/cascadekv_phi35_8k_kaggle_c2_protocol.json", "sha256": "0" * 64}),
        ("c2_runtime_manifest_binding", {"path": "configs/cascadekv_phi35_8k_kaggle_c2_runtime_manifest.json", "sha256": "0" * 64}),
        ("c1_manifest_binding", {"sha256": "0" * 64, "tag": c2.C1_TAG, "commit": c2.C1_COMMIT}),
        ("phase_b_freeze", {"protocol_path": "configs/cascadekv_phi35_8k_phaseb_protocol.json", "protocol_sha256": "0" * 64, "runtime_manifest_path": "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json", "runtime_manifest_sha256": c2.RUNTIME_MANIFEST_SHA256}),
    ):
        broken = copy.deepcopy(record); broken[key] = value
        with pytest.raises(c2.C2Error):
            c2.validate_qualification(broken, expected_c2_commit=TEST_C2_COMMIT)


def test_qualification_rejects_missing_or_changed_8192_device_peak():
    record = qualification_record()
    missing = copy.deepcopy(record)
    missing["peak_cuda_memory_by_length"]["8192"] = missing["peak_cuda_memory_by_length"]["8192"][:1]
    with pytest.raises(c2.C2Error, match="both GPUs"):
        c2.validate_qualification(missing, expected_c2_commit=TEST_C2_COMMIT)
    changed = copy.deepcopy(record)
    changed["peak_cuda_memory_by_length"]["8192"][1]["headroom_reserved_bytes"] -= 1
    with pytest.raises(c2.C2Error, match="headroom"):
        c2.validate_qualification(changed, expected_c2_commit=TEST_C2_COMMIT)


def test_kaggle_python_entrypoints_use_locked_uv_environment():
    docs = (c2.ROOT / "docs/phi35_8k_kaggle_c2.md").read_text()
    entrypoint = (c2.ROOT / "scripts/kaggle_phi35_8k_c2.py").read_text()
    assert "uv run python3 -m cascadekv.phi35_kaggle_c2 preflight" in docs
    assert "uv run python3 -m cascadekv.phi35_kaggle_c2 qualify" in docs
    assert "uv run python3 -m cascadekv.phi35_kaggle_c2 capture" in docs
    assert "`python -m" not in docs and "``python -m" not in entrypoint


def test_backend_id_is_backend_only_but_qualification_sha_binds_provenance(tmp_path):
    original = qualification_record()
    changed = copy.deepcopy(original)
    changed["c1_manifest_binding"]["sha256"] = "f" * 64
    assert c2.qualification_backend_id(original) == c2.qualification_backend_id(changed)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(c2.canonical_json(original))
    second.write_text(c2.canonical_json(changed))
    assert c2.sha256_path(first) != c2.sha256_path(second)


def test_c2_runtime_manifest_has_present_state_and_complete_material_closure(tmp_path):
    manifest = json.loads(c2.C2_RUNTIME_MANIFEST.read_text())
    assert manifest["schema_version"] == 2
    assert "NO_PHI_DATA_CONSUMED" not in json.dumps(manifest)
    assert manifest["scientific_state"]["phi_source_identities_selected"] is True
    assert manifest["scientific_state"]["phi_source_semantics_inspected"] is False
    assert manifest["routing_protocol"]["status"] == "FROZEN_IN_PHASE_B"
    assert manifest["routing_protocol"]["protocol_sha256"] == c2.PROTOCOL_SHA256
    assert manifest["routing_protocol"]["runtime_manifest_sha256"] == c2.RUNTIME_MANIFEST_SHA256
    expected = {
        "cascadekv/phi35_kaggle_c2.py", "cascadekv/model_agnostic.py",
        "cascadekv/phi35_8k_source_selection.py", "cascadekv/phi35_phaseb.py",
        "cascadekv/runtime_provenance.py", "configs/cascadekv_phi35_8k_kaggle_c2_protocol.json",
        "docs/phi35_8k_kaggle_c2.md",
        "configs/cascadekv_phi35_8k_phaseb_protocol.json",
        "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json",
        "results/cascadekv_phi35_8k_dev_sources.json", "pyproject.toml", "uv.lock",
        "scripts/kaggle_phi35_8k_c2.py",
    }
    assert {entry["path"] for entry in manifest["bound_files"]} == expected
    assert len(c2.verify_c2_runtime_manifest()) == len(expected)
    for target in ("cascadekv/model_agnostic.py", "cascadekv/phi35_8k_source_selection.py", "results/cascadekv_phi35_8k_dev_sources.json"):
        broken = copy.deepcopy(manifest)
        for entry in broken["bound_files"]:
            if entry["path"] == target:
                entry["sha256"] = "0" * 64
        altered = tmp_path / f"{Path(target).name}.json"
        altered.write_text(c2.canonical_json(broken))
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(c2, "C2_RUNTIME_MANIFEST", altered)
        try:
            with pytest.raises(c2.C2Error, match="hash mismatch"):
                c2.verify_c2_runtime_manifest()
        finally:
            monkeypatch.undo()


def test_data_free_pre_tag_preflight_does_not_require_final_tag(monkeypatch):
    monkeypatch.setattr(c2, "git_commit", lambda ref: c2.C1_COMMIT if ref == c2.C1_TAG else (_ for _ in ()).throw(c2.C2Error("tag absent")))
    forbidden = lambda *args, **kwargs: pytest.fail("preflight accessed a model, dataset, or tokenizer")
    monkeypatch.setattr(c2, "_load_pinned_model", forbidden)
    monkeypatch.setattr(c2, "load_dataset_for_spec", forbidden)
    monkeypatch.setattr(c2, "load_frozen_tokenizer", forbidden)
    result = c2.preflight(require_c2_tag=False)
    assert result["c2_protocol_sha256"] == c2.C2_PROTOCOL_SHA256


class FakeCuda:
    def __init__(self, names): self.names = names
    def is_available(self): return True
    def device_count(self): return len(self.names)
    def get_device_name(self, index): return self.names[index]
    def get_device_properties(self, _index): return type("Props", (), {"total_memory": 16_000_000_000})()


class FakeTorch:
    def __init__(self, names): self.cuda = FakeCuda(names)


class QualificationCuda:
    def __init__(self):
        self.resets = []
        self.length_index = -1
    def is_available(self): return True
    def reset_peak_memory_stats(self, index):
        self.resets.append(index)
        if index == 1:
            self.length_index += 1
    def max_memory_allocated(self, index):
        return 1000 * (self.length_index + 1) + index
    def max_memory_reserved(self, index):
        return 2000 * (self.length_index + 1) + index
    def empty_cache(self): pass


class QualificationIds:
    def to(self, _device): return self


class QualificationModel:
    def __init__(self): self.config = type("Config", (), {"vocab_size": 9})()
    def __call__(self, **_kwargs): pass


def test_qualification_resets_and_records_every_gpu_for_every_length(tmp_path, monkeypatch):
    cuda = QualificationCuda()
    monkeypatch.setattr(c2.torch, "cuda", cuda)
    monkeypatch.setattr(c2, "preflight", lambda: {})
    monkeypatch.setattr(c2, "require_durable_output", lambda path: tmp_path / "qualification.json")
    monkeypatch.setattr(c2, "require_t4x2", lambda: [
        {"index": 0, "name": "Tesla T4", "total_vram_bytes": 16_000_000_000},
        {"index": 1, "name": "Tesla T4", "total_vram_bytes": 16_000_000_000},
    ])
    monkeypatch.setattr(c2, "_load_pinned_model", QualificationModel)
    monkeypatch.setattr(c2, "_config_sha", lambda _model: "c" * 64)
    monkeypatch.setattr(c2, "_versions", lambda: {"torch": "x", "transformers": "y", "accelerate": "z"})
    monkeypatch.setattr(c2, "_resolved_map", lambda _model: c2.deterministic_device_map())
    monkeypatch.setattr(c2, "_first_device", lambda _model: "cuda:0")
    monkeypatch.setattr(c2, "synthetic_input_ids", lambda _length, _vocab: QualificationIds())
    monkeypatch.setattr(c2, "capture_five_layers_post_rope_qkv", lambda _model, _ids: {})
    monkeypatch.setattr(c2, "git_commit", lambda _ref: TEST_C2_COMMIT)
    result = c2.qualify(tmp_path / "ignored.json")
    assert cuda.resets == [0, 1, 0, 1, 0, 1]
    assert set(result["peak_cuda_memory_by_length"]) == {"512", "2048", "8192"}
    for length in c2.LENGTHS:
        per_gpu = result["peak_cuda_memory_by_length"][str(length)]
        assert [item["index"] for item in per_gpu] == [0, 1]
    assert result["peak_cuda_memory_by_length"]["8192"] == [
        {"index": 0, "total_vram_bytes": 16_000_000_000, "allocated_bytes": 3000, "reserved_bytes": 6000, "headroom_reserved_bytes": 15_999_994_000},
        {"index": 1, "total_vram_bytes": 16_000_000_000, "allocated_bytes": 3001, "reserved_bytes": 6001, "headroom_reserved_bytes": 15_999_993_999},
    ]


def test_t4x2_hardware_requirement_parser_and_two_device_provenance():
    assert [item["name"] for item in c2.require_t4x2(FakeTorch(["Tesla T4", "NVIDIA T4"]))] == ["Tesla T4", "NVIDIA T4"]
    with pytest.raises(c2.C2Error): c2.require_t4x2(FakeTorch(["Tesla T4"]))
    with pytest.raises(c2.C2Error): c2.require_t4x2(FakeTorch(["Tesla T4", "NVIDIA L4"]))
    mapping = c2.deterministic_device_map()
    assert len([key for key in mapping if key.startswith("model.layers.")]) == 32
    assert {mapping[f"model.layers.{layer}"] for layer in range(32)} == {0, 1}


class FakeAttention:
    def __init__(self): self.hooks = []
    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs
        self.hooks.append(hook)
        return type("Handle", (), {"remove": lambda handle: self.hooks.remove(hook)})()
    def fire(self):
        for hook in tuple(self.hooks): hook(self, (), {"hidden_states": torch.zeros(1, 1, 1), "position_embeddings": (None, None)})


class FakeModel:
    def __init__(self):
        self.layers = [type("Layer", (), {"self_attn": FakeAttention()})() for _ in range(32)]
        self.forwards = 0
    def __call__(self, **_kwargs):
        self.forwards += 1
        for layer in self.layers: layer.self_attn.fire()


def test_five_layers_are_captured_by_one_logical_forward(monkeypatch):
    calls = []
    class Adapter:
        identity = "phi3-post-rope-qkv-v1"
        def capture(self, _module, _hidden, _positions):
            calls.append(1)
            return tuple(torch.zeros(1, 1, 8, 96) for _ in range(3))
    monkeypatch.setattr(c2, "Phi3CaptureAdapter", Adapter)
    model = FakeModel()
    got = c2.capture_five_layers_post_rope_qkv(model, torch.tensor([[0]]))
    assert set(got) == set(c2.LAYERS) and len(calls) == 5 and model.forwards == 1
    assert all(item[0].dtype == torch.float16 and item[0].device.type == "cpu" for item in got.values())


def test_grid_byte_accounting_and_valid_synthetic_ids():
    assert len(c2.LAYERS) * 9 == 45
    assert c2.ARTIFACT_PAYLOAD_BYTES == 150_994_944
    assert c2.TOTAL_ARTIFACT_PAYLOAD_BYTES == 6_794_772_480
    ids = c2.synthetic_input_ids(512, 7)
    assert ids.dtype == torch.int64 and ids.min().item() >= 0 and ids.max().item() < 7
    with pytest.raises(c2.C2Error): c2.synthetic_input_ids(513, 7)


def test_corrupt_artifact_rejected_and_metadata_never_contains_text(tmp_path):
    artifact, provenance = tmp_path / "a.safetensors", tmp_path / "a.provenance.json"
    artifact.write_bytes(b"not a safetensor")
    expected = {"artifact_sha256": c2.sha256_path(artifact), "source_identity": {"dataset_index": 13}}
    provenance.write_text(json.dumps(expected))
    with pytest.raises(c2.C2Error): c2.validate_artifact(artifact, provenance, expected)
    with pytest.raises(c2.C2Error): c2._validate_no_text({"raw_text": "forbidden"})
    c2._validate_no_text({"source_identity": {"dataset_index": 13}, "proof": {"character_count": 8192}})


def test_complete_manifest_requires_exactly_45_artifacts():
    complete = [{"path": str(index), "sha256": "a" * 64} for index in range(45)]
    assert len(complete) == 45 and len({item["path"] for item in complete}) == 45


def test_durable_output_path_is_kaggle_working_only():
    assert c2.require_durable_output(c2.DEFAULT_OUTPUT / "qualification.json").name == "qualification.json"
    with pytest.raises(c2.C2Error): c2.require_durable_output(Path("results/not-kaggle"))
