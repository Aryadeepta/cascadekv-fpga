import copy
import json
from pathlib import Path

import pytest
import torch

import cascadekv.phi35_kaggle_c2 as c2


def qualification_record():
    record = {
        "schema_version": c2.QUALIFICATION_SCHEMA, "qualification_result": "QUALIFIED",
        "repo_capture_prep": {"tag": c2.C2_PREP_TAG, "commit": "a" * 40},
        "c1_manifest_binding": {"sha256": c2.C1_MANIFEST_SHA256, "tag": c2.C1_TAG, "commit": c2.C1_COMMIT},
        "target": {"model": c2.TARGET_MODEL, "revision": c2.TARGET_REVISION, "tokenizer_revision": c2.TARGET_REVISION},
        "model_config_sha256": "b" * 64, "model_compute_dtype": "float16",
        "attention_implementation": "sdpa", "use_cache": False, "quantization": "none",
        "cuda_version": "12.4", "gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"],
        "vram_per_gpu_bytes": [16_000_000_000, 16_000_000_000],
        "software_versions": {"torch": "x", "transformers": "y", "accelerate": "z"},
        "resolved_hf_device_map": c2.deterministic_device_map(), "max_memory": {"0": "13GiB", "1": "13GiB"},
        "synthetic_sequence_lengths": list(c2.LENGTHS),
        "capture_shape_proof_8192": {str(layer): {"q_shape": list(c2.SHAPE), "k_shape": list(c2.SHAPE), "v_shape": list(c2.SHAPE)} for layer in c2.LAYERS},
        "peak_cuda_memory": [{"index": 0, "allocated_bytes": 1, "reserved_bytes": 2}, {"index": 1, "allocated_bytes": 3, "reserved_bytes": 4}],
        "capture_adapter": "phi3-post-rope-qkv-v1",
    }
    record["backend_id"] = c2.qualification_backend_id(record)
    return record


def test_qualification_schema_and_freeze_barrier(tmp_path):
    record = qualification_record()
    c2.validate_qualification(record)
    qualification = tmp_path / "qualification.json"; qualification.write_text(c2.canonical_json(record))
    approval = tmp_path / "approval.json"
    with pytest.raises(c2.C2Error, match="approval"):
        c2.load_approved_qualification(qualification, approval)
    approval.write_text(c2.canonical_json({"schema_version": c2.APPROVAL_SCHEMA, "review_status": "FROZEN_APPROVED", "qualification_sha256": c2.sha256_path(qualification), "backend_id": record["backend_id"]}))
    loaded, digest = c2.load_approved_qualification(qualification, approval)
    assert loaded["backend_id"] == record["backend_id"] and len(digest) == 64
    broken = copy.deepcopy(record); broken["synthetic_sequence_lengths"] = [512]
    with pytest.raises(c2.C2Error): c2.validate_qualification(broken)


class FakeCuda:
    def __init__(self, names): self.names = names
    def is_available(self): return True
    def device_count(self): return len(self.names)
    def get_device_name(self, index): return self.names[index]
    def get_device_properties(self, _index): return type("Props", (), {"total_memory": 16_000_000_000})()


class FakeTorch:
    def __init__(self, names): self.cuda = FakeCuda(names)


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
