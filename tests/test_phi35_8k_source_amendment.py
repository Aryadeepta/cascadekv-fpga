import copy
import hashlib
import builtins

import numpy as np
import pytest

import cascadekv.phi35_8k_source_amendment as amendment


class _Tokenizer:
    """Synthetic tokenizer which deliberately maps fixture rows to token IDs."""
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def __call__(self, value, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": True, "max_length": amendment.MINIMUM}
        self.calls.append(value)
        return type("Encoded", (), {"input_ids": self.ids[value]})()


def _ids(seed):
    return [seed] * amendment.MINIMUM


def _fixture_rows(spec, *, qa_extra=(), duplicate_narrative=False):
    field = spec["field"]
    upper = max(20, max(qa_extra, default=0) + 1)
    rows = [{} for _ in range(upper)]
    if field == "text":
        indexes = (13, 14, 15)
    elif field == "report":
        indexes = (16, 17, 18)
    else:
        indexes = (13, 14, 15, *qa_extra)
    for index in indexes:
        rows[index] = {field: f"synthetic-{field}-{13 if field == 'context' and index == 14 else index}"}
    return rows


def _fixture_loader(rows):
    return lambda spec: rows[spec["field"]]


def _fixture_tokenizer(rows, *, duplicate_narrative=False):
    mapping = {}
    for field, family_seed in (("text", 100), ("report", 200), ("context", 300)):
        for index, row in enumerate(rows[field]):
            if field not in row:
                continue
            seed = family_seed + index
            if field == "context" and index == 14:
                seed = 313  # QA 13/14 token IDs are intentionally identical.
            elif field == "context" and index == 13:
                seed = 313
            elif duplicate_narrative and field == "text" and index == 14:
                seed = 113
            mapping[row[field]] = _ids(seed)
    return _Tokenizer(mapping)


def _prepared_inputs(monkeypatch, *, qa_extra=(16,), duplicate_narrative=False):
    specs = amendment.frozen_specs()
    rows = {
        "text": _fixture_rows(specs["narrative"], qa_extra=qa_extra),
        "report": _fixture_rows(specs["report"], qa_extra=qa_extra),
        "context": _fixture_rows(specs["qa"], qa_extra=qa_extra),
    }
    tokenizer = _fixture_tokenizer(rows, duplicate_narrative=duplicate_narrative)
    # The synthetic IDs do not claim to reproduce the real incident hashes.
    qa_hashes = {index: amendment.canonical_input_ids_sha256(tokenizer.ids[rows["context"][index]["context"]]) for index in (13, 14, 15)}
    monkeypatch.setattr(amendment, "OBSERVED_QA_HASHES", qa_hashes)
    return specs, rows, tokenizer


def test_canonical_input_hash_is_cpu_contiguous_int64_sha256():
    ids = np.arange(amendment.MINIMUM, dtype=np.int32).reshape(1, amendment.MINIMUM)
    expected = hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()
    assert amendment.canonical_input_ids_sha256(ids) == expected
    assert amendment.canonical_input_ids_sha256(list(range(amendment.MINIMUM))) == hashlib.sha256(np.arange(amendment.MINIMUM, dtype=np.int64).tobytes()).hexdigest()


def test_qa_duplicate_input_does_not_consume_slot_and_stops_at_third_unique(monkeypatch):
    specs, rows, tokenizer = _prepared_inputs(monkeypatch, qa_extra=(16, 17))
    selected, rejected, last = amendment._select_qa(specs["qa"], tokenizer, _fixture_loader(rows))
    assert [(item["dataset_index"], item["role"]) for item in selected] == [
        (13, "calibration_1"), (15, "calibration_2"), (16, "validation"),
    ]
    assert rejected[0]["rejection_reason"] == "duplicate_input"
    assert rejected[0]["dataset_index"] == 14
    assert rejected[0]["duplicate_of_dataset_index"] == 13
    assert rejected[0]["input_ids_sha256"] == selected[0]["proof"]["input_ids_sha256"]
    assert last == 16
    assert rows["context"][17]["context"] not in tokenizer.calls


def test_duplicate_rows_need_unique_input_hashes_not_unique_dataset_indexes(monkeypatch):
    specs, rows, tokenizer = _prepared_inputs(monkeypatch, qa_extra=(16,))
    first = amendment.inspect_candidate(rows["context"], tokenizer, specs["qa"], 13)
    second = amendment.inspect_candidate(rows["context"], tokenizer, specs["qa"], 14)
    assert first["input_ids_sha256"] == second["input_ids_sha256"]
    selected, rejected, _last = amendment._select_qa(specs["qa"], tokenizer, _fixture_loader(rows))
    assert [item["dataset_index"] for item in selected] == [13, 15, 16]
    assert [item["dataset_index"] for item in rejected] == [14]


def test_fixed_narrative_or_report_duplicate_fails_closed(monkeypatch):
    specs, rows, tokenizer = _prepared_inputs(monkeypatch, duplicate_narrative=True)
    with pytest.raises(amendment.SourceAmendmentError, match="duplicate"):
        amendment._select_fixed_family("narrative", specs["narrative"], tokenizer, _fixture_loader(rows))


def test_manifest_inventory_includes_qa_duplicate_and_global_hashes_are_unique(monkeypatch, tmp_path):
    specs, rows, tokenizer = _prepared_inputs(monkeypatch)
    monkeypatch.setattr(amendment, "preflight", lambda: {})
    freeze = {
        "tag": amendment.AMENDMENT_FREEZE_TAG,
        "commit": "a" * 40,
        "amendment_protocol_sha256": "b" * 64,
        "amendment_runtime_manifest_sha256": "c" * 64,
    }
    monkeypatch.setattr(amendment, "amendment_freeze_binding", lambda: freeze)
    monkeypatch.setattr(amendment, "KAGGLE_AMENDMENT_ROOT", tmp_path)
    manifest = amendment.select(output=tmp_path / "cascadekv_phi35_8k_dev_sources_v2.json", tokenizer=tokenizer, dataset_loader=_fixture_loader(rows))
    amendment.validate_manifest(manifest)
    qa_inventory = manifest["unavailable_inventory"]["qa"]
    assert 14 in qa_inventory["indices"]
    assert qa_inventory["next_untouched_index"] == 17
    assert len({item["input_ids_sha256"] for item in manifest["selected_sources"]}) == 9
    assert "synthetic-context-13" not in (tmp_path / "cascadekv_phi35_8k_dev_sources_v2.json").read_text()
    duplicate_hashes = copy.deepcopy(manifest)
    duplicate_hashes["selected_sources"][1]["input_ids_sha256"] = duplicate_hashes["selected_sources"][0]["input_ids_sha256"]
    duplicate_hashes["selected_sources"][1]["proof"]["input_ids_sha256"] = duplicate_hashes["selected_sources"][0]["input_ids_sha256"]
    duplicate_hashes["selected_sources"][1]["reproof"]["input_ids_sha256"] = duplicate_hashes["selected_sources"][0]["input_ids_sha256"]
    with pytest.raises(amendment.SourceAmendmentError, match="globally unique"):
        amendment.validate_manifest(duplicate_hashes)
    raw_text = copy.deepcopy(manifest)
    raw_text["selected_sources"][0]["raw_text"] = "synthetic fixture only"
    with pytest.raises(amendment.SourceAmendmentError, match="raw source text"):
        amendment.validate_manifest(raw_text)
    missing_freeze = copy.deepcopy(manifest)
    del missing_freeze["amendment_freeze"]
    with pytest.raises(amendment.SourceAmendmentError, match="schema"):
        amendment.validate_manifest(missing_freeze)
    wrong_freeze = copy.deepcopy(manifest)
    wrong_freeze["amendment_freeze"]["amendment_runtime_manifest_sha256"] = "d" * 64
    with pytest.raises(amendment.SourceAmendmentError, match="amendment freeze"):
        amendment.validate_manifest(wrong_freeze)
    wrong_c2 = copy.deepcopy(manifest)
    wrong_c2["c2_v2_freeze"]["runtime_manifest_sha256"] = "d" * 64
    with pytest.raises(amendment.SourceAmendmentError, match="C2-v2"):
        amendment.validate_manifest(wrong_c2)


def test_preflight_is_repository_only(monkeypatch):
    calls = []
    real_git = amendment.git_commit
    real_hash = amendment.sha256_path
    monkeypatch.setattr(amendment, "git_commit", lambda ref: calls.append(("git", ref)) or real_git(ref))
    monkeypatch.setattr(amendment, "sha256_path", lambda path: calls.append(("hash", path.name)) or real_hash(path))
    result = amendment.preflight(require_amendment_tag=False)
    assert result["base_c1_manifest_sha256"] == amendment.C1_MANIFEST_SHA256
    assert all(kind in {"git", "hash"} for kind, _ in calls)
    assert ("git", amendment.AMENDMENT_FREEZE_TAG) not in calls


def test_local_preflight_imports_no_network_tokenizer_dataset_or_model_stack(monkeypatch):
    real_import = builtins.__import__
    forbidden = {"datasets", "huggingface_hub", "torch", "transformers"}

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"local preflight imported forbidden dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    amendment.preflight(require_amendment_tag=False)


def test_preflight_requires_exact_c1_commit(monkeypatch):
    monkeypatch.setattr(amendment, "git_commit", lambda ref: "0" * 40 if ref == amendment.C1_TAG else amendment.C1_COMMIT)
    with pytest.raises(amendment.SourceAmendmentError, match="C1 freeze tag"):
        amendment.preflight(require_amendment_tag=False)


def test_preflight_requires_exact_c2_v2_commit(monkeypatch):
    def commits(ref):
        if ref == amendment.C1_TAG:
            return amendment.C1_COMMIT
        if ref == amendment.C2_TAG:
            return "0" * 40
        return amendment.C1_COMMIT
    monkeypatch.setattr(amendment, "git_commit", commits)
    with pytest.raises(amendment.SourceAmendmentError, match="C2-v2 freeze tag"):
        amendment.preflight(require_amendment_tag=False)


def test_amendment_protocol_binds_exact_incident_evidence():
    amendment.validate_amendment_protocol()
    changed = copy.deepcopy(amendment._load_json(amendment.AMENDMENT_PROTOCOL))
    changed["incident_evidence"]["qa_input_ids_sha256"]["14"] = "0" * 64
    with pytest.raises(amendment.SourceAmendmentError, match="incident evidence"):
        amendment.validate_amendment_protocol(changed)


def test_amendment_runtime_closure_is_complete_tracked_and_hash_valid():
    verified = amendment.verify_amendment_runtime_closure()
    paths = {entry["path"] for entry in verified}
    assert {
        "cascadekv/phi35_8k_source_amendment.py",
        "cascadekv/phi35_8k_source_selection.py",
        "cascadekv/phi35_phaseb.py",
        "cascadekv/runtime_provenance.py",
        "configs/cascadekv_phi35_8k_source_amendment_protocol.json",
        "configs/cascadekv_phi35_8k_phaseb_protocol.json",
        "configs/cascadekv_phi35_8k_phaseb_runtime_manifest.json",
        "configs/cascadekv_phi35_8k_kaggle_c2_protocol.json",
        "configs/cascadekv_phi35_8k_kaggle_c2_runtime_manifest.json",
        "results/cascadekv_phi35_8k_dev_sources.json",
        "pyproject.toml", "uv.lock", "scripts/kaggle_phi35_8k_source_amendment.py",
    } == paths


@pytest.mark.parametrize("mutated", [
    "cascadekv/phi35_8k_source_amendment.py",
    "cascadekv/phi35_8k_source_selection.py",
])
def test_runtime_closure_rejects_mutated_selection_code(monkeypatch, mutated):
    real_hash = amendment.sha256_path
    monkeypatch.setattr(
        amendment, "sha256_path",
        lambda path: "0" * 64 if path == amendment.ROOT / mutated else real_hash(path),
    )
    with pytest.raises(amendment.SourceAmendmentError, match="hash differs"):
        amendment.verify_amendment_runtime_closure()


@pytest.mark.parametrize("path", [
    amendment.C1_MANIFEST, amendment.C2_PROTOCOL, amendment.C2_RUNTIME_MANIFEST,
])
def test_preflight_rejects_mutated_immutable_bindings(monkeypatch, path):
    real_hash = amendment.sha256_path
    monkeypatch.setattr(amendment, "sha256_path", lambda value: "0" * 64 if value == path else real_hash(value))
    with pytest.raises(amendment.SourceAmendmentError, match="immutable prior SHA256"):
        amendment.preflight(require_amendment_tag=False)


def test_real_cli_preflight_requires_amendment_tag_at_head(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["phi35_8k_source_amendment", "preflight"])
    assert amendment.main() == 1
    assert "amendment freeze" in capsys.readouterr().out


def test_amendment_freeze_tag_must_resolve_exactly_to_head(monkeypatch):
    monkeypatch.setattr(
        amendment, "git_commit",
        lambda ref: "a" * 40 if ref == amendment.AMENDMENT_FREEZE_TAG else "b" * 40,
    )
    with pytest.raises(amendment.SourceAmendmentError, match="exactly to HEAD"):
        amendment.amendment_freeze_binding()


def test_cli_has_no_default_selection_operation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["phi35_8k_source_amendment"])
    monkeypatch.setattr(amendment, "select", lambda **_kwargs: pytest.fail("selection must be explicit"))
    with pytest.raises(SystemExit) as exc:
        amendment.main()
    assert exc.value.code == 2


def test_output_must_be_separate_kaggle_amendment_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(amendment, "KAGGLE_AMENDMENT_ROOT", tmp_path / "amendment")
    assert amendment.require_amendment_output(tmp_path / "amendment" / "out.json").name == "out.json"
    with pytest.raises(amendment.SourceAmendmentError):
        amendment.require_amendment_output(tmp_path / "cascadekv_phi35_8k_capture" / "out.json")
