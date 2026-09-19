import copy
import json
from pathlib import Path

import pytest

from cascadekv.phi35_8k_source_selection import (
    FAMILIES, MINIMUM, SourceSelectionError, execute, frozen_specs,
    next_uninspected_index, validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/cascadekv_phi35_8k_dev_sources.json"


def _frozen_manifest():
    return json.loads(MANIFEST.read_text())


class _Tokenizer:
    def __call__(self, value, **_kwargs):
        # Synthetic length-only fixture: no test assertion uses source contents.
        return type("Encoded", (), {"input_ids": [0] * min(MINIMUM, len(value))})()


def _dataset(frontier, eligible_offsets):
    rows = [{} for _ in range(frontier)]
    for offset in range(8):
        # A character-count-only synthetic payload.  The resolver sees no fixture metadata.
        rows.append({"text": "x" * (MINIMUM if offset in eligible_offsets else MINIMUM - 1),
                     "report": "x" * (MINIMUM if offset in eligible_offsets else MINIMUM - 1),
                     "context": "x" * (MINIMUM if offset in eligible_offsets else MINIMUM - 1)})
    return rows


def test_c1_manifest_is_text_free_ascending_and_stops_after_third_eligible(tmp_path):
    plans = {"narrative": {0, 1, 2}, "report": {2, 3, 4}, "qa": {0, 1, 2}}

    def loader(spec):
        family = next(name for name, field in (("narrative", "text"), ("report", "report"), ("qa", "context")) if spec["field"] == field)
        return _dataset(spec["first_permitted_index"], plans[family])

    manifest = execute(output=tmp_path / "sources.json", tokenizer=_Tokenizer(), dataset_loader=loader)
    validate_manifest(manifest)
    assert len(manifest["selected_sources"]) == 9
    for family in FAMILIES:
        choices = [item for item in manifest["selected_sources"] if item["family"] == family]
        assert [item["role"] for item in choices] == ["calibration_1", "calibration_2", "validation"]
        assert manifest["last_inspected_index"][family] == choices[-1]["dataset_index"]
        assert all(item["dataset_index"] <= choices[-1]["dataset_index"] for item in manifest["rejected_sources"][family])
    rendered = (tmp_path / "sources.json").read_text()
    assert "xxxxxxxx" not in rendered


def test_c1_validator_rejects_any_post_validation_proof_or_revision_change(tmp_path):
    def loader(spec):
        eligible_offsets = {2, 3, 4} if spec["field"] == "report" else {0, 1, 2}
        return _dataset(spec["first_permitted_index"], eligible_offsets)

    manifest = execute(output=tmp_path / "sources.json", tokenizer=_Tokenizer(), dataset_loader=loader)
    future = copy.deepcopy(manifest)
    future["rejected_sources"]["narrative"].append({"dataset_index": future["last_inspected_index"]["narrative"] + 1, "proof": {"field_exists": True, "is_python_string": True, "character_count": MINIMUM - 1, "bounded_frozen_tokenizer_length": MINIMUM - 1, "at_least_8192": False}})
    with pytest.raises(SourceSelectionError):
        validate_manifest(future)
    wrong_revision = copy.deepcopy(manifest)
    wrong_revision["target"]["tokenizer_revision"] = "not-the-frozen-revision"
    with pytest.raises(SourceSelectionError):
        validate_manifest(wrong_revision)


def test_frozen_prior_and_combined_inventories_and_next_untouched_frontiers():
    specs = frozen_specs()
    manifest = _frozen_manifest()
    validate_manifest(manifest)
    expected_prior = {"narrative": list(range(13)), "report": list(range(14)), "qa": list(range(13))}
    expected_combined = {"narrative": list(range(16)), "report": list(range(19)), "qa": list(range(16))}
    expected_next = {"narrative": 16, "report": 19, "qa": 16}
    for family in FAMILIES:
        inventory = manifest["unavailable_inventory"][family]
        assert specs[family]["prior_unavailable_indices"] == expected_prior[family]
        assert inventory["prior_unavailable_indices"] == expected_prior[family]
        assert inventory["indices"] == expected_combined[family]
        assert next_uninspected_index(manifest["last_inspected_index"][family]) == expected_next[family]
    assert manifest["unavailable_inventory"]["report"]["prior_unavailable_indices"].count(4) == 1


def test_frozen_manifest_selected_and_rejected_identities_are_unchanged():
    manifest = _frozen_manifest()
    selected = {family: [] for family in FAMILIES}
    for source in manifest["selected_sources"]:
        selected[source["family"]].append(source["dataset_index"])
        assert source["proof"] == source["reproof"]
    assert selected == {"narrative": [13, 14, 15], "report": [16, 17, 18], "qa": [13, 14, 15]}
    assert {family: [item["dataset_index"] for item in rejected]
            for family, rejected in manifest["rejected_sources"].items()} == {
                "narrative": [], "report": [14, 15], "qa": [],
            }


def test_validator_rejects_dropped_index_zero_from_prior_inventory():
    manifest = _frozen_manifest()
    broken = copy.deepcopy(manifest)
    broken["unavailable_inventory"]["narrative"]["prior_unavailable_indices"].remove(0)
    with pytest.raises(SourceSelectionError):
        validate_manifest(broken)


def test_validator_rejects_dropped_nonzero_prior_identity():
    manifest = _frozen_manifest()
    broken = copy.deepcopy(manifest)
    broken["unavailable_inventory"]["report"]["prior_unavailable_indices"].remove(12)
    with pytest.raises(SourceSelectionError):
        validate_manifest(broken)


def test_validator_rejects_invented_post_frontier_prior_identity():
    manifest = _frozen_manifest()
    broken = copy.deepcopy(manifest)
    broken["unavailable_inventory"]["qa"]["prior_unavailable_indices"].append(16)
    with pytest.raises(SourceSelectionError):
        validate_manifest(broken)
