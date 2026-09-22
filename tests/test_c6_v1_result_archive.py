import hashlib
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

from cascadekv.stdout_artifact_recovery import recover


ROOT = Path(__file__).parents[1]
TARBALL = ROOT / "artifacts/c6-v1/cascadekv-c6-result.tar.gz"
STDOUT = ROOT / "artifacts/c6-v1/evidence/c6_v1_kaggle_stdout.md"
RESULT = ROOT / "results/archive/cascadekv_phi35_8k_c6_v1_result.json"
POSTMORTEM = ROOT / "results/archive/cascadekv_phi35_8k_c6_v1_postmortem.json"
EXPECTED = {
    "SHA256SUMS": "bd8beaff33248bd09ebbf1af52e68d8dc9546893163bbdf395c927e6bd4d181e",
    "development.json": "88bed81c5e92a38c2d252f39de1c7f7c91c812ceceafff91ceda04c015c93e47",
    "development_capture_manifest.json": "67c166fb16127846004b81879db1b0d0b3537d24f73e04a4cc43b5e17a003f98",
    "development_sources.json": "4dc6ddce6146f5e5150faccb14d29bbd7a7e054b472e60fbef83320a3d96925e",
    "holdout.json": "ed4a8a8fc56655249f0e2b996dc14fffcab8ad08935d46836bc19113e4c54f49",
    "holdout_capture_manifest.json": "6ec0f682b48e8e5facb73d29699a33f53e1eb908038027a23df01f0404ea9658",
    "protocol.json": "78b7b61e0cf4f3304496a9277e6f2fd966c93403f56349d235bf3ecf0fdc615a",
    "qualification.json": "f4703bdcd477ee91f108dde4415e14831ec20a8b156fe59876d8e3a7699bc07e",
    "runtime_manifest.json": "c19665dae088682758880a0b9dd8635b08de89339835bc737a13c5538f250cf2",
    "test.json": "122a3480c44072d886d6bcb4f379aaf3d3d13cb3c0d0be7ab79a67b91916a9d3",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def members():
    with tarfile.open(TARBALL) as archive:
        return {member.name: archive.extractfile(member).read() for member in archive.getmembers()}


def load(name):
    return json.loads(members()[name])


def test_original_container_members_sums_and_stdout_provenance(tmp_path):
    assert TARBALL.stat().st_size == 74912
    assert digest(TARBALL.read_bytes()) == "4e95dfde98e1e35a61dd08dffed9e7ef93ca466791485f9068392d1b2bded685"
    original = members()
    assert set(original) == set(EXPECTED)
    assert {name: digest(data) for name, data in original.items()} == EXPECTED
    sums = dict(line.split(maxsplit=1) for line in original["SHA256SUMS"].decode().splitlines())
    assert sums == {value: name for name, value in EXPECTED.items() if name != "SHA256SUMS"}
    assert digest(STDOUT.read_bytes()) == "c8efdf27958af3de3c31f68456bedc7debb6ed9b067e273e155519097c391af4"
    assert digest(STDOUT.read_bytes()) != "c8b43eb0d8d0a809a207f49ce29ddc584948228d2db53c0ba30ffa0536ee44ad"
    restored = recover(STDOUT.read_text(), tmp_path / "restored")
    assert set(restored) == set(original)
    assert {name: path.read_bytes() for name, path in restored.items()} == original


def test_c6_result_firewall_holdout_and_no_raw_text():
    dev, test, holdout, sources = (load(name) for name in ("development.json", "test.json", "holdout.json", "development_sources.json"))
    assert dev["classification"] == "C6-DEVELOPMENT-GO"
    assert dev["development_passing_targets"] == ["T5"]
    assert dev["selected_target"] == test["selected_target"] == holdout["selected_target"] == "T5"
    assert test["classification"] == "C6-PROSPECTIVE-NO-PASS"
    firewall = test["firewall_evidence"]
    assert firewall["optimizer_rerun"] is False and firewall["schedule_unchanged"] is True
    assert firewall["opened_development_tensor_identities"] == []
    assert firewall["schedule_sha256_before"] == firewall["schedule_sha256_after"] == "0f90a03895a02ce1efe2619cba25995ec7e6118e28a9fb5592cd3a15f03dbbdd"
    assert {row["family"]: (row["dataset_index"], row["input_ids_sha256"]) for row in holdout["selected_sources"]} == {
        "narrative": (18, "f1a03f658277e09e34ba9ef2cfdd660da932ff2fb82c2e0f05958584b57746b2"),
        "qa": (19, "6cf71fff9851c3a858f563e3b12d5f4d24ad1f0ca8bc726147aa29b50d76bb28"),
        "report": (22, "eb76c0dde3ee100dbfeba31ab3d9cbe018712568c8ab1e9b8ca4297d5c2ce63b"),
    }
    assert holdout["rejected_sources"] == {"narrative": [], "qa": [], "report": []}
    assert holdout["next_untouched_frontier"] == {"narrative": 19, "qa": 20, "report": 23}
    assert holdout["raw_text_persisted"] is False and "raw_text" not in json.dumps(sources, sort_keys=True)


def test_t5_gates_schedule_actions_and_layers():
    dev, test = load("development.json"), load("test.json")
    development, prospective = dev["development_target_metrics"]["T5"], test["target_test_metrics"]["T5"]
    assert development == {"mean_cosine": .9863149033600671, "mean_relative_l2": .11671852580316984, "mean_total_kv_bytes": 485372.5788888889, "observation_count": 7200}
    assert prospective == {"mean_cosine": .9847454029756287, "mean_relative_l2": .12684388312686273, "mean_total_kv_bytes": 485284.0777777778, "observation_count": 1440}
    base, pbase = dev["development_baseline_metrics"], test["baseline_test_metrics"]
    assert development["mean_cosine"] >= .985 and development["mean_relative_l2"] <= .12
    assert development["mean_relative_l2"] < base["uniform10"]["mean_relative_l2"] and development["mean_total_kv_bytes"] < base["flat5"]["mean_total_kv_bytes"]
    assert prospective["mean_cosine"] < .985 and prospective["mean_relative_l2"] > .12
    assert prospective["mean_relative_l2"] < pbase["uniform10"]["mean_relative_l2"] and prospective["mean_total_kv_bytes"] < pbase["flat5"]["mean_total_kv_bytes"]
    table = dev["schedules"]["T5"]["table"]
    assert Counter(table.values()) == {"A0": 5, "A1": 11, "A2": 6, "A3": 9, "A4": 80, "A6": 11, "A7": 1, "A8": 6, "A9": 9, "A10": 17, "A11": 5}
    assert len(table) == 160 and sum(action in {"A10", "A11"} for action in table.values()) == 22
    assert all(int(cell.split(":")[0]) == 0 for cell, action in table.items() if action in {"A10", "A11"})
    for layer, expected in {0: .29791271631291294, 8: .069572871993719, 16: .06630654037917003, 24: .06250550337153506, 31: .08729499695851221}.items():
        values = [dev["action_cell_statistics"][f"{layer}:{head}:{table[f'{layer}:{head}']}"]["relative_l2"] for head in range(32)]
        assert sum(values) / 32 == expected


def test_archive_binds_frozen_tag_and_historical_c5_transport_label():
    result, postmortem = json.loads(RESULT.read_text()), json.loads(POSTMORTEM.read_text())
    assert subprocess.check_output(["git", "rev-parse", "cascadekv-phi35-8k-c6-freeze-v1^{}"], cwd=ROOT, text=True).strip() == result["frozen_scientific_execution"]["freeze_commit"]
    assert result["stdout_recovery"]["canonical_transport_schema_label"] == "cascadekv-c5-stdout-recovery-v1"
    assert result["stdout_recovery"]["transport_label_is_historical_c5_framing_not_experiment_identity"] is True
    assert result["original_kaggle_result_tarball"]["locally_byte_verified"] is True
    assert result["repository_stdout_evidence"]["sha256"] == digest(STDOUT.read_bytes())
    assert postmortem["schedule_diagnosis"]["a10_a11_only_layer0"] is True
