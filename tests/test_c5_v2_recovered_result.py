import hashlib
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

from cascadekv import stdout_artifact_recovery as recovery


ROOT = Path(__file__).parents[1]
FILES = ROOT / "artifacts/c5-v2/recovered/files"
EVIDENCE = ROOT / "artifacts/c5-v2/evidence/c5_v2_kaggle_stdout.md"
RESULT = ROOT / "results/archive/cascadekv_phi35_8k_c5_v2_recovered_result.json"
POSTMORTEM = ROOT / "results/archive/cascadekv_phi35_8k_c5_v2_recovered_postmortem.json"
EXPECTED = {
    "SHA256SUMS": "bf88cb65164ec903c2f41ea0f2ebe0ef6de38ca62d08abf31f44cd4a57c4c751",
    "development.json": "edd5a43262582055d29cac3165982e8d69912150e1d22ce7e190660e6cc9360d",
    "development_capture_manifest.json": "7a1b513127bff05bc4693623d344885762790ce86af8a36b1abbcad7bbf5ef13",
    "development_sources.json": "493162dacc2cf36c53d3363ab3adaa6fe943366735e960d05824b47e29f1f3b4",
    "holdout.json": "873955fa9a89d63f89033f2d90faf4919c63dbf1f68de707dfb2d9376d198ef8",
    "holdout_capture_manifest.json": "96976aba7e2a6deb8a8a5f7826c9828c40b7a34636622b37be3c131f060a6bb6",
    "protocol.json": "2ab29d9ef370ba2f9308763e916b0c15795fbe379e317bf95bba20ae27b412d1",
    "qualification.json": "f4703bdcd477ee91f108dde4415e14831ec20a8b156fe59876d8e3a7699bc07e",
    "runtime_manifest.json": "c6644332a676a839450814bcd2c8c22d5bfee4bb445c5e34099527b07511a818",
    "test.json": "6b42073b5984681bedf6c9d43bff1987145a398c1aa86bde3a978d2f96b8271a",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((FILES / name).read_text())


def test_stdout_recovery_is_exact_and_sha256sums_verify(tmp_path):
    restored = recovery.recover(EVIDENCE.read_text(), tmp_path / "restored")
    assert set(restored) == set(EXPECTED)
    assert {name: digest(path) for name, path in restored.items()} == EXPECTED
    assert {name: digest(FILES / name) for name in EXPECTED} == EXPECTED
    sums = dict(line.split(maxsplit=1) for line in (FILES / "SHA256SUMS").read_text().splitlines())
    assert sums == {value: name for name, value in EXPECTED.items() if name != "SHA256SUMS"}


def test_truthful_container_provenance_and_reconstructed_archive(tmp_path):
    result = json.loads(RESULT.read_text())
    container = result["container_provenance"]
    assert container["original_kaggle_tarball_available_locally"] is False
    assert container["original_kaggle_tarball_sha256_transcript_attested"] == "9c87f801e1b037572e2c631b00f959bc01ae578ed0fe0d98028c585a387de2b0"
    assert container["original_kaggle_tarball_sha256_locally_verified"] is False
    archive = ROOT / container["stdout_reconstructed_convenience_archive"]["path"]
    assert digest(archive) == "b7ba80ae227c9b557d9ef1e39cc493fb6294c08686cdc5e371ec27c532e4347b"
    assert container["stdout_reconstructed_convenience_archive"]["is_original_kaggle_result_tarball"] is False
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(EXPECTED)
        tar.extractall(tmp_path, filter="data")
    assert {name: digest(tmp_path / name) for name in EXPECTED} == EXPECTED


def test_classification_selected_target_firewall_and_frontier():
    development, holdout, prospective = load("development.json"), load("holdout.json"), load("test.json")
    assert development["classification"] == "C5-DEVELOPMENT-GO"
    assert development["development_passing_targets"] == ["T5"]
    assert development["selected_target"] == prospective["selected_target"] == holdout["selected_target"] == "T5"
    assert prospective["classification"] == "C5-PROSPECTIVE-NO-PASS"
    firewall = prospective["firewall_evidence"]
    assert firewall["optimizer_rerun"] is False and firewall["schedule_unchanged"] is True
    assert firewall["opened_development_tensor_identities"] == []
    assert firewall["schedule_sha256_before"] == firewall["schedule_sha256_after"] == development["schedule_sha256"]
    assert {x["family"]: x["dataset_index"] for x in holdout["selected_sources"]} == {"narrative": 17, "qa": 18, "report": 21}
    assert holdout["next_untouched_frontier"] == {"narrative": 18, "qa": 19, "report": 22}
    rejected = holdout["rejected_sources"]["report"]
    assert rejected == [{"dataset_index": 20, "proof": {"at_least_8192": False, "bounded_frozen_tokenizer_length": 7218, "character_count": 31271, "field_exists": True, "is_python_string": True}, "reason": "mechanically_ineligible"}]


def test_t5_metrics_margins_actions_layers_and_no_raw_source_text():
    development, prospective, holdout, sources = load("development.json"), load("test.json"), load("holdout.json"), load("development_sources.json")
    dev, test = development["development_target_metrics"]["T5"], prospective["target_test_metrics"]["T5"]
    assert dev == {"mean_cosine": 0.9850333360217822, "mean_relative_l2": 0.11976855315762129, "mean_total_kv_bytes": 485339.77222222224, "observation_count": 5760}
    assert test == {"mean_cosine": 0.9843795528635383, "mean_relative_l2": 0.1221502432927124, "mean_total_kv_bytes": 484990.2166666667, "observation_count": 1440}
    assert dev["mean_cosine"] - .985 > 0 and .12 - dev["mean_relative_l2"] > 0
    assert test["mean_cosine"] - .985 < 0 and .12 - test["mean_relative_l2"] < 0
    schedule = development["schedules"]["T5"]["table"]
    assert Counter(schedule.values()) == {"A0": 3, "A1": 6, "A2": 8, "A3": 2, "A4": 81, "A5": 4, "A6": 36, "A7": 1, "A8": 3, "A9": 16}
    assert sum(value in {"A7", "A8", "A9"} for value in schedule.values()) == 20
    values = [development["action_cell_statistics"][f"0:{head}:{schedule[f'0:{head}']}"]["relative_l2"] for head in range(32)]
    assert sum(values) / len(values) == max(sum([development["action_cell_statistics"][f"{layer}:{head}:{schedule[f'{layer}:{head}']}"]["relative_l2"] for head in range(32)]) / 32 for layer in (0, 8, 16, 24, 31))
    assert holdout["raw_text_persisted"] is False
    assert "raw_text" not in json.dumps(sources, sort_keys=True)


def test_frozen_execution_files_and_tags_are_unchanged():
    result = json.loads(RESULT.read_text())
    frozen = result["frozen_scientific_execution"]
    assert subprocess.check_output(["git", "rev-parse", "cascadekv-phi35-8k-c5-freeze-v2^{}"], cwd=ROOT, text=True).strip() == frozen["freeze_commit"]
    assert subprocess.check_output(["git", "rev-parse", "cascadekv-phi35-8k-c5-freeze-v1^{}"], cwd=ROOT, text=True).strip() == "85b8a658d3fccf237f25c4f1d434fa39229c9cff"
    assert subprocess.check_output(["git", "rev-parse", "cascadekv-phi35-8k-c4-result-recovered-v1^{}"], cwd=ROOT, text=True).strip() == "98beb485c2ac7d9e209b5f8f1dfe2fdb7e4bda75"
    runtime = load("runtime_manifest.json")
    for bound in runtime["bound_files"]:
        assert digest(ROOT / bound["path"]) == bound["sha256"]
    assert digest(EVIDENCE) == json.loads(RESULT.read_text())["stdout_evidence"]["sha256"]
