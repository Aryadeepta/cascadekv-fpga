"""Durable, offline Proof-B matrices for the frozen C7 executor.

These tests use only synthetic bytes and mocked capture boundaries.  They never
load a model/tokenizer, inspect a frontier, or access a dataset.
"""
import builtins
import copy
import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascadekv import phi35_8k_c7 as c7
from cascadekv.stdout_artifact_recovery import BEGIN, recover

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = c7.RAW_TEXT_SENTINEL


def _runner():
    spec = importlib.util.spec_from_file_location("c7_proof_b_runner", ROOT / "scripts/kaggle_phi35_8k_c7.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _surfaces(root, stdout, tar_path, recovered):
    """Return every terminal evidence byte surface, including tar members."""
    out = [stdout.encode()]
    out.extend(p.read_bytes() for p in root.rglob("*") if p.is_file())
    with tarfile.open(tar_path) as archive:
        out.extend(archive.extractfile(n).read() for n in archive.getnames())
    out.extend(p.read_bytes() for p in recovered.values())
    return out


def _install_synthetic_lifecycle(runner, monkeypatch, go, seen):
    monkeypatch.setattr(runner, "_audit_checkout", lambda: None)
    monkeypatch.setattr(runner.c7, "preflight", lambda **_: None)
    monkeypatch.setattr(runner, "_qualification", lambda root: (root / "qualification.json").write_bytes(b"q") or root / "qualification.json")
    monkeypatch.setattr(runner.c7, "build_development_manifest", lambda *, output: output.write_text("{}"))
    # The sentinel is present at the raw boundary only; the runner must never
    # serialize this synthetic source object.
    monkeypatch.setattr(runner.c7, "validate_development_manifest", lambda *_: [{"raw_input": SENTINEL}])
    def capture_sources(*, kind, sources, output, **_):
        assert any(x.get("raw_input") == SENTINEL for x in sources)
        seen.append("development_capture" if kind == "development" else "holdout_capture")
        output.mkdir(); (output / "final_capture_manifest.json").write_text("capture")
    monkeypatch.setattr(runner.c7, "capture_sources", capture_sources)
    result = {"classification": "C7-DEVELOPMENT-GO" if go else "C7-DEVELOPMENT-NO-GO", "schedule_sha256": "s" * 64,
              "development_passing_targets": ["T0"] if go else [], "selected_target": "T0" if go else None}
    monkeypatch.setattr(runner.c7, "develop", lambda *, output, **_: (output.write_text(json.dumps(result)), result)[1])
    monkeypatch.setattr(runner.c7, "final_audit", lambda _: seen.append("final_audit") or "ok")
    if go:
        monkeypatch.setattr(runner.c7, "assert_holdout_may_start", lambda *_: "T0")
        def select_holdout(*, output, **_):
            output.write_text("{}")
            return {"selected_sources": [{"raw_input": SENTINEL}]}
        monkeypatch.setattr(runner.c7, "select_holdout", select_holdout)
        monkeypatch.setattr(runner.c7, "prospective_test", lambda *, output, **_: output.write_text("{}"))


@pytest.mark.parametrize("go,expected", [(False, 7), (True, 10)])
def test_raw_text_terminal_lifecycle_tar_stdout_recovery_and_sums(tmp_path, monkeypatch, capsys, go, expected):
    runner = _runner(); seen = []; _install_synthetic_lifecycle(runner, monkeypatch, go, seen)
    root = tmp_path / "root"; results = tmp_path / "results"
    real_bundle = runner.bundle
    monkeypatch.setattr(runner, "bundle", lambda *args: seen.append("bundle") or real_bundle(*args))
    assert runner.run(SimpleNamespace(root=root, result_dir=results)) == 0
    stdout = capsys.readouterr().out
    tar_path = results / "cascadekv-c7-result.tar.gz"
    recovered = recover(stdout[stdout.index(BEGIN):], tmp_path / "recovered")
    with tarfile.open(tar_path) as archive:
        members = {n: archive.extractfile(n).read() for n in archive.getnames()}
    assert len(members) == expected == len(recovered)
    assert set(members) == set(recovered)
    assert all(recovered[name].read_bytes() == data for name, data in members.items())
    sums = {name: digest for digest, name in (line.split("  ", 1) for line in members["SHA256SUMS"].decode().splitlines())}
    assert {name: hashlib.sha256(data).hexdigest() for name, data in members.items() if name != "SHA256SUMS"} == sums
    assert all(SENTINEL.encode() not in data for data in _surfaces(root, stdout, tar_path, recovered))
    assert seen.index("final_audit") < seen.index("bundle")
    assert stdout.index("===== C7 PHASE: STDOUT RECOVERY =====") < stdout.index("===== C7 PHASE: " + ("HOLDOUT" if go else "DEVELOPMENT") + " CLEANUP =====")
    assert not (root / ("holdout_capture" if go else "development_capture")).exists()


@pytest.mark.parametrize("go,fail_at", [(go, point) for go in (False, True) for point in ("final_audit", "bundle", "stdout")])
def test_failure_order_preserves_active_capture_until_terminal_step(tmp_path, monkeypatch, capsys, go, fail_at):
    runner = _runner(); seen = []; _install_synthetic_lifecycle(runner, monkeypatch, go, seen)
    root = tmp_path / "root"
    if fail_at == "final_audit":
        monkeypatch.setattr(runner.c7, "final_audit", lambda _: (_ for _ in ()).throw(c7.C7Error("audit")))
    elif fail_at == "bundle":
        monkeypatch.setattr(runner, "bundle", lambda *_: (_ for _ in ()).throw(c7.C7Error("bundle")))
    else:
        monkeypatch.setattr(runner, "bundle", lambda *_: "frame")
        monkeypatch.setattr("builtins.print", lambda *a, **k: (_ for _ in ()).throw(c7.C7Error("stdout")) if a and a[0] == "frame" else None)
    with pytest.raises(c7.C7Error): runner.run(SimpleNamespace(root=root, result_dir=tmp_path / "results"))
    assert (root / ("holdout_capture" if go else "development_capture")).exists()


CREATE_ONLY_CASES = (
    "nonempty_root", "protocol_copy", "runtime_copy", "qualification", "development_sources", "development_capture", "partial_capture",
    "development_capture_manifest", "development_result", "holdout", "holdout_capture", "holdout_capture_manifest", "test", "sums",
    "result_tar", "root_symlink", "protocol_symlink", "runtime_symlink", "qualification_symlink", "capture_symlink", "tar_symlink",
)
assert len(CREATE_ONLY_CASES) == 21


@pytest.mark.parametrize("state", ("absent", "empty"))
def test_create_only_accepts_two_valid_root_states(tmp_path, state):
    runner = _runner(); root = tmp_path / "root"
    if state == "empty": root.mkdir()
    runner._clean_root(root)
    assert root.is_dir() and not list(root.iterdir())


@pytest.mark.parametrize("state", CREATE_ONLY_CASES)
def test_create_only_stale_matrix_rejects_all_21_explicit_states(tmp_path, state):
    runner = _runner(); root = tmp_path / "root"; results = tmp_path / "results"; root.mkdir()
    if state == "nonempty_root": (root / "x").write_text("x"); target = root
    elif state == "result_tar" or state == "tar_symlink":
        results.mkdir(); target = results / "cascadekv-c7-result.tar.gz"
        if state == "tar_symlink": target.symlink_to(results / "elsewhere")
        else: target.write_bytes(b"x")
        for n in ("protocol.json",): (root / n).write_text("x")
        with pytest.raises(c7.C7Error): runner.bundle(root, results)
        return
    else:
        name = {"protocol_copy":"protocol.json", "runtime_copy":"runtime_manifest.json", "qualification":"qualification.json", "development_sources":"development_sources.json", "development_capture":"development_capture", "partial_capture":"development_capture/partial.bin", "development_capture_manifest":"development_capture_manifest.json", "development_result":"development.json", "holdout":"holdout.json", "holdout_capture":"holdout_capture", "holdout_capture_manifest":"holdout_capture_manifest.json", "test":"test.json", "sums":"SHA256SUMS", "root_symlink":"x", "protocol_symlink":"protocol.json", "runtime_symlink":"runtime_manifest.json", "qualification_symlink":"qualification.json", "capture_symlink":"development_capture"}[state]
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
        if "symlink" in state: target.symlink_to(root / "elsewhere")
        elif state.endswith("capture"): target.mkdir()
        else: target.write_bytes(b"x")
    # A nonempty root, including the Proof-B partial-capture defect, cannot be reused.
    with pytest.raises(c7.C7Error): runner._clean_root(root)


@pytest.mark.parametrize("execution,tag,head,ok", [(False, None, "H", True), (True, "H", "H", True), (True, None, "H", False), (True, "X", "H", False), (True, "H", "X", False)])
def test_freeze_tag_preflight_matrix(tmp_path, monkeypatch, execution, tag, head, ok):
    def git(ref):
        if ref == c7.C6_PARENT["result_tag"]: return c7.C6_PARENT["result_commit"]
        if ref == c7.C6_PARENT["freeze_tag"]: return c7.C6_PARENT["freeze_commit"]
        if ref == c7.C7_TAG:
            if tag is None: raise c7.C7Error("missing")
            return tag
        return head
    monkeypatch.setattr(c7, "_git", git); monkeypatch.setattr(c7, "validate_protocol", lambda: {})
    monkeypatch.setattr(c7, "verify_runtime_closure", lambda: ())
    if ok: assert c7.preflight(execution=execution)["local_only"] is (not execution)
    else:
        with pytest.raises(c7.C7Error): c7.preflight(execution=execution)


C6_ADVERSE_CASES = ("result_tag_missing", "result_tag_wrong", "freeze_tag_missing", "freeze_tag_wrong", "result_json_missing", "result_json_mutated", "postmortem_json_missing", "postmortem_json_mutated", "postmortem_md_missing", "postmortem_md_mutated", "tar_missing", "tar_mutated", "stdout_missing", "stdout_mutated")
assert len(C6_ADVERSE_CASES) == 14


@pytest.mark.parametrize("case", C6_ADVERSE_CASES)
def test_c6_parent_archive_adverse_matrix(case, monkeypatch):
    def git(ref):
        if case == "result_tag_missing" and ref == c7.C6_PARENT["result_tag"]: raise c7.C7Error("missing")
        if case == "freeze_tag_missing" and ref == c7.C6_PARENT["freeze_tag"]: raise c7.C7Error("missing")
        if ref == c7.C6_PARENT["result_tag"]: return "bad" if case == "result_tag_wrong" else c7.C6_PARENT["result_commit"]
        if ref == c7.C6_PARENT["freeze_tag"]: return "bad" if case == "freeze_tag_wrong" else c7.C6_PARENT["freeze_commit"]
        return "head"
    monkeypatch.setattr(c7, "_git", git); monkeypatch.setattr(c7, "validate_protocol", lambda: {}); monkeypatch.setattr(c7, "verify_runtime_closure", lambda: ())
    original_file = Path.is_file
    needle = {"result_json":"cascadekv_phi35_8k_c6_v1_result.json", "postmortem_json":"cascadekv_phi35_8k_c6_v1_postmortem.json", "postmortem_md":"cascadekv_phi35_8k_c6_v1_postmortem.md", "tar":"cascadekv-c6-result.tar.gz", "stdout":"c6_v1_kaggle_stdout.md"}.get(case.rsplit("_", 1)[0])
    if case.endswith("missing"): monkeypatch.setattr(Path, "is_file", lambda p: False if p.name == needle else original_file(p))
    elif "tag" not in case:
        original_sha = c7.sha256_path; monkeypatch.setattr(c7, "sha256_path", lambda p: "0" * 64 if Path(p).name == needle else original_sha(p))
    with pytest.raises(c7.C7Error): c7.preflight()


def _at(payload, path):
    node = payload
    for key in path[:-1]: node = node[key]
    return node, path[-1]


def _replace(path, value):
    def mutate(payload):
        node, key = _at(payload, path); node[key] = value
    return mutate


def _reorder_mapping(path, first, second):
    def mutate(payload):
        node, key = _at(payload, path)
        values = node[key]
        items = list(values.items())
        first_index = next(i for i, (name, _) in enumerate(items) if name == first)
        second_index = next(i for i, (name, _) in enumerate(items) if name == second)
        items[first_index], items[second_index] = items[second_index], items[first_index]
        values.clear()
        values.update(items)
    return mutate


def _swap(path, first, second):
    def mutate(payload):
        values = _at(payload, path)[0][_at(payload, path)[1]]
        values[first], values[second] = values[second], values[first]
    return mutate


def _remove(path, value):
    def mutate(payload):
        values = _at(payload, path)[0][_at(payload, path)[1]]
        values.remove(value)
    return mutate


def _append(path, value):
    def mutate(payload): _at(payload, path)[0][_at(payload, path)[1]].append(value)
    return mutate


# Each row is an intentional C7 contract binding, rather than an incidental
# traversal of the first leaves in serialized JSON.
PROTOCOL_MUTATIONS = (
    # Parent (11)
    ("parent_result_tag", _replace(("c6_result_parent", "result_tag"), "cascadekv-phi35-8k-c6-result-v2")),
    ("parent_result_commit", _replace(("c6_result_parent", "result_commit"), "0" * 40)),
    ("parent_freeze_tag", _replace(("c6_result_parent", "freeze_tag"), "cascadekv-phi35-8k-c6-freeze-v2")),
    ("parent_freeze_commit", _replace(("c6_result_parent", "freeze_commit"), "1" * 40)),
    ("parent_c6_protocol_sha", _replace(("c6_result_parent", "protocol_sha256"), "2" * 64)),
    ("parent_c6_runtime_sha", _replace(("c6_result_parent", "runtime_sha256"), "3" * 64)),
    ("parent_original_tar_sha", _replace(("c6_result_parent", "original_kaggle_tarball_sha256"), "4" * 64)),
    ("parent_stdout_sha", _replace(("c6_result_parent", "repository_stdout_sha256"), "5" * 64)),
    ("parent_result_archive_sha", _replace(("c6_result_parent", "result_archive_sha256"), "6" * 64)),
    ("parent_postmortem_json_sha", _replace(("c6_result_parent", "postmortem_archive_sha256"), "7" * 64)),
    ("parent_postmortem_markdown_sha", _replace(("c6_result_parent", "postmortem_markdown_sha256"), "8" * 64)),
    # Development (6)
    ("development_narrative_identities", _replace(("development_sources", "identities", "narrative"), [13, 14, 15, 16, 17, 19])),
    ("development_report_identities", _replace(("development_sources", "identities", "report"), [16, 17, 18, 19, 20, 22])),
    ("development_qa_identities", _replace(("development_sources", "identities", "qa"), [13, 15, 16, 17, 19, 18])),
    ("development_report20_excluded", _replace(("development_sources", "report20_excluded"), False)),
    ("development_source_text_stored", _replace(("development_sources", "source_text_stored"), True)),
    ("development_archived_provenance_only", _replace(("development_sources", "archived_consumed_provenance_only"), False)),
    # Geometry (9)
    ("geometry_context_length", _replace(("geometry", "context_length"), 8191)),
    ("geometry_layers", _replace(("geometry", "layers"), [0, 8, 16, 24, 30])),
    ("geometry_query_positions", _replace(("geometry", "query_positions"), [4095, 6144, 8191])),
    ("geometry_heads", _replace(("geometry", "heads"), 31)),
    ("geometry_head_dim", _replace(("geometry", "head_dim"), 95)),
    ("geometry_schedule_cells", _replace(("geometry", "schedule_cells"), 159)),
    ("geometry_observations_per_source", _replace(("geometry", "observations_per_source_per_method"), 479)),
    ("geometry_development_observations", _replace(("geometry", "development_observations_per_method"), 8639)),
    ("geometry_family_observations", _replace(("geometry", "family_observations_per_cell_per_method"), 17)),
    # Actions (13)
    ("a0_kind", _replace(("methods", "actions", "A0", "kind"), "flat_q8k4")),
    ("a0_candidate_fraction", _replace(("methods", "actions", "A0", "candidate_fraction"), .06)),
    ("a0_routing_profile", _replace(("methods", "actions", "A0", "routing_profile"), "phi35-8k-depth-transfer-qwen10-v1")),
    ("a9_kind", _replace(("methods", "actions", "A9", "kind"), "flat_q8k4")),
    ("a9_candidate_fraction", _replace(("methods", "actions", "A9", "candidate_fraction"), .21)),
    ("a9_routing_profile", _replace(("methods", "actions", "A9", "routing_profile"), "phi35-8k-depth-transfer-qwen5-v1")),
    ("a10_kind", _replace(("methods", "actions", "A10", "kind"), "flat_q8k4")),
    ("a10_candidate_fraction", _replace(("methods", "actions", "A10", "candidate_fraction"), .30)),
    ("a10_routing_profile", _replace(("methods", "actions", "A10", "routing_profile"), "phi35-8k-depth-transfer-qwen10-v1")),
    ("a11_kind", _replace(("methods", "actions", "A11", "kind"), "flat_q8k4")),
    ("a11_candidate_fraction", _replace(("methods", "actions", "A11", "candidate_fraction"), .30)),
    ("a11_routing_profile", _replace(("methods", "actions", "A11", "routing_profile"), "phi35-8k-depth-transfer-qwen5-v1")),
    ("action_ordering", _reorder_mapping(("methods", "actions"), "A10", "A11")),
    # Action validity (5)
    ("validity_remove_a10_layer0", _remove(("action_validity", "layer_0"), "A10")),
    ("validity_remove_a11_layer0", _remove(("action_validity", "layer_0"), "A11")),
    ("validity_add_a10_nonzero", _append(("action_validity", "nonzero_sampled_layers"), "A10")),
    ("validity_add_a11_nonzero", _append(("action_validity", "nonzero_sampled_layers"), "A11")),
    ("validity_enforcement", _replace(("action_validity", "enforcement"), "DP enumeration domain only")),
    # Optimizer (5)
    ("optimizer_version", _replace(("optimizer", "version"), "c7-robust-max-family-integer-microbyte-dp-v2")),
    ("optimizer_scope", _replace(("optimizer", "scope"), "development_and_holdout")),
    ("optimizer_objective", _replace(("optimizer", "objective"), "mean_final_attention_output_relative_l2")),
    ("optimizer_family_aggregation", _replace(("optimizer", "family_aggregation"), "mean of narrative/report/qa means")),
    ("optimizer_tie_breaking", _replace(("optimizer", "tie_breaking"), "traffic ascending")),
    # Targets (5)
    ("target_t0", _replace(("targets", "fractions", "T0"), .116)),
    ("target_middle", _replace(("targets", "fractions", "T2"), .146)),
    ("target_t4", _replace(("targets", "fractions", "T4"), .181)),
    ("target_t5_rule", _replace(("targets", "fractions", "T5"), "highest_traffic_below_development_flat5")),
    ("target_ordering", _swap(("targets", "order"), 0, 1)),
    # Quality gates (4), lifecycle (4), frontier (4), raw text (1)
    ("gate_cosine", _replace(("quality_gates", "mean_cosine_gte"), .984)),
    ("gate_rell2", _replace(("quality_gates", "mean_relative_l2_lte"), .121)),
    ("gate_uniform10", _replace(("quality_gates", "winner_relative_l2_lt"), "uniform5")),
    ("gate_flat5", _replace(("quality_gates", "winner_modeled_kv_traffic_lt"), "flat10")),
    ("lifecycle_no_go", _replace(("lifecycle", "development_no_go"), "C7-DEVELOPMENT-STOP")),
    ("lifecycle_go", _replace(("lifecycle", "development_go"), "C7-DEVELOPMENT-PASS")),
    ("lifecycle_freeze_before_prospective", _replace(("lifecycle", "freeze_before_prospective"), False)),
    ("lifecycle_selected_target_locks", _replace(("lifecycle", "selected_target_locks_prospective_verdict"), False)),
    ("frontier_narrative", _replace(("prospective_frontier", "start_indices", "narrative"), 20)),
    ("frontier_qa", _replace(("prospective_frontier", "start_indices", "qa"), 21)),
    ("frontier_report", _replace(("prospective_frontier", "start_indices", "report"), 24)),
    ("frontier_untouched", _replace(("prospective_frontier", "untouched_during_preparation"), False)),
    ("raw_text_persisted", _replace(("raw_text_persisted",), True)),
    # Schema shape (2)
    ("schema_missing_top_level", lambda payload: payload.pop("raw_text_persisted")),
    ("schema_unexpected_top_level", lambda payload: payload.__setitem__("unapproved_c7_field", True)),
)
assert len(PROTOCOL_MUTATIONS) == 69


@pytest.mark.parametrize("case_name,mutate", PROTOCOL_MUTATIONS, ids=[x[0] for x in PROTOCOL_MUTATIONS])
def test_all_69_semantic_protocol_mutations_are_independently_rejected(case_name, mutate):
    payload = json.loads(c7.PROTOCOL.read_text())
    original = copy.deepcopy(payload)
    mutate(payload)
    if case_name == "action_ordering":
        actions = payload["methods"]["actions"]
        original_actions = original["methods"]["actions"]
        expected_order = list(original_actions)
        a10_index = expected_order.index("A10")
        a11_index = expected_order.index("A11")
        expected_order[a10_index], expected_order[a11_index] = expected_order[a11_index], expected_order[a10_index]
        assert list(actions) == expected_order
        assert list(actions) != list(c7.ACTIONS)
        assert list(actions) != list(original_actions)
        assert set(actions) == set(original_actions)
        assert all(actions[action] == original_actions[action] for action in actions)
    with pytest.raises(c7.C7Error): c7.validate_protocol(payload)


@pytest.mark.parametrize("kind", ("development", "holdout"))
@pytest.mark.parametrize("state", ("empty", "partial_artifact", "partial_provenance", "final_manifest", "symlink"))
def test_capture_sources_refuses_existing_output_before_heavy_boundary(tmp_path, monkeypatch, kind, state):
    output = tmp_path / "capture"
    if state == "symlink": output.symlink_to(tmp_path / "other")
    else:
        output.mkdir()
        name = {"empty": None, "partial_artifact": "artifacts/partial.safetensors", "partial_provenance": "artifacts/partial.provenance.json", "final_manifest": "final_capture_manifest.json"}[state]
        if name: (output / name).parent.mkdir(parents=True, exist_ok=True); (output / name).write_text("partial")
    crossed = []
    original_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("cascadekv.phi35_kaggle_c2") or name.startswith("cascadekv.phi35_8k_source_selection"):
            crossed.append(name); raise AssertionError("heavy capture boundary crossed")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(c7.C7Error):
        c7.capture_sources(kind=kind, sources=[], output=output, qualification=tmp_path / "qualification.json")
    assert crossed == []


@pytest.mark.parametrize("state", ("existing", "symlink"))
def test_develop_from_rows_refuses_existing_output_before_evaluation(tmp_path, monkeypatch, state):
    output = tmp_path / "development.json"
    if state == "symlink": output.symlink_to(tmp_path / "other.json")
    else: output.write_text("existing")
    monkeypatch.setattr(c7, "validate_development_manifest", lambda *_: (_ for _ in ()).throw(AssertionError("evaluation boundary crossed")))
    with pytest.raises(c7.C7Error):
        c7.develop_from_rows(rows={}, development_source_manifest=tmp_path / "sources.json", qualification=tmp_path / "qualification.json", capture_manifest=tmp_path / "capture.json", output=output)


def _isolated_runtime(tmp_path, monkeypatch):
    manifest = json.loads(c7.RUNTIME.read_text()); root = tmp_path / "closure"; root.mkdir()
    protocol = root / manifest["protocol"]["path"]; protocol.parent.mkdir(parents=True); protocol.write_bytes(c7.PROTOCOL.read_bytes())
    for row in manifest["bound_files"]:
        dst = root / row["path"]; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(c7.ROOT / row["path"], dst)
    manifest["protocol"]["sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    for row in manifest["bound_files"]: row["sha256"] = hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
    runtime = root / "runtime.json"; runtime.write_text(json.dumps(manifest))
    monkeypatch.setattr(c7, "ROOT", root); monkeypatch.setattr(c7, "PROTOCOL", protocol); monkeypatch.setattr(c7, "RUNTIME", runtime)
    return manifest, root


def test_real_runtime_manifest_has_exactly_23_bound_files():
    assert len(json.loads(c7.RUNTIME.read_text())["bound_files"]) == 23


@pytest.mark.parametrize("index", range(23))
def test_runtime_each_of_23_bound_bytes_rejected(tmp_path, monkeypatch, index):
    manifest, root = _isolated_runtime(tmp_path, monkeypatch)
    path = root / manifest["bound_files"][index]["path"]; path.write_bytes(path.read_bytes() + b"proof-b")
    with pytest.raises(c7.C7Error): c7.verify_runtime_closure()


RUNTIME_STRUCTURAL_MUTATIONS = ("duplicate", "absolute", "traversal", "missing", "symlink", "malformed_sha", "wrong_sha", "protocol_path", "protocol_sha", "overlap", "bound_not_list", "entry_shape")
assert len(RUNTIME_STRUCTURAL_MUTATIONS) == 12


@pytest.mark.parametrize("case", RUNTIME_STRUCTURAL_MUTATIONS)
def test_runtime_structural_matrix_12_rejected(tmp_path, monkeypatch, case):
    manifest, root = _isolated_runtime(tmp_path, monkeypatch); rows = manifest["bound_files"]
    if case == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif case == "absolute": rows[0]["path"] = "/tmp/absolute"
    elif case == "traversal": rows[0]["path"] = "../escape"
    elif case == "missing": (root / rows[0]["path"]).unlink()
    elif case == "symlink":
        p = root / rows[0]["path"]; target = root / "target"; target.write_bytes(p.read_bytes()); p.unlink(); p.symlink_to(target)
    elif case == "malformed_sha": rows[0]["sha256"] = "z" * 64
    elif case == "wrong_sha": rows[0]["sha256"] = "0" * 64
    elif case == "protocol_path": manifest["protocol"]["path"] = "wrong.json"
    elif case == "protocol_sha": manifest["protocol"]["sha256"] = "0" * 64
    elif case == "overlap": rows[0]["path"] = manifest["protocol"]["path"]
    elif case == "bound_not_list": manifest["bound_files"] = {}
    else: rows[0] = {"path": rows[0]["path"]}
    c7.RUNTIME.write_text(json.dumps(manifest))
    with pytest.raises(c7.C7Error): c7.verify_runtime_closure()
