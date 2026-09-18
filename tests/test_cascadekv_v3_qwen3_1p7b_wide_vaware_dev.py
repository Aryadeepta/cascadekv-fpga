import hashlib
import json
import sys
import types
from itertools import product

import pytest
import torch

from experiments import cascadekv_v3_qwen3_1p7b_wide_vaware_dev as w


def candidate(i, **overrides):
    return {"name": f"target_wide_T{i}", "cosine": 0.985, "relative_l2": 0.119, "kv": 99, **overrides}


def test_frozen_inputs_target_bindings_and_fresh_identities():
    manifest = w.require_inputs()
    assert hashlib.sha256(w.MANIFEST.read_bytes()).hexdigest() == w.MANIFEST_SHA
    assert w.MODEL == "Qwen/Qwen3-1.7B"
    assert w.MODEL_SHA == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert w.CONFIG_SHA == "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
    assert manifest["target_model"] == {"name": w.MODEL, "resolved_commit_sha": w.MODEL_SHA, "config_sha256": w.CONFIG_SHA}
    assert w.sha(w.V1) == w.V1_SHA and w.sha(w.V2) == w.V2_SHA and w.sha(w.V3) == w.V3_SHA
    assert w.sha(w.NARROW) == w.NARROW_SHA

    sources = manifest["sources"]
    assert {name: source["index"] for name, source in sources.items()} == {
        "narrative_calibration": 9, "narrative_validation": 10,
        "report_calibration": 10, "report_validation": 11,
        "qa_calibration": 9, "qa_validation": 10,
    }
    calibration = {w.identity(source) for source in sources.values() if source["role"] == "calibration"}
    validation = {w.identity(source) for source in sources.values() if source["role"] == "validation"}
    consumed = {w.identity(row) for row in manifest["complete_consumed_identity_inventory"]}
    assert {name for name, source in sources.items() if source["role"] == "calibration"} == {
        "narrative_calibration", "report_calibration", "qa_calibration"
    }
    assert {name for name, source in sources.items() if source["role"] == "validation"} == {
        "narrative_validation", "report_validation", "qa_validation"
    }
    assert len(calibration) == 3 and len(validation) == 3
    assert calibration.isdisjoint(validation)
    assert (calibration | validation).isdisjoint(consumed)


def test_action_semantics_are_exactly_frozen():
    assert w.ACTION_SEMANTICS == {
        "A0": [0.05, "hierarchy using frozen 5%-routing profile"],
        "A1": [0.075, "hierarchy using frozen 5%-routing profile"],
        "A2": [0.10, "hierarchy using frozen 5%-routing profile"],
        "A3": [0.15, "hierarchy using frozen 5%-routing profile"],
        "A4": [0.05, "flat"],
    }
    assert w.ACTIONS == ("A0", "A1", "A2", "A3", "A4")


def test_exact_targets_and_each_invalid_order_fails_closed():
    assert w.wide_targets(10, 20, 50) == {"T0": 10, "T1": 15.0, "T2": 20, "T3": 30.0, "T4": 40.0, "T5": 50}
    for baselines in ((10, 10, 20), (10, 20, 20), (20, 10, 50), (10, 50, 20)):
        with pytest.raises(RuntimeError, match="WIDE-TARGET-BASELINE-ORDER-INVALID"):
            w.wide_targets(*baselines)
    assert w.wide_targets.__code__.co_argcount == 3
    assert "NARROW" not in w.wide_targets.__code__.co_names


def test_absolute_quality_boundaries_pass_when_comparators_are_strictly_worse():
    assert w.passes(candidate(0, cosine=0.985, relative_l2=0.120, kv=99), 100, 0.121, 0.121) is True


def test_equal_flat5_bandwidth_fails():
    assert w.passes(candidate(0, kv=100), 100, 0.121, 0.121) is False


def test_equal_v2_relative_l2_fails():
    assert w.passes(candidate(0, relative_l2=0.120), 100, 0.120, 0.121) is False


def test_equal_frozen_v3_relative_l2_fails():
    assert w.passes(candidate(0, relative_l2=0.120), 100, 0.121, 0.120) is False


def test_fixed_winner_order_and_malformed_fail_closed():
    candidates = [candidate(i) for i in range(6)]
    candidates[0]["cosine"] = 0.984
    candidates[1]["relative_l2"] = 0.01
    assert w.choose_winner(list(reversed(candidates)), 100, 0.121, 0.121)["name"] == "target_wide_T1"
    assert w.choose_winner(candidates[:-1], 100, 0.121, 0.121) is None
    candidates[0] = {**candidates[1]}
    assert w.choose_winner(candidates, 100, 0.121, 0.121) is None
    unknown = [candidate(i) for i in range(6)]
    unknown[-1]["name"] = "target_wide_T6"
    assert w.choose_winner(unknown, 100, 0.121, 0.121) is None
    assert w.choose_winner([candidate(0)] + [candidate(i, relative_l2=0.0001, cosine=0.999) for i in range(1, 6)], 100, 0.121, 0.121)["name"] == "target_wide_T0"


def test_all_required_candidates_failing_has_no_winner():
    candidates = [candidate(i, cosine=0.984) for i in range(6)]
    assert {row["name"] for row in candidates} == set(w.TARGET_NAMES)
    assert w.choose_winner(candidates, 100, 0.121, 0.121) is None


def test_status_is_zero_before_the_future_driver_is_run():
    status = w.status()
    assert (status["valid_caches"], status["valid_action_shards"], status["total_caches"], status["total_action_shards"]) == (0, 0, 30, 30)
    assert status["result_exists"] is False
    source = w.Path(w.__file__).read_text()
    harness = w.Path("scripts/run_cascadekv_v3_qwen3_1p7b_wide_vaware_dev.sh").read_text()
    assert "--capture" in harness and "--evaluate" in harness and "--merge" in harness
    assert "from_pretrained" not in source


def _narrow_snapshot():
    return {name: getattr(w._n, name) for name in w._NARROW_BINDINGS}


def test_bound_narrow_restores_every_binding_after_success_and_exception(monkeypatch):
    before = _narrow_snapshot()
    with w.bound_narrow():
        assert w._n.TAG == w.TAG
    assert _narrow_snapshot() == before

    def boom(*args):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(w._n, "validate_shard", boom)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        w.validate_shard("does-not-matter", "qa_calibration", 0)
    # The monkeypatched helper itself is intentionally retained, while every
    # protocol global is restored exactly to its pre-call object/value.
    assert _narrow_snapshot() == before


def test_construct_candidates_restores_narrow_bindings_on_success_and_exception(monkeypatch):
    action = []
    base = []
    for layer, head, name in product(w.LAYERS, range(8), w.ACTIONS):
        action.append({"layer": layer, "kv_head": head, "action": name,
                       "metrics": {"relative_l2_error": .1, "cosine_similarity": .99},
                       "traffic": {"total_kv_bytes": {"A0": 10, "A1": 20, "A2": 30, "A3": 40, "A4": 50}[name]}})
    for name, value in (("frozen_cascadekv_v2", 20), ("uniform_v1_10", 30), ("flat_q8k4_5", 50)):
        base.append({"method": name, "traffic": {"total_kv_bytes": value}})
    before = _narrow_snapshot()
    w.construct_candidates(action, base)
    assert _narrow_snapshot() == before
    monkeypatch.setattr(w._n, "optimize_calibration", lambda *_: (_ for _ in ()).throw(RuntimeError("dp boom")))
    with pytest.raises(RuntimeError, match="dp boom"):
        w.construct_candidates(action, base)
    assert _narrow_snapshot() == before


def test_require_complete_is_a_machine_readable_30_by_30_barrier(tmp_path):
    with pytest.raises(RuntimeError, match="30 valid caches"):
        w.require_complete(tmp_path / "cache", tmp_path / "shards")


def _metric_row(source, role, layer, action=None, method=None, quality=.10, kv=30):
    row = {"source": source, "role": role, "layer": layer, "kv_head": 0,
           "metrics": {"relative_l2_error": quality, "cosine_similarity": .99,
                       "absolute_l2_error": .2, "relative_exact_attention_mass": .9,
                       "top8_recall": .8},
           "traffic": {key: 0. for key in w.TRAFFIC_FIELDS}}
    row["traffic"].update({"total_k_bytes": kv / 2, "selected_v_fp16_bytes": kv / 2,
                            "total_kv_bytes": kv})
    if action is not None:
        row["action"] = action
    if method is not None:
        row["method"] = method
    return row


def _write_minimal_complete_inputs(tmp_path, validation_quality=.10):
    """Rows are deliberately minimal: merge's central validators are patched
    only to isolate fitting/reporting from I/O schema validation here."""
    cache, shards = tmp_path / "cache", tmp_path / "shards"
    for source, layer in product(w.CAL + w.VAL, w.LAYERS):
        role = "calibration" if source in w.CAL else "validation"
        rows = [_metric_row(source, role, layer, action=a, quality=validation_quality if role == "validation" else .1,
                            kv={"A0": 10, "A1": 20, "A2": 30, "A3": 40, "A4": 50}[a])
                for a in w.ACTIONS for _head in range(8)]
        # Give every table lookup its own (layer, head) row without adding
        # irrelevant production-sized QKV fixtures.
        for i, row in enumerate(rows):
            row["kv_head"] = i % 8
        baselines = []
        for method in w.BASELINES:
            kv = {"frozen_cascadekv_v2": 20, "uniform_v1_10": 30, "flat_q8k4_5": 50}.get(method, 40)
            baselines.append(_metric_row(source, role, layer, method=method,
                                         quality=.2, kv=kv))
        path = w.shard_path(shards, source, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": rows, "baseline_rows": baselines}))
        cache_path = w.cache_path(cache, source, layer)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.touch()
    return cache, shards


def test_real_merge_emits_six_nonempty_calibration_summaries_and_is_validation_isolated(tmp_path, monkeypatch):
    cache, shards = _write_minimal_complete_inputs(tmp_path, .01)
    monkeypatch.setattr(w, "validate_cache", lambda *args: (True, "synthetic"))
    monkeypatch.setattr(w, "validate_shard", lambda *args: (True, "synthetic"))
    one = tmp_path / "one.json"; w.merge(cache, shards, one)
    first = json.loads(one.read_text())
    assert set(first["calibration_summaries"]) == set(w.TARGET_NAMES)
    assert all(set(x) == {"mean_cosine", "mean_relative_l2", "mean_absolute_l2", "mean_attention_mass", "mean_top8_recall", "mean_total_k_bytes", "mean_selected_v_bytes", "mean_total_kv_bytes", "p5_cosine", "p95_relative_l2", "worst_relative_l2"} for x in first["calibration_summaries"].values())
    # Replacing validation metrics must not participate in table fitting or
    # calibration reporting.
    cache2, shards2 = _write_minimal_complete_inputs(tmp_path / "other", .99)
    two = tmp_path / "two.json"; w.merge(cache2, shards2, two)
    second = json.loads(two.read_text())
    for key in ("fresh_calibration_baselines", "traffic_targets", "calibration_frozen_tables", "calibration_summaries"):
        assert first[key] == second[key]


def test_merge_partial_is_not_scientific_and_order_violation_fails_closed(tmp_path, monkeypatch):
    cache, shards = _write_minimal_complete_inputs(tmp_path)
    monkeypatch.setattr(w, "validate_cache", lambda *args: (True, "synthetic"))
    monkeypatch.setattr(w, "validate_shard", lambda path, *_: (path != w.shard_path(shards, w.CAL[0], w.LAYERS[0]), "synthetic"))
    out = tmp_path / "partial.json"; w.merge(cache, shards, out)
    partial = json.loads(out.read_text())
    assert partial["status"] == "partial" and partial["classification"] is None
    with pytest.raises(RuntimeError, match="WIDE-TARGET-BASELINE-ORDER-INVALID"):
        w.wide_targets(30, 20, 50)


def _synthetic_qkv():
    # A zero-stride view has the production shape/dtype but serializes as a
    # tiny synthetic fixture rather than allocating model-scale tensors.
    return torch.zeros(1, dtype=torch.float16).as_strided((1, 16, 4096, 128), (0, 0, 0, 0)), torch.zeros(1, dtype=torch.float16).as_strided((1, 8, 4096, 128), (0, 0, 0, 0)), torch.zeros(1, dtype=torch.float16).as_strided((1, 8, 4096, 128), (0, 0, 0, 0))


def test_real_wide_capture_uses_frozen_model_binding_skips_valid_and_cleans_failed_output(tmp_path, monkeypatch):
    calls = []
    q, k, v = _synthetic_qkv()
    class Tokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("tokenizer", args, kwargs)); return lambda *a, **kw: types.SimpleNamespace(input_ids=torch.zeros((1, 4), dtype=torch.long))
    class Model:
        layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace(scaling=1.0)) for _ in range(28)]
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("model", args, kwargs)); return Model()
        def eval(self): return self
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tokenizer, AutoModel=Model))
    monkeypatch.setattr(w._n, "resolve", lambda *_: "synthetic")
    monkeypatch.setattr(w._n, "capture_target_layer_post_rope_qkv_low_memory", lambda *_: (q, k, v))
    root = tmp_path / "capture"
    w.capture("qa_calibration", 0, root)
    path = w.cache_path(root, "qa_calibration", 0)
    assert w.validate_cache(path, "qa_calibration", 0)[0]
    assert [(x[0], x[1][0], x[2]["revision"]) for x in calls] == [("tokenizer", w.MODEL, w.MODEL_SHA), ("model", w.MODEL, w.MODEL_SHA)]
    w.capture("qa_calibration", 0, root)
    assert len(calls) == 2                         # valid cache never recaptures
    torch.save({"bad": True}, path)
    w.capture("qa_calibration", 0, root)
    assert len(calls) == 4                         # corrupt cache regenerates
    failing = tmp_path / "failing"
    monkeypatch.setattr(w._n, "capture_target_layer_post_rope_qkv_low_memory", lambda *_: (_ for _ in ()).throw(RuntimeError("capture boom")))
    with pytest.raises(RuntimeError, match="capture boom"):
        w.capture("qa_calibration", 0, failing)
    assert not w.cache_path(failing, "qa_calibration", 0).exists()


def test_real_wide_evaluate_writes_240_actions_336_baselines_and_preserves_profiles(tmp_path, monkeypatch):
    q, k, v = _synthetic_qkv(); source, layer = "qa_calibration", 14
    cache, shards = tmp_path / "cache", tmp_path / "shards"; cp = w.cache_path(cache, source, layer); cp.parent.mkdir()
    manifest_source = w.load_manifest()["sources"][source]
    torch.save({"metadata": {"manifest_sha256": w.MANIFEST_SHA, "protocol_tag": w.TAG, "protocol_commit": w.TAG_COMMIT,
                "model_name": w.MODEL, "model_sha": w.MODEL_SHA, "model_revision": w.MODEL_SHA, "target_config_sha256": w.CONFIG_SHA,
                "source_identity": manifest_source, "role": "calibration", "layer": layer, "context_length": 4096,
                "tensor_shapes": {"q": [1,16,4096,128], "k": [1,8,4096,128], "v": [1,8,4096,128]}, "attention_scaling": 1.0}, "query": q, "key": k, "value": v}, cp)
    profiles = []
    class Forest:
        def __init__(self, **kwargs): pass
        def append_atom(self, value): pass
    monkeypatch.setattr(w._n, "StreamingLiftingForest", Forest)
    monkeypatch.setattr(w._n, "load_v1", lambda: {})
    monkeypatch.setattr(w._n, "attention_reference", lambda *_: None)
    monkeypatch.setattr(w._n, "output_metrics", lambda *_: {"relative_exact_attention_mass": .9, "top8_recall": .8, "cosine_similarity": .99, "relative_l2_error": .1, "absolute_l2_error": .2})
    monkeypatch.setattr(w._n, "_evaluate_hierarchy", lambda budget, profile, *_args, **_kw: (profiles.append((budget, profile)) or {0}, {"total_k_bytes": 1., "selected_v_fp16_bytes": 1.}))
    monkeypatch.setattr(w._n, "evaluate_flat", lambda budget, *_: (profiles.append((budget, "flat")) or {0}, {"total_k_bytes": 1., "selected_v_fp16_bytes": 1.}))
    w.evaluate(source, layer, cache, shards)
    sp = w.shard_path(shards, source, layer); payload = json.loads(sp.read_text())
    assert len(payload["rows"]) == 240 and len(payload["baseline_rows"]) == 336
    assert {r["action"] for r in payload["rows"]} == set(w.ACTIONS)
    assert {r["method"] for r in payload["baseline_rows"]} == set(w.BASELINES)
    assert w.validate_shard(sp, source, layer)[0]
    assert (.05, .05) in profiles and (.075, .05) in profiles and (.10, .05) in profiles and (.15, .05) in profiles and (.05, "flat") in profiles
    assert (.10, .10) in profiles and (.05, .05) in profiles  # uniform10/uniform5
    before = len(profiles); w.evaluate(source, layer, cache, shards); assert len(profiles) == before
    sp.write_text("{}")
    w.evaluate(source, layer, cache, shards); assert len(profiles) > before
