from collections import defaultdict

import torch
from pytest import approx

from cascadekv.cascade import (
    FP16_K_BYTES_PER_VECTOR,
    INT4_K_BYTES_PER_VECTOR,
    INT4_ROUTER_FP16_RERANK_STORAGE_BYTES_PER_VECTOR,
    CascadeResult,
    CascadeSchedule,
    Q8K4Cascade,
    accounting_totals,
    random_cascade,
    router_fp16_rerank_totals,
)
from cascadekv.quantize import quantized_ordered_progressive_scores
from experiments.qwen_cascade import add_result, dense_baselines, select_adaptive


def prepared(keys: int = 64) -> Q8K4Cascade:
    torch.manual_seed(7)
    return Q8K4Cascade(torch.randn(128), torch.randn(keys, 128), torch.randperm(128))


def test_survivors_are_nested_and_ids_remain_score_aligned() -> None:
    cascade = prepared()
    result = cascade.run(CascadeSchedule("test", (48, 40, 32)))
    for before, after in zip(result.stage_ids, result.stage_ids[1:]):
        assert set(after.tolist()).issubset(set(before.tolist()))
    assert torch.allclose(result.stage_scores[0], cascade.direct_prefix_scores(16)[result.stage_ids[0]])


def test_progressive_prefix_and_full_accumulation_match_existing_quantized_scorer() -> None:
    cascade = prepared()
    expected, _, _ = quantized_ordered_progressive_scores(
        cascade.q, cascade.keys, cascade.order, q_bits=8, k_bits=4, group_size=16, unbiased=False
    )
    result = cascade.run(CascadeSchedule("test", (48, 40, 32)))
    for stage, dimension in enumerate((16, 32, 64, 128)):
        assert torch.allclose(result.stage_scores[stage], expected[dimension][result.stage_ids[stage]])
    assert torch.allclose(cascade.direct_prefix_scores(128), expected[128])


def test_oracle_fp32_rerank_is_diagnostic_only_and_uses_final_survivors() -> None:
    cascade = prepared()
    result = cascade.run(CascadeSchedule("all", (64, 64, 64)), oracle_fp32_rerank=True)
    dense = (cascade.keys * cascade.q).sum(-1)
    assert result.oracle_fp32_rerank_scores is not None
    assert torch.allclose(result.oracle_fp32_rerank_scores, dense[result.final_ids])
    expected = torch.topk(dense, 8).indices
    actual = result.final_ids[torch.topk(result.oracle_fp32_rerank_scores, 8).indices]
    assert set(expected.tolist()) == set(actual.tolist())
    assert torch.allclose(result.final_scores, cascade.direct_prefix_scores(128)[result.final_ids])


def test_hand_computed_byte_and_mac_accounting_counts_each_group_once() -> None:
    cascade = prepared(100)
    result = cascade.run(CascadeSchedule("hand", (75, 50, 32)))
    rows = result.accounting
    assert [(row.newly_read_k_coordinate_bytes, row.newly_read_k_scale_bytes) for row in rows] == [
        (800, 200), (600, 150), (800, 200), (1024, 256)
    ]
    assert [row.integer_macs for row in rows] == [1600, 1200, 1600, 2048]
    totals = accounting_totals(rows, 100)
    assert totals["total_k_bytes"] == 4030
    assert totals["total_quantized_macs"] == 6448
    assert totals["macs_avoided_percent"] == 49.625


def test_fp16_rerank_traffic_and_mac_accounting_are_separate() -> None:
    cascade = prepared(100)
    result = cascade.run(CascadeSchedule("hand", (75, 50, 32)))
    totals = router_fp16_rerank_totals(result.accounting, 100, result.stage_ids[2].numel())
    assert {key: value for key, value in totals.items() if key != "traffic_avoided_vs_dense_fp16_percent"} == {
        "router_int4_k_bytes": 2750,
        "rerank_fp16_k_bytes": 8192,
        "total_k_bytes_with_rerank": 10942,
        "dense_fp16_k_bytes": 25600,
        "router_integer_macs": 4400,
        "rerank_high_precision_macs": 4096,
    }
    assert totals["traffic_avoided_vs_dense_fp16_percent"] == approx(57.2578125)
    assert INT4_K_BYTES_PER_VECTOR == 80
    assert FP16_K_BYTES_PER_VECTOR == 256
    assert INT4_ROUTER_FP16_RERANK_STORAGE_BYTES_PER_VECTOR == 336


def test_final_survivor_floor_and_random_baseline_are_deterministic() -> None:
    schedule = CascadeSchedule("floor", (20, 10, 2))
    assert schedule.counts(64, final_k=32) == (32, 32, 32)
    left, right = random_cascade(64, schedule, seed=4), random_cascade(64, schedule, seed=4)
    assert all(torch.equal(a, b) for a, b in zip(left, right))


def test_calibration_schedule_selection_cannot_inspect_evaluation_tensors() -> None:
    calibration = {
        "0": {
            "cheap": {"stage_2_top8_survival": 0.96, "oracle_relative_mass_top8": 0.98, "stage_2_q8k4_top8_survival": 0.96, "total_k_bytes_with_rerank": 10},
            "safe": {"stage_2_top8_survival": 1.0, "oracle_relative_mass_top8": 1.0, "stage_2_q8k4_top8_survival": 1.0, "total_k_bytes_with_rerank": 20},
        }
    }
    selected = select_adaptive(calibration)
    assert selected["0"]["schedule"] == "cheap"
    assert selected["0"]["target_satisfied"] is True


def test_routing_selection_relaxes_only_the_fp32_survivor_target() -> None:
    calibration = {
        "0": {
            "cheap-relaxed": {"stage_2_top8_survival": 0.945, "oracle_relative_mass_top8": 0.98, "stage_2_q8k4_top8_survival": 0.96, "total_k_bytes_with_rerank": 10},
            "fails-oracle": {"stage_2_top8_survival": 0.96, "oracle_relative_mass_top8": 0.96, "stage_2_q8k4_top8_survival": 0.99, "total_k_bytes_with_rerank": 1},
        }
    }
    selected = select_adaptive(calibration)["0"]
    assert selected["schedule"] == "cheap-relaxed"
    assert selected["target_satisfied"] is False
    assert selected["preferred_0_95_target_relaxed"] is True


def test_dense_baseline_reports_its_held_out_sample_count() -> None:
    torch.manual_seed(11)
    samples = {0: [(torch.randn(128), torch.randn(40, 128))]}
    result = dense_baselines(samples, {0: torch.randperm(128)})
    assert result["dense_q8k4_vs_fp32"]["sample_count"] == 1


def test_final_top8_recall_uses_reranked_top8_not_all_top32_survivors() -> None:
    # FP32's top eight all survive to the final 32, but the deployable score
    # ranks them below eight other survivors.  Top-32 recall is perfect while
    # top-8 recall must expose the final ranking error.
    full = torch.arange(40, dtype=torch.float32)
    final_ids = torch.arange(8, 40)
    final_scores = torch.cat((torch.full((24,), 100.0), torch.zeros(8)))
    result = CascadeResult(
        stage_ids=(final_ids, final_ids, final_ids, final_ids),
        stage_scores=(final_scores, final_scores, final_scores, final_scores),
        final_scores=final_scores,
        accounting=(),
    )
    record = defaultdict(list)
    add_result(record, result, full, full)
    assert record["cascade_vs_fp32_top32_recall"] == [1.0]
    assert record["cascade_vs_fp32_top8_recall"][0] < 1.0
