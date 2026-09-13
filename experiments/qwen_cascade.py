#!/usr/bin/env python3
"""Experiment 4: held-out progressive Q8xK4 candidate-cascade routing."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascadekv.cascade import (
    FP16_K_BYTES_PER_VECTOR,
    INT4_K_BYTES_PER_VECTOR,
    INT4_ROUTER_FP16_RERANK_STORAGE_BYTES_PER_VECTOR,
    CascadeSchedule,
    Q8K4Cascade,
    accounting_totals,
    fraction_schedule,
    random_cascade,
    router_fp16_rerank_totals,
)
from experiments.qwen_coordinate_order import (
    CALIBRATION_TEXTS,
    EVALUATION_TEXTS,
    LAYERS,
    query_positions,
    samples_for_capture,
)
from experiments.qwen_partial_dot import capture_post_rope_qk

FINAL_K = 32
RANDOM_SEEDS = (0, 1, 2, 3, 4)
FRACTIONS = {
    "conservative": (0.75, 0.50, 0.25),
    "balanced": (0.50, 0.25, 0.125),
    "aggressive": (0.375, 0.1875, 0.0625),
}
EXPLICIT = {
    "192-128-64": (192, 128, 64), "128-64-32": (128, 64, 32),
    "128-48-16": (128, 48, 16), "96-32-16": (96, 32, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-query-positions", type=int, default=5)
    parser.add_argument("--min-query-position", type=int, default=128,
                        help="causal query position; keep >=128 so candidates greatly exceed final_k=32")
    parser.add_argument("--orders", type=Path, default=Path("results/coordinate_orders.json"))
    parser.add_argument("--output", type=Path, default=Path("results/cascade_sweep.json"))
    parser.add_argument("--schedules-output", type=Path, default=Path("results/cascade_schedules.json"))
    return parser.parse_args()


def schedules_for(candidate_count: int) -> list[CascadeSchedule]:
    return [fraction_schedule(name, fractions, candidate_count) for name, fractions in FRACTIONS.items()] + [
        CascadeSchedule(name, values) for name, values in EXPLICIT.items()
    ]


def _recall(ids: torch.Tensor, truth: torch.Tensor) -> float:
    return (truth[:, None] == ids[None, :]).any(dim=1).float().mean().item()


def _mass(ids: torch.Tensor, full: torch.Tensor, truth: torch.Tensor) -> float:
    probabilities = torch.softmax(full, dim=0)
    return (probabilities[ids].sum() / probabilities[truth].sum()).clamp(0, 1).item()


def empty_record() -> dict[str, list[float]]:
    return defaultdict(list)


def add_result(record: dict[str, list[float]], result: Any, full: torch.Tensor, dense_q8k4: torch.Tensor) -> None:
    """Record deployable quality separately from the high-precision oracle."""
    truth8, truth32 = torch.topk(full, 8).indices, torch.topk(full, 32).indices
    q8k4_truth8, q8k4_truth32 = torch.topk(dense_q8k4, 8).indices, torch.topk(dense_q8k4, 32).indices
    for stage, ids in enumerate(result.stage_ids):
        record[f"stage_{stage}_top8_survival"].append(_recall(ids, truth8))
        record[f"stage_{stage}_top32_survival"].append(_recall(ids, truth32))
        record[f"stage_{stage}_q8k4_top8_survival"].append(_recall(ids, q8k4_truth8))
    reranked_top8_ids = result.final_ids[torch.topk(result.final_scores, 8).indices]
    reranked_top32_ids = result.final_ids[torch.topk(result.final_scores, 32).indices]
    record["cascade_vs_fp32_top8_recall"].append(_recall(reranked_top8_ids, truth8))
    record["cascade_vs_fp32_top32_recall"].append(_recall(reranked_top32_ids, truth32))
    record["cascade_vs_fp32_relative_attention_mass_top8"].append(_mass(reranked_top8_ids, full, truth8))
    record["cascade_vs_fp32_relative_attention_mass_top32"].append(_mass(reranked_top32_ids, full, truth32))
    record["deployable_final_top8_recall"].append(_recall(reranked_top8_ids, truth8))
    record["deployable_relative_mass_top8"].append(_mass(reranked_top8_ids, full, truth8))
    record["cascade_vs_q8k4_top8_recall"].append(_recall(reranked_top8_ids, q8k4_truth8))
    record["cascade_vs_q8k4_top32_recall"].append(_recall(reranked_top32_ids, q8k4_truth32))
    if result.oracle_fp32_rerank_scores is not None:
        oracle_top8 = result.final_ids[torch.topk(result.oracle_fp32_rerank_scores, 8).indices]
        record["oracle_final_top8_recall"].append(_recall(oracle_top8, truth8))
        record["oracle_relative_mass_top8"].append(_mass(oracle_top8, full, truth8))
        record["oracle_fp32_rerank_macs"].append(float(result.final_ids.numel() * 128))
    stage2 = result.stage_ids[2]
    lost = 8 - int((truth8[:, None] == stage2[None, :]).any(dim=1).sum())
    record["catastrophic_any_top8_lost"].append(float(lost > 0))
    record["catastrophic_more_than_25pct_top8_lost"].append(float(lost > 2))
    totals = accounting_totals(result.accounting, full.numel())
    for key, value in totals.items():
        record[key].append(float(value))
    # Architecture B stops INT4 routing after the 64-D stage and performs a
    # high-precision dot only for those survivors.
    rerank_totals = router_fp16_rerank_totals(result.accounting, full.numel(), result.stage_ids[2].numel())
    for key, value in rerank_totals.items():
        record[key].append(float(value))


def summarize(record: dict[str, list[float]]) -> dict[str, float | int]:
    # Dense baselines do not have cascade-only deployable metrics.  Every
    # record nevertheless appends one value per held-out sample.
    sample_count = len(next(iter(record.values()), ()))
    return {"sample_count": sample_count} | {
        name: sum(values) / len(values) for name, values in record.items() if values
    }


def run_random(
    prepared: Q8K4Cascade, schedule: CascadeSchedule, full: torch.Tensor, seed: int, accounting: Any
) -> Any:
    ids = random_cascade(full.numel(), schedule, seed=seed, final_k=FINAL_K)
    # Random routing still uses the deployable Q8xK4 score on its final survivors.
    return type("RandomResult", (), {
        "stage_ids": ids, "final_ids": ids[-1], "final_scores": prepared.direct_prefix_scores(128)[ids[-1]],
        "oracle_fp32_rerank_scores": (prepared.keys[ids[-1]] * prepared.q).sum(-1), "accounting": accounting,
    })()


def evaluate(
    samples: dict[int, list[tuple[torch.Tensor, torch.Tensor]]], orders: dict[int, torch.Tensor], *, random: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate: Any = defaultdict(empty_record)
    per_layer: Any = defaultdict(lambda: defaultdict(empty_record))
    for layer, layer_samples in samples.items():
        for sample_index, (q, keys) in enumerate(layer_samples):
            full = (keys * q).sum(-1) / math.sqrt(q.numel())
            prepared = Q8K4Cascade(q, keys, orders[layer])
            dense_q8k4 = prepared.direct_prefix_scores(128)
            for schedule in schedules_for(keys.shape[0]):
                if random:
                    accounting = prepared.accounting_for_schedule(schedule, final_k=FINAL_K)
                    # Average each metric over five fixed seeds, preserving matched traffic.
                    for seed in RANDOM_SEEDS:
                        result = run_random(prepared, schedule, full, seed + sample_index * 1009 + layer * 100000, accounting)
                        add_result(aggregate[schedule.name], result, full, dense_q8k4)
                        add_result(per_layer[str(layer)][schedule.name], result, full, dense_q8k4)
                else:
                    result = prepared.run(schedule, final_k=FINAL_K, oracle_fp32_rerank=True)
                    add_result(aggregate[schedule.name], result, full, dense_q8k4)
                    add_result(per_layer[str(layer)][schedule.name], result, full, dense_q8k4)
    return ({name: summarize(record) for name, record in aggregate.items()}, {
        layer: {name: summarize(record) for name, record in rows.items()} for layer, rows in per_layer.items()
    })


def dense_baselines(
    samples: dict[int, list[tuple[torch.Tensor, torch.Tensor]]], orders: dict[int, torch.Tensor]
) -> dict[str, Any]:
    """Measure the frozen full-width quantizer without any candidate pruning."""
    record = empty_record()
    for layer, layer_samples in samples.items():
        for q, keys in layer_samples:
            full = (keys * q).sum(-1) / math.sqrt(q.numel())
            q8k4 = Q8K4Cascade(q, keys, orders[layer]).direct_prefix_scores(128)
            truth8, truth32 = torch.topk(full, 8).indices, torch.topk(full, 32).indices
            approx8, approx32 = torch.topk(q8k4, 8).indices, torch.topk(q8k4, 32).indices
            record["top8_recall"].append(_recall(approx8, truth8))
            record["top32_recall"].append(_recall(approx32, truth32))
            record["relative_attention_mass_top8"].append(_mass(approx8, full, truth8))
            record["relative_attention_mass_top32"].append(_mass(approx32, full, truth32))
    return {
        "fp32": {"top8_recall": 1.0, "top32_recall": 1.0, "relative_attention_mass_top8": 1.0,
                 "relative_attention_mass_top32": 1.0},
        "dense_q8k4_vs_fp32": summarize(record),
        "accounting": "dense Q8xK4: N*80 K bytes and N*128 integer MACs; dense FP16 K comparison: N*256 K bytes",
    }


def select_routing_sketch_schedule(rows: dict[str, Any]) -> dict[str, Any]:
    """Select Architecture B exclusively from calibration measurements."""
    def constraints(row: dict[str, Any], survival_floor: float) -> bool:
        return (
            row["stage_2_top8_survival"] >= survival_floor
            and row["oracle_relative_mass_top8"] >= 0.97
            and row["stage_2_q8k4_top8_survival"] >= 0.95
        )

    strict = [(name, row) for name, row in rows.items() if constraints(row, 0.95)]
    relaxed = [(name, row) for name, row in rows.items() if constraints(row, 0.94)]
    if strict:
        candidates, target, relaxed_target = strict, "fp32_top8_survivor_recall_after_64d_routing >= 0.95", False
    elif relaxed:
        candidates, target, relaxed_target = relaxed, "RELAXED: fp32_top8_survivor_recall_after_64d_routing >= 0.94", True
    else:
        # Keep output usable for a new dataset while making the failed target
        # unambiguous.  This path is not a valid deployment recommendation.
        candidates, target, relaxed_target = list(rows.items()), "UNMET: no schedule met the 0.94 routing target", True
    name, row = min(candidates, key=lambda item: item[1]["total_k_bytes_with_rerank"])
    return {
        "schedule": name,
        "target_satisfied": bool(strict),
        "preferred_0_95_target_relaxed": relaxed_target,
        "selection_target": target,
        "calibration": row,
    }


def select_adaptive(calibration_per_layer: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {layer: select_routing_sketch_schedule(rows) for layer, rows in calibration_per_layer.items()}


def combine_selected(per_layer: dict[str, Any], selected: dict[str, dict[str, Any]]) -> dict[str, float]:
    records = [per_layer[layer][choice["schedule"]] for layer, choice in selected.items()]
    keys = records[0].keys()
    return {key: sum(float(row[key]) for row in records) / len(records) for key in keys if key != "sample_count"} | {
        "sample_count": sum(int(row["sample_count"]) for row in records)
    }


def capture_splits(args: argparse.Namespace) -> tuple[dict[str, dict[int, list[tuple[torch.Tensor, torch.Tensor]]]], dict[str, list[int]], int]:
    if args.min_query_position < 128:
        raise ValueError("--min-query-position must be at least 128 for a meaningful pruning sweep")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)
    model.eval()
    captured, handles = capture_post_rope_qk({layer: model.layers[layer].self_attn for layer in LAYERS})
    datasets: dict[str, Any] = {"calibration": defaultdict(list), "evaluation": defaultdict(list)}
    lengths = {"calibration": [], "evaluation": []}
    forwards = 0
    try:
        with torch.inference_mode():
            for split, texts in (("calibration", CALIBRATION_TEXTS), ("evaluation", EVALUATION_TEXTS)):
                for text in texts:
                    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                    length = int(tokens["input_ids"].shape[-1])
                    positions = query_positions(length, args.num_query_positions, args.min_query_position)
                    if not positions:
                        raise RuntimeError(f"{split} text tokenized too short")
                    model(**tokens, use_cache=False)
                    forwards += 1
                    lengths[split].append(length)
                    for layer in LAYERS:
                        datasets[split][layer].extend(samples_for_capture(captured, length, positions, layer))
    finally:
        for handle in handles:
            handle.remove()
    return datasets, lengths, forwards


def print_summary(uniform_name: str, uniform: dict[str, Any], adaptive: dict[str, Any], random: dict[str, Any], selected: dict[str, Any], evaluation: dict[str, Any], baselines: dict[str, Any]) -> None:
    dense = baselines["dense_q8k4_vs_fp32"]
    print("Dense Q8K4 quantization ceiling:")
    print(f"    top8 recall vs FP32 = {dense['top8_recall']:.6f}")
    print(f"    relmass8 vs FP32   = {dense['relative_attention_mass_top8']:.6f}")
    print("Architecture A: int4_only (all schedules; 80 bytes/K vector)")
    print("schedule        K_bytes  integer_MACs  K_reduction  MAC_reduction  fp32_top8  fp32_mass8  q8k4_top8")
    for name, row in evaluation.items():
        print(f"{name:15} {row['total_k_bytes']:.0f}    {row['total_quantized_macs']:.0f}"
              f"       {row['int4_bytes_avoided_percent']:.2f}%       {row['macs_avoided_percent']:.2f}%"
              f"       {row['cascade_vs_fp32_top8_recall']:.4f}"
              f"     {row['cascade_vs_fp32_relative_attention_mass_top8']:.4f}     {row['cascade_vs_q8k4_top8_recall']:.4f}")
    print("Architecture B: int4_router_fp16_rerank (336 bytes/K stored: 80 INT4 + 256 FP16)")
    print("schedule        router_INT4_B  rerank_FP16_B  total_K_B  saved_vs_dense_FP16  router_int_MACs  rerank_HP_MACs")
    for name, row in evaluation.items():
        print(f"{name:15} {row['router_int4_k_bytes']:.0f}           {row['rerank_fp16_k_bytes']:.0f}"
              f"          {row['total_k_bytes_with_rerank']:.0f}      {row['traffic_avoided_vs_dense_fp16_percent']:.2f}%"
              f"             {row['router_integer_macs']:.0f}            {row['rerank_high_precision_macs']:.0f}")
    print("Selected routing-sketch schedules per layer (calibration-only):")
    for layer in map(str, LAYERS):
        row, name = selected[layer]["evaluation"], selected[layer]["schedule"]
        print(f"    layer {layer}: {name}; FP32 survivor recall={row['stage_2_top8_survival']:.4f}; "
              f"oracle relmass8={row['oracle_relative_mass_top8']:.4f}; router={row['router_int4_k_bytes']:.0f} B; "
              f"rerank={row['rerank_fp16_k_bytes']:.0f} B; total={row['total_k_bytes_with_rerank']:.0f} B; "
              f"{selected[layer]['selection_target']}")
    print("INT4-only best:")
    print(f"    schedule = {uniform_name}; top8 recall = {uniform['cascade_vs_fp32_top8_recall']:.6f}; relmass8 = {uniform['cascade_vs_fp32_relative_attention_mass_top8']:.6f}; bytes saved vs dense Q8K4 = {uniform['int4_bytes_avoided_percent']:.2f}%")
    print("INT4-router + FP16 rerank:")
    print(f"    selected schedule = {uniform_name}; FP32 top8 survivor recall = {uniform['stage_2_top8_survival']:.6f}; oracle relmass8 = {uniform['oracle_relative_mass_top8']:.6f}; router bytes = {uniform['router_int4_k_bytes']:.0f}; FP16 rerank bytes = {uniform['rerank_fp16_k_bytes']:.0f}; total K bytes = {uniform['total_k_bytes_with_rerank']:.0f}; traffic saved vs dense FP16 = {uniform['traffic_avoided_vs_dense_fp16_percent']:.2f}%")
    print(f"Random matched: dense-Q8K4 top8 survival = {random['stage_2_q8k4_top8_survival']:.6f}")
    print("Quality decomposition:")
    print(f"    quantization loss (top8 recall) = {1.0 - dense['top8_recall']:.6f}")
    print(f"    routing loss (top8 recall) = {dense['top8_recall'] - uniform['cascade_vs_fp32_top8_recall']:.6f}")
    print(f"    quantization loss (relmass8) = {1.0 - dense['relative_attention_mass_top8']:.6f}")
    print(f"    routing loss (relmass8) = {dense['relative_attention_mass_top8'] - uniform['cascade_vs_fp32_relative_attention_mass_top8']:.6f}")


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.orders.read_text())
    orders = {layer: torch.tensor(frozen[str(layer)]["energy"], dtype=torch.long) for layer in LAYERS}
    datasets, lengths, forwards = capture_splits(args)
    calibration, calibration_per_layer = evaluate(datasets["calibration"], orders)
    selected = select_adaptive(calibration_per_layer)  # Deliberately before evaluation is examined.
    evaluation, evaluation_per_layer = evaluate(datasets["evaluation"], orders)
    evaluation_dense_baselines = dense_baselines(datasets["evaluation"], orders)
    random_evaluation, _ = evaluate(datasets["evaluation"], orders, random=True)
    uniform_selection = select_routing_sketch_schedule(calibration)
    uniform_name = uniform_selection["schedule"]
    uniform = evaluation[uniform_name]
    adaptive = combine_selected(evaluation_per_layer, selected)
    for layer, choice in selected.items():
        choice["evaluation"] = evaluation_per_layer[layer][choice["schedule"]]
    random_matched = random_evaluation[uniform_name]
    schedule_output = {
        "experiment": 4, "selection_split": "calibration only",
        "objective": "Architecture B: minimum total traffic (router INT4 K bytes + final-survivor FP16 K bytes) subject to FP32 top-8 survivor recall after 64-D routing >= 0.95, oracle FP32-rerank relative mass@8 >= 0.97, and 64-D survivor recall versus dense Q8xK4 top-8 >= 0.95; relax only the FP32 survivor target to >= 0.94 when needed",
        "final_k_floor": FINAL_K, "uniform_selected": uniform_selection, "selected": selected,
        "leakage_policy": "selection completes from calibration tensors before held-out evaluation tensors are scored",
    }
    result = {
        "experiment": 4, "model": "Qwen/Qwen3-0.6B", "layers": list(LAYERS), "quantization": "Q8 x K4, group_size=16",
        "coordinate_order": "frozen contribution_energy", "final_score": "Architecture A: accumulated 128-D Q8xK4 score over final survivors; Architecture B: FP16 authoritative-K score over 64-D survivors",
        "architecture_a_int4_only": {
            "authoritative_k": "Q8xK4 group16", "storage_bytes_per_k_vector": INT4_K_BYTES_PER_VECTOR,
            "routing": "full progressive cascade through 128-D", "accounting_fields": ["total_k_bytes", "total_quantized_macs", "int4_bytes_avoided_percent"],
        },
        "architecture_b_int4_router_fp16_rerank": {
            "authoritative_k": "FP16", "routing_sketch": "ordered Q8xK4 group16 through 64-D", "storage_bytes_per_k_vector": INT4_ROUTER_FP16_RERANK_STORAGE_BYTES_PER_VECTOR,
            "fp16_k_bytes_per_vector": FP16_K_BYTES_PER_VECTOR, "int4_routing_sketch_bytes_per_vector": INT4_K_BYTES_PER_VECTOR,
            "rerank": "full high-precision 128-D dot over final 64-D routing survivors",
            "accounting_fields": ["router_int4_k_bytes", "rerank_fp16_k_bytes", "total_k_bytes_with_rerank", "traffic_avoided_vs_dense_fp16_percent", "router_integer_macs", "rerank_high_precision_macs"],
        },
        "candidate_schedules": {**FRACTIONS, **EXPLICIT}, "random_seeds": list(RANDOM_SEEDS), "final_k_floor": FINAL_K,
        "capture": "post-q_norm/k_norm and post-RoPE; all Q heads and GQA-mapped KV heads", "token_lengths": lengths,
        "total_model_forwards": forwards, "calibration": calibration, "calibration_per_layer": calibration_per_layer,
        "dense_baselines": evaluation_dense_baselines,
        "evaluation": evaluation, "evaluation_per_layer": evaluation_per_layer, "random_matched_evaluation": random_evaluation,
        "uniform_best": {"selection": uniform_selection, "held_out": uniform}, "layer_adaptive": {"held_out": adaptive, "per_layer": selected},
        "accounting_notes": "Architecture A reports progressive INT4 K traffic and integer MACs through 128-D. Architecture B reports INT4 router traffic/MACs only through 64-D, then separate FP16 K traffic and high-precision MACs for final survivors; these MAC types are never combined.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.schedules_output.write_text(json.dumps(schedule_output, indent=2) + "\n")
    print_summary(uniform_name, uniform, adaptive, random_matched, selected, evaluation, evaluation_dense_baselines)
    print(f"Wrote {args.output} and {args.schedules_output}")


if __name__ == "__main__":
    main()
