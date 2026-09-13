#!/usr/bin/env python3
"""Experiment 2: held-out, calibration-derived progressive Q/K orderings."""

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

from cascadekv.coordinate_order import (
    contribution_energy,
    covariance_with_full_score,
    greedy_block_ordering,
    validate_permutation,
)
from cascadekv.hadamard import random_coordinate_indices, signed_hadamard
from cascadekv.scorer import (
    attention_mass_recall,
    full_dot_scores,
    gqa_kv_head_for_query,
    has_nontrivial_top_k,
    normalized_score_mse,
    ordered_progressive_dot_scores,
    pearson_correlation,
    relative_attention_mass_recall,
    spearman_correlation,
    top_k_recall,
)
from experiments.qwen_partial_dot import capture_post_rope_qk, query_positions

DIMENSIONS = (16, 32, 64, 128)
LAYERS = (0, 7, 14, 21, 27)
SEEDS = (0, 1, 2, 3, 4)
LEARNED = ("energy_ordered", "covariance_ordered", "greedy_ordered")
BASELINES = ("raw_prefix", "random_coordinates", "rotated_prefix")
METHODS = LEARNED + BASELINES

# These two explicitly disjoint, deterministic corpora are the only text used
# by Experiment 2.  They are intentionally local; no external dataset is read.
CALIBRATION_TEXTS = (
    """CALIBRATION: A survey crew mapped a tidal inlet before sunrise. They checked benchmarks, marked each
    sounding, compared notes with the previous chart, and recorded uncertainty where weeds concealed the
    channel. At noon the lead engineer reviewed the route with the boat operator and chose a second pass.
    The crew then documented equipment settings, weather, and the reason for every correction.""" * 5,
    """CALIBRATION: In a conservatory workshop, students repaired a small wooden instrument. They measured
    the bridge, inspected the grain, tuned one string at a time, and listened for sympathetic vibrations.
    Their instructor asked them to separate observation from explanation in the repair log. After testing,
    they returned each tool to its marked place and wrote which adjustment changed the tone.""" * 5,
    """CALIBRATION: A municipal planning meeting considered a new pedestrian crossing. Residents described
    school traffic, delivery schedules, winter visibility, and the timing of bus arrivals. Analysts placed
    counts on a map, tested alternative signal phases, and stated which evidence was still missing. The
    final memo preserved both the preferred design and the conditions that could change it.""" * 5,
)
EVALUATION_TEXTS = (
    """EVALUATION: A mountain observatory prepared for a week of clear nights. Technicians aligned mirrors,
    checked cooling pumps, catalogued reference stars, and rehearsed the handoff between shifts. Each image
    received a timestamp and a note about haze near the horizon. In the morning the team compared exposures
    and isolated the frames affected by a passing aircraft.""" * 5,
    """EVALUATION: The harbor archive digitized a box of ship manifests. Volunteers transcribed names, dates,
    cargo descriptions, and marginal marks while a curator resolved difficult handwriting against ledgers.
    They retained scans beside corrected fields so later readers could audit decisions. At day's end the
    database report listed uncertain entries rather than silently filling them with guesses.""" * 5,
    """EVALUATION: A greenhouse trial compared irrigation schedules across rows of young tomatoes. Sensors
    logged moisture, temperature, light, and nutrient flow, while gardeners noted leaf curl and flowering.
    The supervisor randomized the row assignments, inspected clogged emitters, and delayed conclusions until
    the second harvest. The notebook distinguished measured outcomes from practical observations.""" * 5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-query-positions", type=int, default=5)
    parser.add_argument("--min-query-position", type=int, default=128)
    parser.add_argument("--orders-output", type=Path, default=Path("results/coordinate_orders.json"))
    parser.add_argument("--output", type=Path, default=Path("results/coordinate_order_sweep.json"))
    return parser.parse_args()


def new_record() -> dict[str, list[float]]:
    return defaultdict(list)


def add_metrics(record: dict[str, list[float]], approximate: torch.Tensor, full: torch.Tensor) -> None:
    record["pearson"].append(pearson_correlation(approximate, full))
    record["spearman"].append(spearman_correlation(approximate, full))
    record["normalized_score_mse"].append(normalized_score_mse(approximate, full))
    for top_k in (8, 32):
        if has_nontrivial_top_k(full.shape[-1], top_k):
            record[f"top_{top_k}_recall"].append(top_k_recall(approximate, full, top_k))
            record[f"attention_mass_top_{top_k}"].append(attention_mass_recall(approximate, full, top_k))
            record[f"relative_attention_mass_top_{top_k}"].append(
                relative_attention_mass_recall(approximate, full, top_k)
            )


def summarize(record: dict[str, list[float]]) -> dict[str, float | int | None]:
    def mean(name: str) -> float | None:
        values = record.get(name, [])
        return sum(values) / len(values) if values else None

    result: dict[str, float | int | None] = {"sample_count": len(record.get("pearson", []))}
    for name in (
        "pearson", "spearman", "normalized_score_mse", "top_8_recall", "attention_mass_top_8",
        "relative_attention_mass_top_8", "top_32_recall", "attention_mass_top_32",
        "relative_attention_mass_top_32",
    ):
        result[name] = mean(name)
    result["top_8_sample_count"] = len(record.get("top_8_recall", []))
    result["top_32_sample_count"] = len(record.get("top_32_recall", []))
    return result


def samples_for_capture(
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]], length: int, positions: list[int], layer: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    query, key = (item[0] for item in captured[layer])
    samples = []
    for position in positions:
        for query_head in range(query.shape[0]):
            key_head = gqa_kv_head_for_query(query_head, query.shape[0], key.shape[0])
            samples.append((query[query_head, position], key[key_head, :position]))
    return samples


def calibration_contributions(samples: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Build paired q*k rows from calibration samples only (never evaluation)."""
    return torch.cat([q.unsqueeze(0) * keys for q, keys in samples], dim=0)


def aggregate(records: Any) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result = {}
    for method in METHODS:
        result[method] = {str(d): summarize(records[method][d]) for d in DIMENSIONS}
    return result


def evaluate(
    samples_by_layer: dict[int, list[tuple[torch.Tensor, torch.Tensor]]], orders: dict[int, dict[str, torch.Tensor]]
) -> tuple[dict[str, dict[str, dict[str, float | int | None]]], dict[str, Any]]:
    records: Any = defaultdict(lambda: defaultdict(new_record))
    per_layer: Any = defaultdict(lambda: defaultdict(lambda: defaultdict(new_record)))
    seed_records: Any = defaultdict(lambda: defaultdict(lambda: defaultdict(new_record)))
    for layer, samples in samples_by_layer.items():
        for q, keys in samples:
            scale = 1.0 / math.sqrt(q.numel())
            full = full_dot_scores(q, keys) * scale
            learned_scores = {
                name: ordered_progressive_dot_scores(q, keys, order, DIMENSIONS)
                for name, order in orders[layer].items()
            }
            raw = ordered_progressive_dot_scores(q, keys, torch.arange(q.numel()), DIMENSIONS)
            for dimension in DIMENSIONS:
                for name, score_set in learned_scores.items():
                    approx = score_set[dimension] * scale
                    add_metrics(records[name][dimension], approx, full)
                    add_metrics(per_layer[str(layer)][name][dimension], approx, full)
                add_metrics(records["raw_prefix"][dimension], raw[dimension] * scale, full)
                add_metrics(per_layer[str(layer)]["raw_prefix"][dimension], raw[dimension] * scale, full)
                for seed in SEEDS:
                    random_indices = random_coordinate_indices(q.numel(), dimension, seed=seed)
                    random_score = (q.numel() / dimension) * (q[random_indices] * keys[:, random_indices]).sum(-1)
                    rotated = ordered_progressive_dot_scores(
                        signed_hadamard(q, seed=seed), signed_hadamard(keys, seed=seed),
                        torch.arange(q.numel()), DIMENSIONS,
                    )[dimension]
                    for name, score in (("random_coordinates", random_score), ("rotated_prefix", rotated)):
                        approx = score * scale
                        add_metrics(records[name][dimension], approx, full)
                        add_metrics(per_layer[str(layer)][name][dimension], approx, full)
                        add_metrics(seed_records[name][str(seed)][dimension], approx, full)
    per_seed = {
        method: {seed: {str(d): summarize(seed_records[method][seed][d]) for d in DIMENSIONS} for seed in map(str, SEEDS)}
        for method in ("random_coordinates", "rotated_prefix")
    }
    return aggregate(records), {str(layer): aggregate(per_layer[str(layer)]) for layer in LAYERS} | {"per_seed": per_seed}


def overlap(orders: dict[int, dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for method in LEARNED:
        values: dict[str, float] = {}
        for width in (16, 32, 64):
            pairs = [
                len(set(orders[left][method][:width].tolist()) & set(orders[right][method][:width].tolist())) / width
                for index, left in enumerate(LAYERS) for right in LAYERS[index + 1 :]
            ]
            values[f"top_{width}_mean_pairwise_overlap"] = sum(pairs) / len(pairs)
        result[method] = values
    return result


def print_table(metrics: dict[str, dict[str, dict[str, float | int | None]]]) -> None:
    print("Held-out aggregate (random/rotated rows average five seeds; learned orders are deterministic)")
    print("method                  dim  Pearson  recall@8  relmass@8     NMSE")
    for dimension in DIMENSIONS:
        for method in METHODS:
            row = metrics[method][str(dimension)]
            print(f"{method:23} {dimension:3d}  {row['pearson']:7.4f}  {row['top_8_recall']:8.4f}  "
                  f"{row['relative_attention_mass_top_8']:9.4f}  {row['normalized_score_mse']:7.4f}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # Keep decoder weights in fp16 to fit the full CPU sweep in modest RAM;
    # captured post-RoPE Q/K tensors are immediately promoted to fp32 by the
    # capture hook, so all calibration and metric arithmetic remains fp32.
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)
    model.eval()
    captured, handles = capture_post_rope_qk({layer: model.layers[layer].self_attn for layer in LAYERS})
    datasets: dict[str, dict[int, list[tuple[torch.Tensor, torch.Tensor]]]] = {"calibration": defaultdict(list), "evaluation": defaultdict(list)}
    lengths: dict[str, list[int]] = {"calibration": [], "evaluation": []}
    forwards = 0
    try:
        with torch.inference_mode():
            for split, texts in (("calibration", CALIBRATION_TEXTS), ("evaluation", EVALUATION_TEXTS)):
                for text in texts:
                    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                    length = int(tokens["input_ids"].shape[-1])
                    positions = query_positions(length, args.num_query_positions, args.min_query_position)
                    if not positions:
                        raise RuntimeError(f"{split} text tokenized too short: {length}")
                    model(**tokens, use_cache=False)
                    forwards += 1
                    lengths[split].append(length)
                    for layer in LAYERS:
                        datasets[split][layer].extend(samples_for_capture(captured, length, positions, layer))
        orders: dict[int, dict[str, torch.Tensor]] = {}
        calibration_metrics: dict[str, Any] = {}
        for layer in LAYERS:
            contributions = calibration_contributions(datasets["calibration"][layer])
            orders[layer] = {
                "energy_ordered": contribution_energy(contributions),
                "covariance_ordered": covariance_with_full_score(contributions),
                "greedy_ordered": greedy_block_ordering(contributions),
            }
            for order in orders[layer].values():
                validate_permutation(order, contributions.shape[1])
        calibration_metrics, calibration_details = evaluate(datasets["calibration"], orders)
        evaluation_metrics, evaluation_details = evaluate(datasets["evaluation"], orders)
    finally:
        for handle in handles:
            handle.remove()
    orders_json = {str(layer): {name.replace("_ordered", ""): order.tolist() for name, order in methods.items()} for layer, methods in orders.items()}
    args.orders_output.parent.mkdir(parents=True, exist_ok=True)
    args.orders_output.write_text(json.dumps(orders_json, indent=2) + "\n")
    result = {
        "model": "Qwen/Qwen3-0.6B", "layers": list(LAYERS), "dimensions": list(DIMENSIONS), "seeds": list(SEEDS),
        "split": {"calibration_text_count": len(CALIBRATION_TEXTS), "evaluation_text_count": len(EVALUATION_TEXTS), "token_lengths": lengths,
                  "leakage_policy": "coordinate orders use calibration Q/K contributions only; held-out evaluation tensors are scored only after orders are fixed"},
        "capture": "post-q_norm/k_norm and post-RoPE; all query heads, GQA-mapped KV heads, later causal positions",
        "candidate_restriction": "top-k metrics included only where preceding candidate keys >= 4*k",
        "ordering_statistics": {"contribution_energy": "mean((q_j*k_j)^2)", "covariance_with_full_score": "absolute population Pearson correlation of c_j=q_j*k_j with full score sum_j c_j", "greedy_block_ordering": "residual-greedy individual selection, emitted in contiguous 16-coordinate groups"},
        "total_model_forwards": forwards, "calibration": calibration_metrics, "evaluation": evaluation_metrics,
        "calibration_per_layer": calibration_details, "evaluation_per_layer": evaluation_details,
        "layer_order_similarity": overlap(orders), "coordinate_orders_file": str(args.orders_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Experiment 2: {forwards} Qwen forwards; calibration lengths={lengths['calibration']}; evaluation lengths={lengths['evaluation']}")
    print_table(evaluation_metrics)
    best = max(LEARNED, key=lambda name: evaluation_metrics[name]["16"]["relative_attention_mass_top_8"] or float("-inf"))
    print(f"Best calibrated method by held-out relmass@8 at 16-D: {best}")
    print("Per-layer held-out relmass@8 for best calibrated method:")
    for layer in LAYERS:
        print(f"layer {layer}: " + ", ".join(f"{d}={evaluation_details[str(layer)][best][str(d)]['relative_attention_mass_top_8']:.4f}" for d in DIMENSIONS))
    print(f"Wrote {args.orders_output} and {args.output}")


if __name__ == "__main__":
    main()
