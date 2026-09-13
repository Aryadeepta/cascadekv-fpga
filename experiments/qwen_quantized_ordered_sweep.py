#!/usr/bin/env python3
"""Experiment 3: held-out quantized scoring with frozen energy orderings."""

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

from cascadekv.quantize import quantization_diagnostics, quantized_ordered_progressive_scores
from cascadekv.scorer import full_dot_scores, ordered_progressive_dot_scores
from experiments.qwen_coordinate_order import (
    DIMENSIONS,
    EVALUATION_TEXTS,
    LAYERS,
    add_metrics,
    new_record,
    query_positions,
    samples_for_capture,
    summarize,
)
from experiments.qwen_partial_dot import capture_post_rope_qk

FORMATS = {"fp32": (None, None), "q8_k4": (8, 4), "q4_k4": (4, 4)}
GROUP_SIZES = (128, 32, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-query-positions", type=int, default=5)
    parser.add_argument("--min-query-position", type=int, default=128)
    parser.add_argument("--orders", type=Path, default=Path("results/coordinate_orders.json"))
    parser.add_argument("--output", type=Path, default=Path("results/int4_ordered_sweep.json"))
    return parser.parse_args()


def aggregate(records: Any) -> dict[str, Any]:
    return {name: {str(group): {str(d): summarize(records[name][group][d]) for d in DIMENSIONS}
                   for group in records[name]} for name in records}


def storage_accounting() -> dict[str, Any]:
    result: dict[str, Any] = {"fp32_bytes": 512, "fp16_bytes": 256, "int4": {}}
    for group in GROUP_SIZES:
        scales = 128 // group
        total = 64 + 2 * scales
        stages = {}
        for dimension in DIMENSIONS:
            required_scales = math.ceil(dimension / group)
            stages[str(dimension)] = {"coordinate_bytes": dimension / 2, "scale_bytes": 2 * required_scales,
                                      "total_bytes": dimension / 2 + 2 * required_scales}
        result["int4"][str(group)] = {
            "packed_coordinate_bytes": 64, "scale_count": scales, "scale_bytes": 2 * scales,
            "total_bytes": total, "compression_vs_fp16": 256 / total, "compression_vs_fp32": 512 / total,
            "metadata_overhead_percent": 100 * (2 * scales) / total, "progressive_read_bytes": stages,
        }
    return result


def add_diagnostics(records: dict[str, list[float]], prefix: str, source: torch.Tensor, quantized: Any) -> None:
    diagnostic = quantization_diagnostics(source, quantized)
    records[f"{prefix}_clipping_rate"].append(diagnostic.clipping_rate)
    records[f"{prefix}_saturation_rate"].append(diagnostic.saturation_rate)
    records[f"{prefix}_mean_absolute_error"].append(diagnostic.mean_absolute_error)
    records[f"{prefix}_mean_squared_error"].append(diagnostic.mean_squared_error)


def summarize_diagnostics(records: dict[str, list[float]]) -> dict[str, float]:
    return {name: sum(values) / len(values) for name, values in records.items()}


def apply_deltas(metrics: dict[str, Any]) -> None:
    reference = metrics["fp32"]["none"]
    for name in ("q8_k4", "q4_k4"):
        for group in metrics[name].values():
            for dimension, row in group.items():
                baseline = reference[dimension]
                for metric, delta_name in (("relative_attention_mass_top_8", "delta_relmass_top_8"),
                                           ("relative_attention_mass_top_32", "delta_relmass_top_32"),
                                           ("top_8_recall", "delta_recall_top_8"),
                                           ("top_32_recall", "delta_recall_top_32")):
                    row[delta_name] = None if row[metric] is None else row[metric] - baseline[metric]


def validate_metric_schema(metrics: dict[str, Any]) -> None:
    """Fail loudly if a written sweep omits a requested routing metric."""
    required = {
        "pearson", "spearman", "top_8_recall", "top_32_recall", "attention_mass_top_8",
        "attention_mass_top_32", "relative_attention_mass_top_8", "relative_attention_mass_top_32",
        "normalized_score_mse",
    }
    delta_required = {"delta_relmass_top_8", "delta_relmass_top_32", "delta_recall_top_8", "delta_recall_top_32"}
    for name, groups in metrics.items():
        for group in groups.values():
            for row in group.values():
                missing = required - row.keys()
                if missing:
                    raise ValueError(f"metric schema missing {sorted(missing)}")
                if name != "fp32" and (missing := delta_required - row.keys()):
                    raise ValueError(f"quantized metric schema missing {sorted(missing)}")


def print_format(metrics: dict[str, Any], name: str) -> None:
    print(f"{name}: group dim relmass@8 delta@8 recall@8 Pearson")
    groups = ("none",) if name == "fp32" else tuple(map(str, GROUP_SIZES))
    for group in groups:
        for dimension in DIMENSIONS:
            row = metrics[name][group][str(dimension)]
            delta = row.get("delta_relmass_top_8")
            print(f"{name:6} {group:>5} {dimension:3} {row['relative_attention_mass_top_8']:.4f} "
                  f"{(delta if delta is not None else 0.0):+.4f} {row['top_8_recall']:.4f} {row['pearson']:.4f}")


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.orders.read_text())
    # Experiment 2 serialized the contribution_energy strategy under its
    # concise on-disk name, "energy"; this file is the frozen architecture.
    orders = {layer: torch.tensor(frozen[str(layer)]["energy"], dtype=torch.long) for layer in LAYERS}
    if args.max_length < 2:
        raise ValueError("max length must be positive")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)
    model.eval()
    captured, handles = capture_post_rope_qk({layer: model.layers[layer].self_attn for layer in LAYERS})
    samples: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    lengths, forwards = [], 0
    try:
        with torch.inference_mode():
            for text in EVALUATION_TEXTS:
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                length = int(tokens["input_ids"].shape[-1])
                positions = query_positions(length, args.num_query_positions, args.min_query_position)
                if not positions:
                    raise RuntimeError("evaluation text tokenized too short")
                model(**tokens, use_cache=False)
                forwards += 1
                lengths.append(length)
                for layer in LAYERS:
                    samples[layer].extend(samples_for_capture(captured, length, positions, layer))
        records: Any = defaultdict(lambda: defaultdict(lambda: defaultdict(new_record)))
        per_layer: Any = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(new_record))))
        diagnostics: Any = defaultdict(lambda: defaultdict(new_record))
        for layer, layer_samples in samples.items():
            for q, keys in layer_samples:
                attention_scale = 1.0 / math.sqrt(q.numel())
                full = full_dot_scores(q, keys) * attention_scale
                fp_scores = ordered_progressive_dot_scores(q, keys, orders[layer], DIMENSIONS)
                for dimension, score in fp_scores.items():
                    add_metrics(records["fp32"]["none"][dimension], score * attention_scale, full)
                    add_metrics(per_layer[str(layer)]["fp32"]["none"][dimension], score * attention_scale, full)
                for name, (q_bits, k_bits) in FORMATS.items():
                    if name == "fp32":
                        continue
                    for group in GROUP_SIZES:
                        scores, q_quantized, k_quantized = quantized_ordered_progressive_scores(
                            q, keys, orders[layer], q_bits=q_bits, k_bits=k_bits, group_size=group
                        )
                        # Quantizers operate after physical energy reordering, so diagnostics use that order too.
                        ordered_q, ordered_k = q[orders[layer]], keys[:, orders[layer]]
                        add_diagnostics(diagnostics[name][group], "q", ordered_q, q_quantized)
                        add_diagnostics(diagnostics[name][group], "k", ordered_k, k_quantized)
                        for dimension, score in scores.items():
                            add_metrics(records[name][group][dimension], score * attention_scale, full)
                            add_metrics(per_layer[str(layer)][name][group][dimension], score * attention_scale, full)
    finally:
        for handle in handles:
            handle.remove()
    metrics = aggregate(records)
    metrics["fp32"] = {"none": metrics["fp32"]["none"]}
    apply_deltas(metrics)
    validate_metric_schema(metrics)
    per_layer_metrics = {layer: aggregate(values) for layer, values in per_layer.items()}
    for values in per_layer_metrics.values():
        apply_deltas(values)
    best_name, best_group = max(
        ((name, str(group)) for name in ("q8_k4", "q4_k4") for group in GROUP_SIZES),
        key=lambda item: metrics[item[0]][item[1]]["16"]["relative_attention_mass_top_8"],
    )
    result = {
        "experiment": 3, "model": "Qwen/Qwen3-0.6B", "layers": list(LAYERS), "dimensions": list(DIMENSIONS),
        "formats": {name: {"q_bits": pair[0], "k_bits": pair[1]} for name, pair in FORMATS.items()},
        "scale_groups": list(GROUP_SIZES), "coordinate_order": "frozen contribution_energy loaded from results/coordinate_orders.json",
        "leakage_policy": "held-out tensors are never used to select coordinate ordering or quantizer parameters",
        "capture": "post-q_norm/k_norm and post-RoPE; all Q heads and GQA-mapped KV heads at later causal positions",
        "candidate_restriction": "top-k metrics included only where candidate keys >= 4*k", "token_lengths": lengths,
        "total_model_forwards": forwards, "metrics": metrics, "per_layer": per_layer_metrics,
        "quantization_diagnostics": {name: {str(group): summarize_diagnostics(diagnostics[name][group]) for group in GROUP_SIZES}
                                     for name in ("q8_k4", "q4_k4")},
        "storage_accounting": storage_accounting(), "best_quantized_by_relmass_top_8_at_16d": {"format": best_name, "group_size": int(best_group)},
        "notes": "K is INT4 because persistent KV-cache storage/bandwidth dominates; Q is transient and INT8 preserves routing fidelity. Partial scores are routing scores; survivors receive full-width scoring before softmax.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Experiment 3: {forwards} Qwen forwards; held-out lengths={lengths}")
    print_format(metrics, "fp32")
    print_format(metrics, "q8_k4")
    print_format(metrics, "q4_k4")
    print(f"Best quantized by aggregate held-out relmass@8 at 16-D: {best_name}, group {best_group}")
    print("Per-layer relmass@8 for that configuration (16/32/64):")
    for layer in LAYERS:
        rows = per_layer_metrics[str(layer)][best_name][best_group]
        print(f"layer {layer}: " + "/".join(f"{rows[str(d)]['relative_attention_mass_top_8']:.4f}" for d in (16, 32, 64)))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
