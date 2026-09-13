#!/usr/bin/env python3
"""Characterize Q8xK4 group scales and fixed-point routing arithmetic."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascadekv.fixed_point import (
    FixedFormat,
    fixed_accumulate,
    float_to_fixed,
    integer_dot_times_scale,
)
from cascadekv.quantize import quantize_symmetric
from cascadekv.scorer import (
    attention_mass_recall,
    has_nontrivial_top_k,
    relative_attention_mass_recall,
    top_k_recall,
)
from experiments.qwen_coordinate_order import (
    DIMENSIONS,
    EVALUATION_TEXTS,
    LAYERS,
    query_positions,
    samples_for_capture,
)
from experiments.qwen_partial_dot import capture_post_rope_qk

GROUP_SIZE = 16
SCALE_FORMATS = (FixedFormat(12, 4), FixedFormat(14, 4), FixedFormat(16, 4))
# Accumulator formats retain enough integer range for observed per-group terms
# and the 128-D sum, while testing whether 20 bits is already sufficient.
ACCUMULATOR_FORMATS = (FixedFormat(20, 10), FixedFormat(22, 10), FixedFormat(24, 10), FixedFormat(24, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-query-positions", type=int, default=5)
    parser.add_argument("--min-query-position", type=int, default=128)
    parser.add_argument("--orders", type=Path, default=Path("results/coordinate_orders.json"))
    parser.add_argument("--output", type=Path, default=Path("results/fixed_point_range.json"))
    parser.add_argument("--text-index", type=int, choices=range(len(EVALUATION_TEXTS)))
    parser.add_argument("--stream-output", type=Path, help="write compact post-quantization stream and stop")
    parser.add_argument("--streams", type=Path, nargs="+", help="aggregate compact streams without loading Qwen")
    return parser.parse_args()


def format_name(format: FixedFormat) -> str:
    return f"Q{format.integer_bits}.{format.fractional_bits} ({format.total_bits}b)"


def distribution(values: list[torch.Tensor]) -> dict[str, float | int]:
    flat = torch.cat([value.detach().reshape(-1).float().cpu() for value in values])
    absolute = flat.abs()
    return {"sample_count": flat.numel(), "minimum": flat.min().item(), "maximum": flat.max().item(),
            "max_absolute": absolute.max().item(), "mean": flat.mean().item(), "standard_deviation": flat.std(unbiased=False).item(),
            **{f"p{label}_absolute": torch.quantile(absolute, percentile).item() for label, percentile in
               (("50", .5), ("90", .9), ("99", .99), ("99.9", .999), ("99.99", .9999))}}


def candidate_metrics(approximate: dict[int, list[torch.Tensor]], reference: dict[int, list[torch.Tensor]]) -> dict[str, Any]:
    result = {}
    for dimension in DIMENSIONS:
        pairs = list(zip(approximate[dimension], reference[dimension], strict=True))
        errors = torch.cat([(approx - float_ref).reshape(-1) for approx, float_ref in pairs])
        normalized = [
            ((approx - float_ref).square().mean() / (float_ref.square().mean() + 1e-12)).item()
            for approx, float_ref in pairs
        ]
        row: dict[str, float | int | None] = {
            "sample_count": errors.numel(),
            "mean_absolute_numerical_error": errors.abs().mean().item(),
            "normalized_score_error": sum(normalized) / len(normalized),
        }
        ranked = [(approx, float_ref) for approx, float_ref in pairs if has_nontrivial_top_k(approx.numel(), 8)]
        if ranked:
            row["top_8_recall"] = sum(top_k_recall(approx, float_ref, 8) for approx, float_ref in ranked) / len(ranked)
            row["relative_attention_mass_top_8"] = sum(
                relative_attention_mass_recall(approx, float_ref, 8) for approx, float_ref in ranked
            ) / len(ranked)
            row["attention_mass_top_8"] = sum(
                attention_mass_recall(approx, float_ref, 8) for approx, float_ref in ranked
            ) / len(ranked)
        else:
            row.update({"top_8_recall": None, "relative_attention_mass_top_8": None, "attention_mass_top_8": None})
        result[str(dimension)] = row
    return result


def evaluate_candidate(
    scale_format: FixedFormat, accumulator_format: FixedFormat, samples: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict[str, Any]:
    approximate: dict[int, list[torch.Tensor]] = defaultdict(list)
    reference: dict[int, list[torch.Tensor]] = defaultdict(list)
    scale_clipping, contribution_clipping, accumulator_clipping = [], [], []
    for dots, scales in samples:
        scale_fixed = float_to_fixed(scales, scale_format)
        contribution = integer_dot_times_scale(dots, scale_fixed, accumulator_format)
        accumulated = fixed_accumulate(contribution, accumulator_format)
        float_scores = (dots.float() * scales).cumsum(-1)
        scale_clipping.append(scale_fixed.clipping_rate)
        contribution_clipping.append(contribution.clipping_rate)
        accumulator_clipping.append(accumulated.clipping_rate)
        for dimension in DIMENSIONS:
            index = dimension // GROUP_SIZE - 1
            multiplier = 128 / dimension
            approximate[dimension].append(accumulated.dequantize()[:, index] * multiplier)
            reference[dimension].append(float_scores[:, index] * multiplier)
    return {"scale_product_format": {"total_bits": scale_format.total_bits, "integer_bits_including_sign": scale_format.integer_bits, "fractional_bits": scale_format.fractional_bits},
            "score_accumulator_format": {"total_bits": accumulator_format.total_bits, "integer_bits_including_sign": accumulator_format.integer_bits, "fractional_bits": accumulator_format.fractional_bits},
            "clipping_saturation_rate": {"scale_product": sum(scale_clipping) / len(scale_clipping), "contribution": sum(contribution_clipping) / len(contribution_clipping), "accumulator": sum(accumulator_clipping) / len(accumulator_clipping)},
            "metrics_against_floating_scale_q8_k4": candidate_metrics(approximate, reference)}


def quantized_groups(q: torch.Tensor, keys: torch.Tensor, order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
    """Return the compact integer-dot and scale stream needed after capture."""
    q_quantized = quantize_symmetric(q[order], 8, GROUP_SIZE)
    k_quantized = quantize_symmetric(keys[:, order], 4, GROUP_SIZE)
    q_codes, k_codes = q_quantized.values.to(torch.int32), k_quantized.values.to(torch.int32)
    dots = torch.stack(
        [(k_codes[:, i : i + GROUP_SIZE] * q_codes[i : i + GROUP_SIZE]).sum(-1, dtype=torch.int32)
         for i in range(0, 128, GROUP_SIZE)],
        -1,
    )
    return dots, q_quantized.scales.unsqueeze(0) * k_quantized.scales, q_quantized, k_quantized


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.orders.read_text())
    orders = {layer: torch.tensor(frozen[str(layer)]["energy"], dtype=torch.long) for layer in LAYERS}
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    lengths: list[int] = []
    forwards = 0
    if args.streams:
        for path in args.streams:
            stream = torch.load(path, map_location="cpu", weights_only=True)
            samples.extend(stream["samples"])
            for name, items in stream["values"].items():
                values[name].extend(items)
            lengths.extend(stream["token_lengths"])
            forwards += stream["model_forwards"]
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)
        model.eval()
        captured, handles = capture_post_rope_qk({layer: model.layers[layer].self_attn for layer in LAYERS})
        text_items = enumerate(EVALUATION_TEXTS) if args.text_index is None else [(args.text_index, EVALUATION_TEXTS[args.text_index])]
        try:
            with torch.inference_mode():
                for _, text in text_items:
                    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                    length = int(tokens["input_ids"].shape[-1])
                    positions = query_positions(length, args.num_query_positions, args.min_query_position)
                    if not positions:
                        raise RuntimeError("evaluation text tokenized too short")
                    model(**tokens, use_cache=False)
                    forwards += 1
                    lengths.append(length)
                    for layer in LAYERS:
                        for q, keys in samples_for_capture(captured, length, positions, layer):
                            dots, scales, q_quantized, k_quantized = quantized_groups(q, keys, orders[layer])
                            contribution, cumulative = dots.float() * scales, (dots.float() * scales).cumsum(-1)
                            # Keep only the compact group stream; release captured Q/K tensors after this text.
                            samples.append((dots, scales))
                            values["q_scale"].append(q_quantized.scales)
                            values["k_scale"].append(k_quantized.scales)
                            values["scale_product"].append(scales)
                            values["raw_integer_dot"].append(dots)
                            values["scaled_group_contribution"].append(contribution)
                            for dimension in DIMENSIONS:
                                values[f"cumulative_quantized_score_{dimension}D"].append(
                                    cumulative[:, dimension // GROUP_SIZE - 1]
                                )
        finally:
            for handle in handles: handle.remove()
    if args.stream_output:
        args.stream_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"samples": samples, "values": dict(values), "token_lengths": lengths, "model_forwards": forwards}, args.stream_output)
        print(f"Wrote compact stream {args.stream_output}")
        return
    candidates = [evaluate_candidate(scale, accumulator, samples) for scale in SCALE_FORMATS for accumulator in ACCUMULATOR_FORMATS if accumulator.fractional_bits >= scale.fractional_bits]
    result = {"experiment": "fixed_point_range", "model": "Qwen/Qwen3-0.6B", "layers": list(LAYERS), "group_size": GROUP_SIZE,
              "capture": "held-out evaluation texts; post-q_norm/k_norm/post-RoPE; all query heads, GQA-mapped KV heads, later causal positions; frozen contribution_energy orders",
              "token_lengths": lengths, "total_model_forwards": forwards, "sample_count_query_key_pairs": sum(item[0].shape[0] for item in samples),
              "distributions": {name: distribution(items) for name, items in values.items()}, "candidate_formats": candidates,
              "storage_implications": "Current K scales remain FP16 (2 bytes/group) for accounting. RTL can convert FP16 metadata on cache ingest to a signed fixed routing scale, or store preconverted fixed K scales. A 14-bit signed scale product is an internal product; preconverting individual K scales needs a separately characterized format and changes no accounting here."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Fixed-point characterization: {forwards} forwards, {len(samples)} query-head samples")
    for name, row in result["distributions"].items(): print(f"{name}: maxabs={row['max_absolute']:.6g} p99.9={row['p99.9_absolute']:.6g}")
    print(f"Wrote {args.output}")


if __name__ == "__main__": main()
