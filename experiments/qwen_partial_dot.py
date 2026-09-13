#!/usr/bin/env python3
"""Rigorous progressive partial-dot sweep on real Qwen3 attention Q/K."""

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
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascadekv.hadamard import random_coordinate_indices, signed_hadamard
from cascadekv.scorer import (
    attention_mass_recall,
    full_dot_scores,
    gqa_kv_head_for_query,
    has_nontrivial_top_k,
    pearson_correlation,
    progressive_dot_scores,
    relative_attention_mass_recall,
    spearman_correlation,
    top_k_recall,
)

DIMENSIONS = (16, 32, 64, 128)
METHODS = ("rotated_prefix", "raw_prefix", "random_coordinates")
DEFAULT_LAYERS = (0, 7, 14, 21, 27)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
_PARAGRAPHS = (
    (
        "The field station stood beside a marsh where reeds bent in the afternoon wind. Each observer "
        "wrote the time, temperature, cloud cover, and the calls heard from the water. At dusk they "
        "compared notes, corrected uncertain identifications, and planned the next route."
    ),
    (
        "A careful engineering review begins by naming the claim being tested and the baseline that could "
        "refute it. Measurements are recorded with their assumptions, sample counts, and failure cases so "
        "later decisions remain explainable. The team then repeats the work under different conditions."
    ),
    (
        "In the workshop, an old clock was disassembled across a clean cloth. Gears were arranged in order, "
        "springs were measured, and every screw was placed in a labeled tray. Its maker listened for a "
        "steady rhythm before closing the case."
    ),
    (
        "The small town library hosted an evening class on local history. Residents brought photographs, "
        "maps, and letters, then discussed which details could be confirmed by more than one source. The "
        "archivist preserved both strong evidence and unanswered questions."
    ),
)


def long_inputs() -> list[str]:
    """Three distinct deterministic texts, each comfortably beyond 256 typical tokens."""
    return [" ".join(_PARAGRAPHS[index:] + _PARAGRAPHS[:index]) * 3 for index in range(3)]


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated integer list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=parse_csv_ints, default=DEFAULT_LAYERS)
    parser.add_argument("--layer", type=int, help="Compatibility alias for a single selected layer")
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-query-positions", type=int, default=5)
    parser.add_argument("--min-query-position", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/partial_dot_sweep.json"))
    return parser.parse_args()


def query_positions(length: int, count: int, minimum: int) -> list[int]:
    if length <= minimum:
        return []
    count = min(count, length - minimum)
    return sorted({minimum + index * (length - 1 - minimum) // max(count - 1, 1) for index in range(count)})


def capture_post_rope_qk(
    attentions: dict[int, torch.nn.Module],
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], list[Any]]:
    """Capture post-q_norm/k_norm, post-RoPE Q/K for all requested layers."""
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def make_hook(layer: int) -> Any:
        def hook(
            module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            hidden_states = kwargs["hidden_states"]
            cos, sin = kwargs["position_embeddings"]
            shape = (*hidden_states.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden_states).view(shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden_states).view(shape)).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            captured[layer] = (query.detach().float().cpu(), key.detach().float().cpu())

        return hook

    for layer, attention in attentions.items():
        handles.append(attention.register_forward_pre_hook(make_hook(layer), with_kwargs=True))
    return captured, handles


def capture_layer0_post_rope_qk_low_memory(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly capture layer-0 Q/K without evaluating attention or later layers.

    This follows the installed Qwen3 ``Qwen3Model``/``Qwen3Attention`` data
    path: embeddings -> layer-0 input RMSNorm -> Q/K projections and Q/K
    norms -> the model's own rotary embedding and ``apply_rotary_pos_emb``.
    It deliberately does not construct a causal mask, values, attention scores,
    or decoder layers 1 onward.
    """
    hidden_states = model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    layer = model.layers[0]
    hidden_states = layer.input_layernorm(hidden_states)
    attention = layer.self_attn
    shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
    query = attention.q_norm(attention.q_proj(hidden_states).view(shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden_states).view(shape)).transpose(1, 2)
    cos, sin = model.rotary_emb(hidden_states, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    return query.detach().float().cpu(), key.detach().float().cpu()


def new_record() -> dict[str, list[float]]:
    return defaultdict(list)


def add_metrics(record: dict[str, list[float]], approximate: torch.Tensor, full: torch.Tensor) -> None:
    """Add one sample; top-k metrics are deliberately omitted when trivial."""
    record["pearson"].append(pearson_correlation(approximate, full))
    record["spearman"].append(spearman_correlation(approximate, full))
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

    return {
        "sample_count": len(record.get("pearson", [])), "pearson": mean("pearson"),
        "spearman": mean("spearman"), "top_8_recall": mean("top_8_recall"),
        "top_8_sample_count": len(record.get("top_8_recall", [])),
        "relative_attention_mass_top_8": mean("relative_attention_mass_top_8"),
        "attention_mass_top_8": mean("attention_mass_top_8"),
        "top_32_recall": mean("top_32_recall"),
        "top_32_sample_count": len(record.get("top_32_recall", [])),
        "relative_attention_mass_top_32": mean("relative_attention_mass_top_32"),
        "attention_mass_top_32": mean("attention_mass_top_32"),
    }


def add_sample_methods(
    records: dict[str, dict[int, dict[int, dict[str, list[float]]]]],
    seed_records: dict[str, dict[int, dict[int, dict[int, dict[str, list[float]]]]]],
    layer: int,
    seed: int,
    q: torch.Tensor,
    keys: torch.Tensor,
    include_raw: bool,
) -> None:
    head_dim, scale = q.numel(), 1.0 / math.sqrt(q.numel())
    full = full_dot_scores(q, keys) * scale
    rotated = progressive_dot_scores(signed_hadamard(q, seed=seed), signed_hadamard(keys, seed=seed), DIMENSIONS)
    raw = progressive_dot_scores(q, keys, DIMENSIONS)
    for dimension in DIMENSIONS:
        add_metrics(records["rotated_prefix"][layer][dimension], rotated[dimension] * scale, full)
        add_metrics(
            seed_records["rotated_prefix"][seed][layer][dimension], rotated[dimension] * scale, full
        )
        if include_raw:  # Raw prefix has no transform seed; don't duplicate its samples.
            add_metrics(records["raw_prefix"][layer][dimension], raw[dimension] * scale, full)
        indices = random_coordinate_indices(head_dim, dimension, seed=seed)
        random_score = (head_dim / dimension) * (q[indices] * keys[:, indices]).sum(-1)
        add_metrics(records["random_coordinates"][layer][dimension], random_score * scale, full)
        add_metrics(
            seed_records["random_coordinates"][seed][layer][dimension], random_score * scale, full
        )


def print_table(title: str, metrics: dict[str, dict[str, dict[str, Any]]]) -> None:
    print(title)
    print("method              dim  Pearson  Spearman  recall@8  relmass@8  n@8")
    for method in METHODS:
        for dimension in DIMENSIONS:
            values = metrics[method][str(dimension)]
            def number(name: str, row: dict[str, Any] = values) -> str:
                value = row[name]
                return f"{value:.4f}" if value is not None else "n/a"
            print(
                f"{method:19} {dimension:3d}  {number('pearson'):>7}  {number('spearman'):>8}  "
                f"{number('top_8_recall'):>8}  {number('relative_attention_mass_top_8'):>9}  "
                f"{values['top_8_sample_count']:3d}"
            )


def main() -> None:
    args = parse_args()
    selected_layers = (args.layer,) if args.layer is not None else args.layers
    if args.max_length < 2 or args.num_query_positions < 1 or args.min_query_position < 1:
        raise ValueError("invalid context or query-position arguments")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # The decoder body runs the exact attention stack but avoids allocating the
    # causal-LM logits, which are irrelevant to Q/K capture and costly at 256 tokens.
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
    model.eval()
    layers = model.layers
    if any(layer < 0 or layer >= len(layers) for layer in selected_layers):
        raise ValueError(f"--layers must lie in [0, {len(layers) - 1}]")
    captured, handles = capture_post_rope_qk({layer: layers[layer].self_attn for layer in selected_layers})
    records: dict[str, dict[int, dict[int, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(new_record))
    )
    seed_records: dict[str, dict[int, dict[int, dict[int, dict[str, list[float]]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(new_record)))
    )
    skipped: list[dict[str, int | str]] = []
    context_lengths: list[int] = []
    total_samples = forwards = 0
    try:
        with torch.inference_mode():
            for text_index, text in enumerate(long_inputs()):
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                length = int(tokens["input_ids"].shape[-1])
                context_lengths.append(length)
                positions = query_positions(length, args.num_query_positions, args.min_query_position)
                if not positions:
                    skipped.append({"text_index": text_index, "token_length": length, "reason": "shorter than min_query_position"})
                    continue
                model(**tokens, use_cache=False)
                forwards += 1
                for layer in selected_layers:
                    query, key = (item[0] for item in captured[layer])
                    for position in positions:
                        for query_head in range(query.shape[0]):
                            key_head = gqa_kv_head_for_query(query_head, query.shape[0], key.shape[0])
                            q, preceding_keys = query[query_head, position], key[key_head, :position]
                            for seed_index, seed in enumerate(args.seeds):
                                add_sample_methods(
                                    records, seed_records, layer, seed, q, preceding_keys,
                                    include_raw=seed_index == 0,
                                )
                            total_samples += 1
    finally:
        for handle in handles:
            handle.remove()

    per_layer = {
        str(layer): {method: {str(dimension): summarize(records[method][layer][dimension]) for dimension in DIMENSIONS} for method in METHODS}
        for layer in selected_layers
    }
    aggregate_records: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(new_record))
    for method in METHODS:
        for layer in selected_layers:
            for dimension in DIMENSIONS:
                for name, values in records[method][layer][dimension].items():
                    aggregate_records[method][dimension][name].extend(values)
    aggregate = {method: {str(dimension): summarize(aggregate_records[method][dimension]) for dimension in DIMENSIONS} for method in METHODS}
    per_seed: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for method in ("rotated_prefix", "random_coordinates"):
        per_seed[method] = {}
        for seed in args.seeds:
            aggregate_seed: dict[int, dict[str, list[float]]] = defaultdict(new_record)
            for layer in selected_layers:
                for dimension in DIMENSIONS:
                    for name, values in seed_records[method][seed][layer][dimension].items():
                        aggregate_seed[dimension][name].extend(values)
            per_seed[method][str(seed)] = {
                str(dimension): summarize(aggregate_seed[dimension]) for dimension in DIMENSIONS
            }
    result = {
        "model": "Qwen/Qwen3-0.6B", "layers": list(selected_layers), "seeds": list(args.seeds),
        "context_lengths": context_lengths,
        "query_positions": {"minimum": args.min_query_position, "per_context": [query_positions(length, args.num_query_positions, args.min_query_position) for length in context_lengths]},
        "total_model_forwards": forwards, "total_evaluated_query_head_samples": total_samples,
        "capture": "post-q_norm/k_norm and post-RoPE, before cache update; all Q heads map to KV heads by GQA group",
        "score_scaling": "partial scores use d/m before normal 1/sqrt(d) attention scale; this does not affect ranking",
        "skipped_contexts": skipped, "aggregate": aggregate, "per_layer": per_layer,
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Qwen3 partial-dot sweep: {forwards} forwards, {total_samples} query/head samples")
    print_table("Aggregate", aggregate)
    for layer in selected_layers:
        print_table(f"Layer {layer}", per_layer[str(layer)])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
