#!/usr/bin/env python3
"""Measure progressive Hadamard partial-dot rankings on real Qwen3 attention Q/K."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

# Running this file directly places ``experiments/`` rather than the repository
# root on sys.path.  Keep the experiment runnable with the documented command.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascadekv.hadamard import signed_hadamard
from cascadekv.scorer import (
    attention_mass_recall,
    full_dot_scores,
    pearson_correlation,
    progressive_dot_scores,
    spearman_correlation,
    top_k_recall,
)


PROMPTS = [
    "The old library smelled of paper and rain, and the librarian quietly explained",
    "A good experiment begins with a clear question, a simple baseline, and careful notes about",
    "After lunch, Maya walked beside the river and noticed that the water reflected",
    "The recipe calls for onions, garlic, tomatoes, and a pinch of salt before you",
    "When the train finally arrived, the passengers found seats and watched the city",
    "Learning a new language is easier when short daily practice becomes part of",
    (
        "On Saturday morning, Daniel packed a notebook, a bottle of water, and a sandwich before "
        "walking to the neighborhood park. He chose a bench beneath an oak tree and spent an hour "
        "writing down the birds he could identify. A child flew a bright kite nearby while "
        "cyclists "
        "followed the path beside the pond."
    ),
    (
        "The team agreed to begin the project with a modest prototype. First they listed the "
        "inputs that mattered, then they wrote a small test for each expected outcome. Every "
        "afternoon they compared the new measurements with the baseline and recorded surprising "
        "results. By Friday, "
        "the report explained both what worked and what still needed careful investigation."
    ),
]
DIMENSIONS = (16, 32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-query-positions", type=int, default=8)
    return parser.parse_args()


def evenly_spaced_positions(length: int, count: int) -> list[int]:
    """Choose causal query positions with at least one strictly preceding key."""
    if length < 2:
        return []
    first = min(8, length - 1)
    count = min(count, length - first)
    return sorted(
        {first + (index * (length - 1 - first)) // max(count - 1, 1) for index in range(count)}
    )


def capture_post_rope_qk(attention: torch.nn.Module) -> tuple[dict[str, torch.Tensor], Any]:
    """Install a non-invasive pre-hook that reproduces Qwen3's post-RoPE Q/K.

    Qwen3's forward path is q_proj/k_proj -> per-head RMSNorm -> RoPE -> cache
    update.  The hook observes the actual attention input and position embeddings,
    then performs those same first three operations.  It does not alter the model.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        hidden_states = kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        captured["q"] = query.detach().float().cpu()
        captured["k"] = key.detach().float().cpu()

    return captured, attention.register_forward_pre_hook(hook, with_kwargs=True)


def add_metrics(
    records: dict[int, dict[str, list[float]]],
    approximate: dict[int, torch.Tensor],
    full: torch.Tensor,
) -> None:
    for dimension, scores in approximate.items():
        metric = records[dimension]
        metric["pearson"].append(pearson_correlation(scores, full))
        metric["spearman"].append(spearman_correlation(scores, full))
        for top_k in (8, 32):
            if full.shape[-1] >= top_k:
                metric[f"top_{top_k}_recall"].append(top_k_recall(scores, full, top_k))
                metric[f"attention_mass_top_{top_k}"].append(
                    attention_mass_recall(scores, full, top_k)
                )


def main() -> None:
    args = parse_args()
    if args.max_length < 2 or args.num_query_positions < 1:
        raise ValueError(
            "--max-length must be at least 2 and --num-query-positions must be positive"
        )
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
    model.eval()
    layers = model.model.layers
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"--layer must be in [0, {len(layers) - 1}]")
    captured, handle = capture_post_rope_qk(layers[args.layer].self_attn)
    records: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sample_count = 0
    try:
        with torch.inference_mode():
            for prompt_index, prompt in enumerate(PROMPTS):
                tokens = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=args.max_length
                )
                model(**tokens, use_cache=False)
                query, key = captured["q"][0], captured["k"][0]
                for position in evenly_spaced_positions(query.shape[1], args.num_query_positions):
                    # Qwen3 uses grouped-query attention: map each query head to its real KV head.
                    query_head = (prompt_index + position) % query.shape[0]
                    key_head = query_head // (query.shape[0] // key.shape[0])
                    q = query[query_head, position]
                    keys = key[key_head, :position]  # strictly preceding causal cache positions
                    q_rot = signed_hadamard(q, seed=args.seed)
                    k_rot = signed_hadamard(keys, seed=args.seed)
                    scale = 1.0 / math.sqrt(q.numel())
                    full = full_dot_scores(q, keys) * scale
                    partial = {
                        d: score * scale
                        for d, score in progressive_dot_scores(q_rot, k_rot, DIMENSIONS).items()
                    }
                    add_metrics(records, partial, full)
                    sample_count += 1
    finally:
        handle.remove()

    metrics = {
        str(dimension): {
            name: sum(values) / len(values) for name, values in values_by_name.items() if values
        }
        for dimension, values_by_name in sorted(records.items())
    }
    result = {
        "model": "Qwen/Qwen3-0.6B",
        "layer": args.layer,
        "seed": args.seed,
        "max_length": args.max_length,
        "num_query_positions": args.num_query_positions,
        "samples": sample_count,
        "capture": (
            "post-q_norm/k_norm and post-RoPE, before cache update; KV heads are mapped for GQA"
        ),
        "metrics": metrics,
    }
    output = Path("results") / f"partial_dot_layer_{args.layer}_seed_{args.seed}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Qwen3 partial-dot experiment: {sample_count} query/head samples")
    print("dim  Pearson  Spearman  top-8  top-32  mass@8  mass@32")
    for dimension, values in metrics.items():
        def value(name: str) -> str:
            return f"{values[name]:.4f}" if name in values else "n/a"

        print(
            f"{int(dimension):3d}  {value('pearson'):>7}  {value('spearman'):>8}"
            f"  {value('top_8_recall'):>5}  {value('top_32_recall'):>6}"
            f"  {value('attention_mass_top_8'):>6}  {value('attention_mass_top_32'):>7}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
