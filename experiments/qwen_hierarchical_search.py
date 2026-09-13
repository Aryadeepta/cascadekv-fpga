#!/usr/bin/env python3
"""Smoke-first Qwen3 token-axis hierarchical KV search experiment.

The default is intentionally one 4096-token context and one layer.  Increase
``--contexts`` only after inspecting that output; this script never extends a
model context window beyond its configured tokenizer/model support.
"""

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

from cascadekv.cascade import Q8K4Cascade
from cascadekv.hierarchical_index import HierarchicalIndex, SearchAccounting, accounting_ratios
from cascadekv.scorer import (
    attention_mass_recall,
    gqa_kv_head_for_query,
    relative_attention_mass_recall,
    top_k_recall,
)
from experiments.qwen_partial_dot import (
    capture_layer0_post_rope_qk_low_memory,
    capture_post_rope_qk,
)


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=csv_ints, default=(4096,))
    parser.add_argument("--layers", type=csv_ints, default=(0,))
    parser.add_argument("--queries-per-context", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--orders", type=Path, default=Path("results/coordinate_orders.json"))
    parser.add_argument("--output", type=Path, default=Path("results/hierarchical_search_sweep.json"))
    parser.add_argument("--synthetic-smoke", action="store_true", help="validate mechanics without loading Qwen")
    parser.add_argument("--synthetic-case", choices=("gaussian", "clustered", "needle", "distributed", "all"), default="all")
    parser.add_argument("--adaptive-policy", choices=("descend_immediately", "ambiguity_refine"), default="ambiguity_refine")
    parser.add_argument("--ambiguity-margin", type=float, default=1.0)
    parser.add_argument("--low-memory-layer0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-low-memory-capture", action="store_true", help="also run a small normal forward and compare layer-0 Q/K")
    return parser.parse_args()


def needle_context(target: int) -> str:
    filler = "The archive recorded weather, routes, instruments, and observations in careful chronological order. "
    needle = "DISTINCTIVE NEEDLE FACT: the amber falcon is stored behind shelf seven under the word heliotrope. "
    # Place the fact at several depths, while the final question probes retrieval-oriented late queries.
    pieces = [filler] * (target * 3)
    for position in (len(pieces) // 8, len(pieces) // 2, 7 * len(pieces) // 8):
        pieces[position] = needle
    return "".join(pieces) + " What object is stored behind shelf seven?"


def score_metrics(ids8: torch.Tensor, ids32: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    if full.numel() >= 32 and ids32.numel() < 32:
        raise ValueError("top32 evaluation requires a search candidate set of at least 32 leaves")
    approximate = torch.full_like(full, float("-inf"))
    approximate[ids8] = full[ids8]
    approximate32 = torch.full_like(full, float("-inf"))
    approximate32[ids32] = full[ids32]
    return {
        "top8_recall": top_k_recall(approximate, full, 8), "top32_recall": top_k_recall(approximate32, full, 32),
        "relative_attention_mass_top8": relative_attention_mass_recall(approximate, full, 8),
        "relative_attention_mass_top32": relative_attention_mass_recall(approximate32, full, 32),
        "attention_mass_top8": attention_mass_recall(approximate, full, 8),
        "attention_mass_top32": attention_mass_recall(approximate32, full, 32),
    }


def bound_looseness(tree: HierarchicalIndex, query: torch.Tensor) -> dict[str, float]:
    """FP32 slack of the primary conservative 64-D routing bound."""
    slacks = torch.tensor([
        tree.node_bound(node.id, query, dimensions=64) - (tree.keys[node.start : node.end] @ query).max()
        for node in tree.nodes
    ])
    return {"average_bound_looseness": float(slacks.mean()), "p95_bound_looseness": float(torch.quantile(slacks, .95))}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"sample_count": len(rows)}
    for key in rows[0]:
        if isinstance(rows[0][key], (int, float)):
            values = [float(row[key]) for row in rows]
            values.sort()
            output[key] = sum(values) / len(values)
            output[f"p95_{key}"] = values[min(len(values) - 1, int(.95 * len(values)))]
    return output


def evaluate_one(query: torch.Tensor, keys: torch.Tensor, order: torch.Tensor, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    tree = HierarchicalIndex(keys, order)
    full = keys @ query
    slack = bound_looseness(tree, query)
    methods: dict[str, tuple[torch.Tensor, SearchAccounting, torch.Tensor, SearchAccounting]] = {}
    dense = torch.arange(keys.shape[0])
    dense_accounting = SearchAccounting(leaf_tokens_evaluated=keys.shape[0], leaf_k_bytes_read=keys.numel() * 4)
    methods["dense_exact_fp32"] = (dense, dense_accounting, dense, dense_accounting)
    progressive = {0: 16, 1: 32, 2: 64}
    searchers = {
        "mean_key_hierarchy": lambda k: tree.beam_search(query, beam_width=max(args.beam_width, k), k=k, summary="mean_key"),
        "full_128d_box": lambda k: tree.best_first_search(query, k=k, summary="full_box_bound"),
        "ordered_progressive_bound": lambda k: tree.best_first_search(query, k=k, schedule=progressive),
        "adaptive_16_32_64_128": lambda k: tree.best_first_search(query, k=k, adaptive=True, adaptive_policy=args.adaptive_policy, ambiguity_margin=args.ambiguity_margin),
        "beam_progressive": lambda k: tree.beam_search(query, beam_width=max(args.beam_width, k), k=k, schedule=progressive),
    }
    for name, search in searchers.items():
        eight, thirty_two = search(8), search(min(32, keys.shape[0]))
        methods[name] = (eight.ids, eight.accounting, thirty_two.ids, thirty_two.accounting)
    generator = torch.Generator().manual_seed(0)
    random_ids = torch.randperm(keys.shape[0], generator=generator)[:args.beam_width]
    random32 = torch.randperm(keys.shape[0], generator=generator)[:min(32, keys.shape[0])]
    methods["random_hierarchical_descent"] = (random_ids, SearchAccounting(leaf_tokens_evaluated=random_ids.numel(), leaf_k_bytes_read=random_ids.numel() * 128 * 4), random32, SearchAccounting(leaf_tokens_evaluated=random32.numel(), leaf_k_bytes_read=random32.numel() * 128 * 4))
    # Quest-like: one 16-token min/max page level, then exact page candidates.
    pages = HierarchicalIndex(keys, order, fanout=16)
    page_nodes = pages.levels[1] if pages.max_level >= 1 else pages.levels[0]
    ranked = sorted(page_nodes, key=lambda n: -pages.node_bound(n, query, summary="full_box_bound"))[:args.beam_width]
    page_ids = torch.tensor([token for node in ranked for token in range(pages.nodes[node].start, pages.nodes[node].end)])
    methods["quest_like_minmax_pages"] = (page_ids, SearchAccounting(leaf_tokens_evaluated=page_ids.numel(), leaf_k_bytes_read=page_ids.numel() * 128 * 4), page_ids, SearchAccounting(leaf_tokens_evaluated=page_ids.numel(), leaf_k_bytes_read=page_ids.numel() * 128 * 4))
    q8 = Q8K4Cascade(query, keys, order).direct_prefix_scores(128)
    q8_ids = q8.topk(min(32, keys.shape[0])).indices
    methods["flat_q8xk4_scan"] = (q8_ids[:8], SearchAccounting(leaf_tokens_evaluated=keys.shape[0], leaf_k_bytes_read=keys.shape[0] * 80), q8_ids, SearchAccounting(leaf_tokens_evaluated=keys.shape[0], leaf_k_bytes_read=keys.shape[0] * 80))
    output = {}
    for name, (ids8, accounting8, ids32, accounting32) in methods.items():
        row = score_metrics(ids8, ids32, full)
        row.update({f"top8_{key}": value for key, value in accounting_ratios(accounting8, keys.shape[0]).items()})
        row.update({f"top32_{key}": value for key, value in accounting_ratios(accounting32, keys.shape[0]).items()})
        row.update({"nodes_visited": accounting32.nodes_visited, "node_expansions": accounting32.node_expansions,
                    "bound_violations": tree.validate_bounds(query.unsqueeze(0)) if "progressive" in name or "adaptive" in name else 0,
                    "refinements_16_to_32": accounting32.refinements_16_to_32, "refinements_32_to_64": accounting32.refinements_32_to_64,
                    "refinements_64_to_128": accounting32.refinements_64_to_128})
        row.update(slack)
        output[name] = row
    return output


def synthetic_capture(case: str, generator: torch.Generator, count: int = 4096) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Controls: Gaussian is a negative control, not a KV proxy."""
    query = torch.randn(128, generator=generator)
    keys = torch.randn(count, 128, generator=generator)
    if case == "clustered":
        # Contiguous fanout descendants share two nested latent centers.
        coarse = torch.randn((count + 255) // 256, 128, generator=generator) * 2.0
        fine = torch.randn((count + 15) // 16, 128, generator=generator) * 1.0
        keys = keys * .15 + coarse.repeat_interleave(256, 0)[:count] + fine.repeat_interleave(16, 0)[:count]
        query = keys[count // 2] + .05 * torch.randn(128, generator=generator)
    elif case == "needle":
        # A few isolated leaves are deliberately made overwhelmingly relevant.
        for offset in (127, count // 2 + 7, count - 11):
            keys[offset] += query * 12.0
    elif case == "distributed":
        # Moderate hits placed across many independent level-one branches.
        for offset in range(7, count, 64):
            keys[offset] += query * .8
    elif case != "gaussian":
        raise ValueError(f"unknown synthetic case: {case}")
    return query, keys, torch.randperm(128, generator=generator)


def captures(args: argparse.Namespace) -> list[tuple[str, int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if args.synthetic_smoke:
        generator = torch.Generator().manual_seed(42)
        cases = ("gaussian", "clustered", "needle", "distributed") if args.synthetic_case == "all" else (args.synthetic_case,)
        return [(f"synthetic_{case}", 4096, 0, *synthetic_capture(case, generator)) for case in cases]
    frozen = json.loads(args.orders.read_text())
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).eval()
    selected = {layer: model.layers[layer].self_attn for layer in args.layers}
    captured, handles = capture_post_rope_qk(selected)
    result = []
    try:
        with torch.inference_mode():
            for requested in args.contexts:
                tokens = tokenizer(needle_context(requested), return_tensors="pt", truncation=True, max_length=requested)
                if args.low_memory_layer0 and args.layers == (0,):
                    low_memory_qk = capture_layer0_post_rope_qk_low_memory(model, tokens.input_ids)
                    if args.verify_low_memory_capture:
                        model(**tokens, use_cache=False)
                        full_qk = captured[0]
                        q_error = (low_memory_qk[0] - full_qk[0]).abs().max().item()
                        k_error = (low_memory_qk[1] - full_qk[1]).abs().max().item()
                        tolerance = 5e-3 if model.dtype == torch.float16 else 1e-5
                        print(f"layer-0 low-memory capture agreement: max_abs_q={q_error:.3g} max_abs_k={k_error:.3g}")
                        if q_error > tolerance or k_error > tolerance:
                            raise RuntimeError(f"low-memory layer-0 Q/K capture disagrees with normal forward (tolerance {tolerance})")
                    captured[0] = low_memory_qk
                else:
                    model(**tokens, use_cache=False)
                length = int(tokens.input_ids.shape[-1])
                for layer in args.layers:
                    query, key = captured[layer]
                    for position in torch.linspace(max(128, length // 2), length - 1, args.queries_per_context).long().tolist():
                        qhead = 0
                        kvhead = gqa_kv_head_for_query(qhead, query.shape[1], key.shape[1])
                        result.append(("qwen_layer0_low_memory" if args.low_memory_layer0 and layer == 0 else "qwen_full_model", length, layer, query[0, qhead, position], key[0, kvhead, :position], torch.tensor(frozen[str(layer)]["energy"])))
    finally:
        for handle in handles:
            handle.remove()
    return result


def main() -> None:
    args = parse_args()
    records: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for case, length, layer, query, keys, order in captures(args):
        for name, row in evaluate_one(query, keys, order, args).items():
            records[case][str(length)][f"layer_{layer}:{name}"].append(row)
    result = {"experiment": "hierarchical_token_search", "model": "structured synthetic controls (Gaussian is negative control)" if args.synthetic_smoke else "Qwen/Qwen3-0.6B",
              "hierarchy_fanout": 16, "block_sizes": [1, 16, 256, 4096, 65536], "search_modes": ["beam", "best_first"],
              "beam_widths": [args.beam_width], "prefix_widths": [16, 32, 64, 128], "coordinate_order": "frozen per-layer contribution_energy",
              "contexts_requested": list(args.contexts), "needle_protocol": "three fixed distinctive facts at 1/8, 1/2, 7/8 depth plus late retrieval query",
              "results": {case: {length: {name: aggregate(rows) for name, rows in methods.items()} for length, methods in lengths.items()} for case, lengths in records.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
