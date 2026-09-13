#!/usr/bin/env python3
"""Smoke-first Qwen3 token-axis hierarchical KV search experiment.

The default is intentionally one 4096-token context and one layer.  Increase
``--contexts`` only after inspecting that output; this script never extends a
model context window beyond its configured tokenizer/model support.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascadekv.cascade import Q8K4Cascade
from cascadekv.hierarchical_index import (
    HierarchicalIndex,
    SearchAccounting,
    accounting_ratios,
    support_index_storage,
)
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
    parser.add_argument("--queries-per-context", type=int, default=8)
    parser.add_argument("--all-q-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context-kinds", type=str, default="needle,natural_a,natural_b")
    parser.add_argument("--beam-width", type=int, default=16, help="legacy box-beam width")
    parser.add_argument("--support-beam-widths", type=csv_ints, default=(4, 8, 16, 32))
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


def ordinary_context(target: int, seed: int) -> str:
    """Deterministic local text fallback; it never needs a dataset download."""
    passages = (
        "A careful reader compares claims with dates, definitions, and concrete examples. ",
        "The workshop notes describe a river survey, a repair log, and a short discussion of uncertainty. ",
        "In the evening the team summarized observations, assigned follow-up work, and preserved the source record. ",
        "Technical writing benefits from explicit assumptions, reproducible measurements, and clear limitations. ",
    )
    return "".join(passages[(i * 5 + seed) % len(passages)] for i in range(target * 3))


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


def bound_looseness(tree: HierarchicalIndex, query: torch.Tensor, *, summary: str = "ordered_prefix_plus_residual_bound",
                    representatives: int = 1, dimensions: int = 64) -> dict[str, float]:
    """FP32 slack of the primary conservative 64-D routing bound."""
    slacks = torch.tensor([
        tree.node_bound(node.id, query, summary=summary, representatives=representatives, dimensions=dimensions)
        - (tree.keys[node.start : node.end] @ query).max()
        for node in tree.nodes
    ])
    return {"mean_bound_slack": float(slacks.mean()), "p95_bound_slack": float(torch.quantile(slacks, .95))}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribution summary using inclusive nearest-rank percentiles."""
    output: dict[str, Any] = {"sample_count": len(rows)}
    for key in rows[0]:
        if isinstance(rows[0][key], (int, float)):
            values = [float(row[key]) for row in rows]
            values.sort()
            output[key] = sum(values) / len(values)
            for percentile in (5, 50, 90, 95, 99):
                rank = max(0, (percentile * len(values) + 99) // 100 - 1)
                output[f"p{percentile}_{key}"] = values[rank]
            output[f"minimum_{key}"] = values[0]
            output[f"maximum_{key}"] = values[-1]
    return output


def evaluate_one(query: torch.Tensor, keys: torch.Tensor, order: torch.Tensor, args: argparse.Namespace,
                 tree: HierarchicalIndex | None = None) -> dict[str, dict[str, Any]]:
    tree = tree or HierarchicalIndex(keys, order, support_set_sizes=(1, 2, 4))
    full = keys @ query
    methods: dict[str, tuple[torch.Tensor, SearchAccounting, torch.Tensor, SearchAccounting]] = {}
    # Deliberately small architecture-gate method set.
    for p in (1, 2):
        name = f"support_p{p}_per_prototype_exact"
        eight = tree.best_first_search(query, k=8, summary="support_set_per_prototype", representatives=p)
        thirty_two = tree.best_first_search(query, k=min(32, keys.shape[0]), summary="support_set_per_prototype", representatives=p)
        methods[name] = (eight.ids, eight.accounting, thirty_two.ids, thirty_two.accounting)
    for width in (8, 16):
        name = f"support_p2_beam{width}"
        eight = tree.beam_search(query, beam_width=width, k=8, summary="support_set_score", representatives=2)
        thirty_two = tree.beam_search(query, beam_width=max(width, 32), k=min(32, keys.shape[0]), summary="support_set_score", representatives=2)
        methods[name] = (eight.ids, eight.accounting, thirty_two.ids, thirty_two.accounting)
    for name, summary, p in (
        ("support_p1_k4_prototype_conservative", "support_set_k4_per_prototype", 1),
        ("support_p1_q8k4_conservative", "support_set_q8k4_per_prototype", 1),
        ("support_p1_rtl_q8k4_conservative", "support_set_fixed_q8k4", 1),
        ("support_p2_k4_prototype_conservative", "support_set_k4_per_prototype", 2),
        ("support_p2_q8k4_conservative", "support_set_q8k4_per_prototype", 2),
        ("support_p2_rtl_q8k4_conservative", "support_set_fixed_q8k4", 2),
    ):
        eight = tree.best_first_search(query, k=8, summary=summary, representatives=p)
        thirty_two = tree.best_first_search(query, k=min(32, keys.shape[0]), summary=summary, representatives=p)
        methods[name] = (eight.ids, eight.accounting, thirty_two.ids, thirty_two.accounting)
    q8 = Q8K4Cascade(query, keys, order).direct_prefix_scores(128)
    q8_ids = q8.topk(min(32, keys.shape[0])).indices
    methods["flat_q8xk4_scan"] = (q8_ids[:8], SearchAccounting(leaf_tokens_evaluated=keys.shape[0], leaf_k_bytes_read=keys.shape[0] * 80), q8_ids, SearchAccounting(leaf_tokens_evaluated=keys.shape[0], leaf_k_bytes_read=keys.shape[0] * 80))
    output = {}
    for name, (ids8, accounting8, ids32, accounting32) in methods.items():
        row = score_metrics(ids8, ids32, full)
        row.update({f"top8_{key}": value for key, value in accounting_ratios(accounting8, keys.shape[0]).items()})
        row.update({f"top32_{key}": value for key, value in accounting_ratios(accounting32, keys.shape[0]).items()})
        support_p = int(name.split("_", 2)[1][1:]) if name.startswith("support_p") else 1
        support_summary = "support_set_per_prototype" if "per_prototype" in name else "support_set_global"
        if "k4_prototype" in name:
            support_summary = "support_set_k4_per_prototype"
        elif "rtl_" in name:
            support_summary = "support_set_fixed_q8k4"
        elif "q8k4_conservative" in name:
            support_summary = "support_set_q8k4_per_prototype"
        is_support = name.startswith("support_p") and "beam" not in name
        row.update({"representatives_per_node": support_p if name.startswith("support_p") else 0,
                    "exact_or_approximate": "approximate" if "beam" in name else "exact",
                    "nodes_visited": accounting32.nodes_visited, "node_expansions": accounting32.node_expansions,
                    "bound_violations": tree.validate_bounds(query.unsqueeze(0), summary=support_summary,
                                                               representatives=support_p,
                                                               dimensions=64 if "progressive" in name else 128)
                    if is_support else (tree.validate_bounds(query.unsqueeze(0)) if "progressive" in name or "adaptive" in name else 0),
                    "refinements_16_to_32": accounting32.refinements_16_to_32, "refinements_32_to_64": accounting32.refinements_32_to_64,
                    "refinements_64_to_128": accounting32.refinements_64_to_128})
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


def captures(args: argparse.Namespace):
    """Yield one captured context at a time, so old Q/K captures can die."""
    if args.synthetic_smoke:
        generator = torch.Generator().manual_seed(42)
        cases = ("gaussian", "clustered", "needle", "distributed") if args.synthetic_case == "all" else (args.synthetic_case,)
        for case in cases:
            query, key, order = synthetic_capture(case, generator)
            yield f"synthetic_{case}", 4096, 0, query[None, None, :].expand(1, 1, 4096, -1), key[None, None], order, True
        return
    frozen = json.loads(args.orders.read_text())
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).eval()
    selected = {layer: model.layers[layer].self_attn for layer in args.layers}
    captured, handles = capture_post_rope_qk(selected)
    try:
        with torch.inference_mode():
            kinds = tuple(item.strip() for item in args.context_kinds.split(",") if item.strip())
            for requested in args.contexts:
                for kind_index, kind in enumerate(kinds):
                    text = needle_context(requested) if kind == "needle" else ordinary_context(requested, kind_index)
                    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=requested)
                    if args.low_memory_layer0 and args.layers == (0,):
                        low_memory_qk = capture_layer0_post_rope_qk_low_memory(model, tokens.input_ids)
                        if args.verify_low_memory_capture:
                            model(**tokens, use_cache=False)
                            full_qk = captured[0]
                            q_error = (low_memory_qk[0] - full_qk[0]).abs().max().item()
                            k_error = (low_memory_qk[1] - full_qk[1]).abs().max().item()
                            tolerance = 5e-3 if model.dtype == torch.float16 else 1e-5
                            if q_error > tolerance or k_error > tolerance:
                                raise RuntimeError("low-memory capture disagreement")
                        captured[0] = low_memory_qk
                    else:
                        model(**tokens, use_cache=False)
                    length = int(tokens.input_ids.shape[-1])
                    for layer in args.layers:
                        query, key = captured[layer]
                        yield kind, length, layer, query.detach().float().cpu(), key.detach().float().cpu(), torch.tensor(frozen[str(layer)]["energy"]), layer == args.layers[-1]
    finally:
        for handle in handles:
            handle.remove()


def result_payload(
    args: argparse.Namespace,
    records: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    by_q_head: dict[str, list[dict[str, Any]]],
    by_kv_head: dict[str, list[dict[str, Any]]],
    by_position_bucket: dict[str, list[dict[str, Any]]],
    completed_contexts: list[str],
) -> dict[str, Any]:
    """Serialize all results available at a context-completion boundary."""
    return {
        "experiment": "hierarchical_token_search",
        "model": "structured synthetic controls (Gaussian is negative control)" if args.synthetic_smoke else "Qwen/Qwen3-0.6B",
        "hierarchy_fanout": 16,
        "block_sizes": [1, 16, 256, 4096, 65536],
        "search_modes": ["beam", "best_first"],
        "beam_widths": [args.beam_width],
        "prefix_widths": [16, 32, 64, 128],
        "coordinate_order": "frozen per-layer contribution_energy",
        "contexts_requested": list(args.contexts),
        "context_kinds": args.context_kinds,
        "completed_contexts": completed_contexts,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "index_storage_projection": {
            str(n): {f"P{p}": support_index_storage(n, p) for p in (1, 2, 4)}
            for n in (4096, 16384, 65536)
        },
        "breakdowns": {
            "per_q_head": {key: aggregate(value) for key, value in by_q_head.items()},
            "per_kv_head": {key: aggregate(value) for key, value in by_kv_head.items()},
            "by_query_position_bucket": {
                key: aggregate(value) for key, value in by_position_bucket.items()
            },
        },
        "needle_protocol": "three fixed distinctive facts at 1/8, 1/2, 7/8 depth plus late retrieval query",
        "results": {
            case: {
                length: {name: aggregate(rows) for name, rows in methods.items()}
                for length, methods in lengths.items()
            }
            for case, lengths in records.items()
        },
    }


def write_checkpoint(
    output: Path,
    args: argparse.Namespace,
    records: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    by_q_head: dict[str, list[dict[str, Any]]],
    by_kv_head: dict[str, list[dict[str, Any]]],
    by_position_bucket: dict[str, list[dict[str, Any]]],
    completed_contexts: list[str],
) -> None:
    """Persist the completed portion of a run before starting the next context."""
    checkpoint = result_payload(
        args, records, by_q_head, by_kv_head, by_position_bucket, completed_contexts
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".progress.json").write_text(json.dumps(checkpoint, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    records: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_q_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_kv_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_position_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completed_contexts: list[str] = []
    for case, length, layer, queries, all_keys, order, context_complete in captures(args):
        positions = torch.linspace(max(128, length // 2), length - 1, args.queries_per_context).long().tolist()
        for position in positions:
            for kvhead in range(all_keys.shape[1]):
                mapped = [head for head in range(queries.shape[1]) if gqa_kv_head_for_query(head, queries.shape[1], all_keys.shape[1]) == kvhead]
                if not args.all_q_heads:
                    mapped = [head for head in mapped if head == 0]
                if not mapped:
                    continue
                keys = all_keys[0, kvhead, :position].contiguous()
                tree = HierarchicalIndex(keys, order, support_set_sizes=(1, 2))
                for qhead in mapped:
                    for name, row in evaluate_one(queries[0, qhead, position], keys, order, args, tree).items():
                        row.update({"q_head": qhead, "kv_head": kvhead, "query_position": position,
                                    "query_position_bucket": int(4 * position / length)})
                        records[case][str(length)][f"layer_{layer}:{name}"].append(row)
                        method = f"layer_{layer}:{name}"
                        by_q_head[f"{method}:q{qhead}"].append(row)
                        by_kv_head[f"{method}:kv{kvhead}"].append(row)
                        by_position_bucket[f"{method}:bucket{row['query_position_bucket']}"].append(row)
                # This is the lifetime boundary: a single tree handles every
                # Q head mapped to this KV head, then no heavy summary survives.
                del tree, keys
        if context_complete:
            completed_contexts.append(case)
            write_checkpoint(args.output, args, records, by_q_head, by_kv_head,
                             by_position_bucket, completed_contexts)
        del queries, all_keys
    result = result_payload(args, records, by_q_head, by_kv_head, by_position_bucket,
                            completed_contexts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
