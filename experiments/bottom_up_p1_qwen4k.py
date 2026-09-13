#!/usr/bin/env python3
"""Final software gate: direct versus online/bottom-up P1 on Qwen layer 0.

Only the two fixed-point conservative P1 modes are evaluated.  The context is
explicitly labelled as a deterministic fallback when no held-out text file is
present in this repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascadekv.hierarchical_index import (
    HierarchicalIndex,
    accounting_ratios,
)
from cascadekv.scorer import relative_attention_mass_recall, top_k_recall
from experiments.qwen_hierarchical_search import needle_context
from experiments.qwen_partial_dot import capture_layer0_post_rope_qk_low_memory

OUTPUT = Path("results/bottom_up_p1_qwen4k.json")


def percentile(values: list[float], p: int) -> float:
    values = sorted(values)
    return values[max(0, (p * len(values) + 99) // 100 - 1)]


def summarize(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    output: dict[str, float | int] = {"query_head_count": len(rows)}
    for key in rows[0]:
        values = [float(row[key]) for row in rows]
        output[f"mean_{key}"] = sum(values) / len(values)
        for p in (50, 90, 95, 99):
            output[f"p{p}_{key}"] = percentile(values, p)
        output[f"max_{key}"] = max(values)
    return output


def main() -> None:
    # The experiment is intentionally CPU-friendly; the host can expose far
    # more logical workers than this small layer-0 capture benefits from.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    orders = json.loads(Path("results/coordinate_orders.json").read_text())
    order = torch.tensor(orders["0"]["energy"])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModel.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float16, low_cpu_mem_usage=True
    ).eval()
    # There is no genuine held-out text document in this repository.  Keep the
    # fallback declaration in the result rather than treating it as corpus data.
    tokens = tokenizer(needle_context(4096), return_tensors="pt", truncation=True, max_length=4096)
    with torch.inference_mode():
        queries, all_keys = capture_layer0_post_rope_qk_low_memory(model, tokens.input_ids)
    queries, all_keys = queries.float().cpu(), all_keys.float().cpu()
    print("Captured layer-0 Q/K; evaluating hierarchy.", flush=True)
    length = int(tokens.input_ids.shape[-1])
    positions = torch.linspace(max(128, length // 2), length - 1, 4).long().tolist()
    methods = {
        "direct_p1_rtl_conservative": "support_set_fixed_q8k4_global",
        "bottom_up_p1_rtl_conservative": "support_p1_bottom_up_conservative",
    }
    rows: dict[str, list[dict[str, float | int]]] = {name: [] for name in methods}
    inflation_values: dict[str, list[float]] = {"16": [], "256": [], "4096": []}
    zero_direct: dict[str, int] = {"16": 0, "256": 0, "4096": 0}
    violations = {name: 0 for name in methods}
    for position in positions:
        for kvhead in range(all_keys.shape[1]):
            keys = all_keys[0, kvhead, :position].contiguous()
            tree = HierarchicalIndex(keys, order, support_set_sizes=(1,))
            # Keep raw ratios across every indexed tree rather than averaging
            # pre-aggregated per-tree percentiles.
            for level in range(1, tree.max_level + 1):
                label = str(tree.fanout**level)
                if label not in inflation_values:
                    continue
                for node_id in tree.levels[level]:
                    direct = float(tree._support_k4_global[1, node_id])
                    bottom_up = float(tree._bottom_up_p1_radius[node_id])
                    if direct == 0.0:
                        zero_direct[label] += 1
                    else:
                        inflation_values[label].append(bottom_up / direct)
            # Qwen3-0.6B has grouped-query attention: each KV head owns a
            # contiguous equal-width group of Q heads.
            q_per_kv = queries.shape[1] // all_keys.shape[1]
            for qhead in range(kvhead * q_per_kv, (kvhead + 1) * q_per_kv):
                query = queries[0, qhead, position]
                full = keys @ query
                exact_ids = full.topk(8).indices
                for name, summary in methods.items():
                    result = tree.best_first_search(query, k=8, summary=summary)
                    approximate = torch.full_like(full, float("-inf"))
                    approximate[result.ids] = full[result.ids]
                    ratios = accounting_ratios(result.accounting, keys.shape[0])
                    rows[name].append({
                        "top8_recall": top_k_recall(approximate, full, 8),
                        "relmass8": relative_attention_mass_recall(approximate, full, 8),
                        "bound_violations": tree.validate_bounds(query.unsqueeze(0), summary=summary),
                        "leaf_fraction": ratios["fraction_leaf_k_read"],
                        "projected_traffic_vs_dense_fp16": ratios["projected_total_bytes_vs_dense_fp16"],
                        "nodes_visited": result.accounting.nodes_visited,
                        "node_expansions": result.accounting.node_expansions,
                        "authoritative_leaf_k_bytes": result.accounting.authoritative_leaf_k_bytes_read,
                        "top8_exact_match": int(torch.equal(result.ids, exact_ids)),
                    })
                    violations[name] += int(rows[name][-1]["bound_violations"])
    results = {name: summarize(value) for name, value in rows.items()}
    inflation: dict[str, dict[str, float | int]] = {}
    for level, values in inflation_values.items():
        row: dict[str, float | int] = {
            "finite_ratio_count": len(values), "zero_direct_radius_count": zero_direct[level],
        }
        if values:
            row.update({"mean": sum(values) / len(values), "p50": percentile(values, 50),
                        "p90": percentile(values, 90), "p95": percentile(values, 95),
                        "p99": percentile(values, 99), "max": max(values)})
        inflation[level] = row
    payload = {
        "experiment": "bottom_up_p1_qwen4k_final_software_gate",
        "model": "Qwen/Qwen3-0.6B", "layer": 0, "context_tokens": length,
        "context_source": "deterministic needle fallback; no genuine held-out text document was present in the repository",
        "query_positions": positions, "all_q_heads": True,
        "query_head_count": len(rows["direct_p1_rtl_conservative"]),
        "modes": methods,
        "bottom_up_construction": "parent prototype selected only from frozen child prototypes; parent radius is triangle envelope over child balls; no descendant leaf scan",
        "fixed_point_bound": "Q8 query, frozen group16 K4 prototype, U0.16 x U7.9 -> Q4.12, Q11.13 RTL score, deterministic arithmetic cushion, and Q quantization-error cushion",
        "bound_violations_total": violations,
        "radius_inflation_bottom_up_over_direct": inflation,
        "results": results,
        "rtl_gate": {
            "passes": violations["bottom_up_p1_rtl_conservative"] == 0
            and results["bottom_up_p1_rtl_conservative"]["mean_top8_recall"] == 1.0
            and results["bottom_up_p1_rtl_conservative"]["p95_projected_traffic_vs_dense_fp16"] < 0.5,
            "criteria": "zero violations, top8 recall 1.0, p95 projected traffic < 0.5x dense FP16",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
