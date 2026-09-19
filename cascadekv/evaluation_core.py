"""Model-agnostic CascadeKV evaluation mechanics.

This is deliberately a software oracle: policy values are supplied by a
validated :class:`RoutingProfile`; it neither selects sources nor creates
scientific artifacts.  The routines preserve the existing lifting, K4,
reserve, exact-rerank, and traffic semantics while deriving all sizes from
``ModelGeometry``.
"""
from __future__ import annotations

import copy
import heapq
import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from cascadekv.adaptive_lifting import LiftingNode, StreamingLiftingForest, reconstruct_pair
from cascadekv.model_agnostic import ModelGeometry, traffic_accounting
from cascadekv.quantize import quantize_symmetric


def candidate_budget(token_count: int, fraction: float) -> int:
    if token_count < 1 or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("candidate budget requires a positive token count and fraction")
    return max(8, int(token_count * fraction) // 8 * 8)


def query_to_kv_head(geometry: ModelGeometry, q_head: int) -> int:
    return geometry.kv_head_for_query(q_head)


def select_flat_k4(query: torch.Tensor, keys: torch.Tensor, budget: int) -> set[int]:
    if not 0 < budget <= keys.shape[0]:
        raise ValueError("flat candidate budget outside key range")
    sketch = quantize_symmetric(keys, 4, 16).dequantize() @ quantize_symmetric(query, 8, 16).dequantize()
    return set(sketch.topk(budget).indices.tolist())


def exact_attention_reference(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, scaling: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scaling = 1.0 / math.sqrt(query.numel()) if scaling is None else scaling
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("attention scaling must be finite and positive")
    logits = (keys.float() @ query.float()) * scaling
    probabilities = torch.softmax(logits, 0)
    return logits, probabilities, probabilities @ values.float()


def exact_rerank_and_output(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, ids: set[int], *, reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None, scaling: float | None = None) -> dict[str, float]:
    if not ids:
        raise ValueError("exact rerank requires candidates")
    logits, probabilities, dense = reference if reference is not None else exact_attention_reference(query, keys, values, scaling)
    index = torch.tensor(sorted(ids), dtype=torch.long)
    scale = 1.0 / math.sqrt(query.numel()) if scaling is None else scaling
    candidate_logits = (keys[index].float() @ query.float()) * scale
    sparse = torch.softmax(candidate_logits, 0) @ values[index].float()
    delta = sparse - dense
    top_count = min(8, logits.numel())
    absolute = float(torch.linalg.vector_norm(delta))
    return {
        "top8_recall": len(ids & set(logits.topk(top_count).indices.tolist())) / top_count,
        "relative_exact_attention_mass": float(probabilities[index].sum()),
        "relative_l2_error": absolute / max(float(torch.linalg.vector_norm(dense)), 1e-12),
        "absolute_l2_error": absolute,
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(sparse, dense, dim=0)),
    }


def reserve_ids(token_count: int, sink: int, local: int, budget: int) -> set[int]:
    if min(token_count, sink, local, budget) < 0 or sink + local > budget:
        raise ValueError("invalid reserve policy for candidate budget")
    return set(range(min(sink, token_count))) | set(range(max(0, token_count - local), token_count))


def _group_energy(query: torch.Tensor, groups: int) -> torch.Tensor:
    if query.numel() % groups:
        raise ValueError("routing group count must divide head dimension")
    return query.float().square().reshape(groups, query.numel() // groups).sum(1)


@dataclass
class RouteState:
    forest: StreamingLiftingForest
    query: torch.Tensor
    groups: int
    weight: float
    chosen: set[int]
    heap: list[tuple[float, int, torch.Tensor, float]]
    admitted_leaf_priorities: list[float]
    active_root_reads: int
    detail_reads: int = 0
    expanded_internal_nodes: int = 0
    variance_scalar_reads: int = 0
    variance_scalar_products: int = 0
    variance_scalar_adds: int = 0


def _priority(state: RouteState, index: int, mean: torch.Tensor) -> float:
    node = state.forest.nodes[index]
    assert node.grouped_m2 is not None
    state.variance_scalar_reads += state.groups
    state.variance_scalar_products += state.groups
    state.variance_scalar_adds += state.groups - 1
    variance = node.grouped_m2[state.groups] / (node.count * (mean.numel() // state.groups))
    return float(state.query @ mean) + state.weight * math.sqrt(max(0., float(_group_energy(state.query, state.groups) @ variance))) * math.sqrt(2 * math.log(max(node.count, 2)))


def initialize_route(forest: StreamingLiftingForest, query: torch.Tensor, groups: int, weight: float, reserve: set[int]) -> RouteState:
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("routing weight must be finite and nonnegative")
    assert forest.nodes is not None and forest.roots is not None
    state = RouteState(forest, query.float(), groups, weight, set(reserve), [], [], len(forest.roots))
    for root in forest.roots:
        mean = quantize_symmetric(forest.nodes[root].mean, 4, 16).dequantize()
        score = _priority(state, root, mean)
        heapq.heappush(state.heap, (-score, root, mean, score))
    return state


def _signals(state: RouteState) -> dict[str, float | int]:
    records = sorted(((score, index, state.forest.nodes[index]) for _, index, _, score in state.heap), key=lambda row: row[0], reverse=True)
    top8, nodes = records[:8], [row[2] for row in records]
    variances = [float((node.grouped_m2[state.groups] / max(node.count, 1)).max()) for _, _, node in top8 if node.grouped_m2 is not None]
    weakest = min(state.admitted_leaf_priorities) if state.admitted_leaf_priorities else 0.
    best = records[0][0] if records else 0.
    return {"frontier_uncertainty_margin": best - weakest if records else 0., "frontier_uncertainty_margin_normalized": (best - weakest) / (abs(weakest) + 1e-6) if records else 0., "weakest_selected_leaf_priority": weakest, "best_unresolved_frontier_priority": best, "frontier_max_grouped_variance_top8": max(variances, default=0.), "frontier_mean_grouped_variance_top8": statistics.mean(variances) if variances else 0., "unresolved_high_level_nodes_entire_frontier": sum(not node.is_leaf for node in nodes), "max_unresolved_subtree_size_entire_frontier": max((node.count for node in nodes), default=0), "mean_unresolved_subtree_size_entire_frontier": statistics.mean([node.count for node in nodes]) if nodes else 0., "detail_reads": state.detail_reads}


def route_snapshot(state: RouteState) -> dict[str, Any]:
    return {"ids": set(state.chosen), "signals": _signals(state), "active_root_reads": state.active_root_reads, "detail_reads": state.detail_reads, "expanded_internal_nodes": state.expanded_internal_nodes, "variance_scalar_reads": state.variance_scalar_reads, "variance_scalar_products": state.variance_scalar_products, "variance_scalar_adds": state.variance_scalar_adds}


def advance_until_budget(state: RouteState, budget: int) -> dict[str, Any]:
    if budget < len(state.chosen):
        raise ValueError("cannot resume a route to a smaller budget")
    while state.heap and len(state.chosen) < budget:
        _, index, mean, score = heapq.heappop(state.heap)
        node = state.forest.nodes[index]
        if node.is_leaf:
            for token in range(node.start, node.end):
                if token not in state.chosen:
                    state.chosen.add(token); state.admitted_leaf_priorities.append(score)
                    if len(state.chosen) == budget: break
            continue
        assert node.left is not None and node.right is not None and node.detail is not None and node.alpha is not None
        left, right = reconstruct_pair(mean, quantize_symmetric(node.detail, 4, 16).dequantize(), node.alpha)
        state.detail_reads += 1; state.expanded_internal_nodes += 1
        for child, child_mean in ((node.left, left), (node.right, right)):
            child_score = _priority(state, child, child_mean)
            heapq.heappush(state.heap, (-child_score, child, child_mean, child_score))
    if len(state.chosen) != budget:
        raise AssertionError("routing frontier exhausted before requested candidate budget")
    return route_snapshot(state)


def hierarchy_route(forest: StreamingLiftingForest, query: torch.Tensor, budget: int, groups: int, weight: float, reserve: set[int]) -> dict[str, Any]:
    return advance_until_budget(initialize_route(forest, query, groups, weight, reserve), budget)


def traffic(route: dict[str, Any], token_count: int, budget: int, head_dim: int, *, variance: bool, flat: bool = False) -> dict[str, float]:
    spec = traffic_accounting(head_dim)
    roots, details, expanded = route["active_root_reads"], route["detail_reads"], route["expanded_internal_nodes"]
    candidate_reads = token_count if flat else budget
    result: dict[str, float] = {"active_root_k4_bytes": roots * spec.k4_key_bytes, "lifting_detail_k4_bytes": details * spec.k4_key_bytes, "grouped_variance_metadata_bytes": route["variance_scalar_reads"] * 2 if variance else 0, "topology_bytes": expanded * 12, "candidate_k4_bytes": candidate_reads * spec.k4_key_bytes, "authoritative_fp16_k_bytes": budget * spec.fp16_key_bytes, "selected_v_fp16_bytes": budget * spec.fp16_value_bytes}
    result["routing_index_bytes"] = result["active_root_k4_bytes"] + result["lifting_detail_k4_bytes"] + result["grouped_variance_metadata_bytes"] + result["topology_bytes"]
    result["candidate_sketch_bytes"] = result["candidate_k4_bytes"]
    result["authoritative_k_bytes"] = result["authoritative_fp16_k_bytes"]
    result["total_k_bytes"] = result["routing_index_bytes"] + result["candidate_sketch_bytes"] + result["authoritative_k_bytes"]
    # These are first-class quantities rather than a caller-side convention:
    # both dense operands are mechanically FP16 vectors of ``head_dim``.
    result["total_kv_bytes"] = result["total_k_bytes"] + result["selected_v_fp16_bytes"]
    result["total_k_vs_dense_fp16_k"] = result["total_k_bytes"] / (token_count * spec.fp16_key_bytes)
    result["total_kv_vs_dense"] = result["total_kv_bytes"] / (token_count * (spec.fp16_key_bytes + spec.fp16_value_bytes))
    result["total_k_vs_flat_q8k4_plus_rerank"] = result["total_k_bytes"] / (token_count * spec.k4_key_bytes + budget * spec.fp16_key_bytes)
    return {key: float(value) for key, value in result.items()}


def build_forests(keys: torch.Tensor, geometry: ModelGeometry, positions: Iterable[int], *, atom_size: int = 8, groups: tuple[int, ...] = (1, 4)) -> dict[tuple[int, int], StreamingLiftingForest]:
    """Causally build a snapshot per requested position/KV head."""
    positions = tuple(positions)
    if any(p < 0 or p >= geometry.context_length for p in positions):
        raise ValueError("position outside geometry")
    if keys.shape[:2] != (geometry.num_kv_heads, geometry.context_length) or keys.shape[-1] != geometry.head_dim:
        raise ValueError("key tensor does not match geometry")
    result: dict[tuple[int, int], StreamingLiftingForest] = {}
    for kv_head in range(geometry.num_kv_heads):
        forest = StreamingLiftingForest(mode="binary_counter", atom_size=atom_size, window=4, group_counts=groups)
        for start in range(0, geometry.context_length, atom_size):
            forest.append_atom(keys[kv_head, start:start + atom_size])
            end = start + atom_size - 1
            if end in positions: result[end, kv_head] = copy.deepcopy(forest)
    if set(result) != {(p, kv) for p in positions for kv in range(geometry.num_kv_heads)}:
        raise ValueError("positions must align with atom boundaries for causal hierarchy snapshots")
    return result
