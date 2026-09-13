"""Radix-tree summaries and reference hierarchical maximum-inner-product search.

This module deliberately models routing in FP32.  It is a correctness and
traffic-accounting oracle, not an FPGA implementation.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Literal

import torch

Summary = Literal["mean_key", "full_box_bound", "ordered_prefix_box", "ordered_prefix_plus_residual_bound"]
AdaptivePolicy = Literal["descend_immediately", "ambiguity_refine"]


@dataclass(frozen=True)
class Node:
    """A contiguous half-open key-ID interval at one level of the radix tree."""

    id: int
    level: int
    start: int
    end: int
    children: tuple[int, ...]
    parent: int | None

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class SearchAccounting:
    nodes_evaluated_by_level: dict[int, int] = field(default_factory=dict)
    nodes_visited: int = 0
    leaf_tokens_evaluated: int = 0
    metadata_bytes_read: int = 0
    unique_metadata_bytes: int = 0
    repeated_metadata_bytes: int = 0
    leaf_k_bytes_read: int = 0
    coordinate_operations: int = 0
    node_expansions: int = 0
    frontier_width_by_level: dict[int, int] = field(default_factory=dict)
    refinements_16_to_32: int = 0
    refinements_32_to_64: int = 0
    refinements_64_to_128: int = 0
    visited_node_ids: set[int] = field(default_factory=set, repr=False)
    fetched_summary_width: dict[int, int] = field(default_factory=dict, repr=False)
    fetched_residual_widths: set[tuple[int, int]] = field(default_factory=set, repr=False)

    def add_node(self, node_id: int, level: int, dimensions: int, *, residual: bool) -> None:
        self.nodes_evaluated_by_level[level] = self.nodes_evaluated_by_level.get(level, 0) + 1
        if node_id not in self.visited_node_ids:
            self.visited_node_ids.add(node_id)
            self.nodes_visited += 1
        # Prefix min/max summaries are physically progressive: a later wider
        # lookup fetches only the coordinates not already fetched for this node.
        old_width = self.fetched_summary_width.get(node_id, 0)
        new_dimensions = max(0, dimensions - old_width)
        if new_dimensions:
            self.fetched_summary_width[node_id] = dimensions
        coordinate_bytes = 2 * new_dimensions * 4
        coordinate_ops = 2 * new_dimensions
        # Each prefix width has a distinct residual-radius scalar.  It is read
        # at most once, even if a bound is revisited from the priority queue.
        residual_bytes = residual_ops = 0
        residual_key = (node_id, dimensions)
        if residual and dimensions < 128 and residual_key not in self.fetched_residual_widths:
            self.fetched_residual_widths.add(residual_key)
            residual_bytes = residual_ops = 4
        added = coordinate_bytes + residual_bytes
        self.metadata_bytes_read += added
        self.unique_metadata_bytes += added
        self.coordinate_operations += coordinate_ops + residual_ops

    def add_refinement(self, old: int, new: int) -> None:
        if (old, new) == (16, 32):
            self.refinements_16_to_32 += 1
        elif (old, new) == (32, 64):
            self.refinements_32_to_64 += 1
        elif (old, new) == (64, 128):
            self.refinements_64_to_128 += 1


@dataclass(frozen=True)
class SearchResult:
    ids: torch.Tensor
    scores: torch.Tensor
    accounting: SearchAccounting


class HierarchicalIndex:
    """Contiguous radix-``fanout`` tree with min/max and residual summaries."""

    def __init__(self, keys: torch.Tensor, ordering: torch.Tensor | list[int] | None = None, *, fanout: int = 16) -> None:
        if keys.ndim != 2 or keys.shape[0] < 1:
            raise ValueError("keys must be a nonempty [tokens, dimensions] tensor")
        if fanout < 2:
            raise ValueError("fanout must be at least two")
        self.keys = keys.float().contiguous()
        self.token_count, self.dimensions = self.keys.shape
        self.fanout = fanout
        self.ordering = torch.arange(self.dimensions) if ordering is None else torch.as_tensor(ordering, dtype=torch.long)
        if self.ordering.ndim != 1 or self.ordering.numel() != self.dimensions or not torch.equal(
            self.ordering.sort().values.cpu(), torch.arange(self.dimensions)
        ):
            raise ValueError("ordering must be a permutation of key coordinates")
        self.ordered_keys = self.keys[:, self.ordering]
        self.nodes, self.levels, self.root_id = self._make_tree()
        self._minimum = torch.empty((len(self.nodes), self.dimensions), dtype=torch.float32)
        self._maximum = torch.empty_like(self._minimum)
        self._mean = torch.empty_like(self._minimum)
        self._radii: dict[int, torch.Tensor] = {}
        self._summarize()

    def _make_tree(self) -> tuple[list[Node], list[list[int]], int]:
        nodes: list[Node] = []
        levels: list[list[int]] = []
        leaf_ids = []
        for token in range(self.token_count):
            leaf_ids.append(len(nodes))
            nodes.append(Node(len(nodes), 0, token, token + 1, (), None))
        levels.append(leaf_ids)
        child_ids = leaf_ids
        level = 1
        while len(child_ids) > 1:
            current: list[int] = []
            for offset in range(0, len(child_ids), self.fanout):
                children = tuple(child_ids[offset : offset + self.fanout])
                node_id = len(nodes)
                nodes.append(Node(node_id, level, nodes[children[0]].start, nodes[children[-1]].end, children, None))
                for child in children:
                    old = nodes[child]
                    nodes[child] = Node(old.id, old.level, old.start, old.end, old.children, node_id)
                current.append(node_id)
            levels.append(current)
            child_ids = current
            level += 1
        return nodes, levels, child_ids[0]

    def _summarize(self) -> None:
        for node in self.nodes:
            values = self.ordered_keys[node.start : node.end]
            self._minimum[node.id] = values.amin(0)
            self._maximum[node.id] = values.amax(0)
            self._mean[node.id] = values.mean(0)
        for width in (16, 32, 64):
            if width < self.dimensions:
                self._radii[width] = torch.stack([
                    self.ordered_keys[n.start : n.end, width:].norm(dim=1).amax() for n in self.nodes
                ])

    @property
    def max_level(self) -> int:
        return len(self.levels) - 1

    def node_bound(self, node_id: int, query: torch.Tensor, *, summary: Summary = "ordered_prefix_plus_residual_bound", dimensions: int | None = None) -> float:
        """Return a score estimate/bound. Only box-plus-residual is conservative."""
        q = query.float()[self.ordering]
        width = self.dimensions if dimensions is None else dimensions
        if not 1 <= width <= self.dimensions:
            raise ValueError("dimensions must lie within key dimensionality")
        if summary == "mean_key":
            return float(torch.dot(q, self._mean[node_id]))
        maximum, minimum = self._maximum[node_id, :width], self._minimum[node_id, :width]
        prefix = torch.where(q[:width] >= 0, q[:width] * maximum, q[:width] * minimum).sum()
        if summary in ("full_box_bound", "ordered_prefix_box"):
            return float(prefix)
        if summary != "ordered_prefix_plus_residual_bound":
            raise ValueError(f"unknown summary: {summary}")
        if width == self.dimensions:
            return float(prefix)
        if width not in self._radii:
            # This also supports dimensions other than the published schedules.
            radius = self.ordered_keys[self.nodes[node_id].start : self.nodes[node_id].end, width:].norm(dim=1).amax()
        else:
            radius = self._radii[width][node_id]
        return float(prefix + q[width:].norm() * radius)

    def validate_bounds(self, queries: torch.Tensor, *, dimensions: int = 128, summary: Summary = "ordered_prefix_plus_residual_bound", tolerance: float = 1e-4) -> int:
        """Count node/query bound violations against every constituent exact score."""
        violations = 0
        for query in queries.float():
            exact = self.keys @ query
            for node in self.nodes:
                if self.node_bound(node.id, query, summary=summary, dimensions=dimensions) + tolerance < exact[node.start : node.end].max().item():
                    violations += 1
        return violations

    def _bound(self, node_id: int, query: torch.Tensor, summary: Summary, width: int, accounting: SearchAccounting) -> float:
        accounting.add_node(node_id, self.nodes[node_id].level, width, residual=summary == "ordered_prefix_plus_residual_bound")
        return self.node_bound(node_id, query, summary=summary, dimensions=width)

    def beam_search(self, query: torch.Tensor, *, beam_width: int = 16, k: int = 8, summary: Summary = "ordered_prefix_plus_residual_bound", schedule: dict[int, int] | None = None) -> SearchResult:
        """Deterministic level-synchronous beam descent followed by exact leaf scoring."""
        if beam_width < 1 or k < 1:
            raise ValueError("beam_width and k must be positive")
        accounting = SearchAccounting()
        frontier = [self.root_id]
        for level in range(self.max_level, 0, -1):
            candidates = [child for parent in frontier for child in self.nodes[parent].children]
            width = (schedule or {}).get(level - 1, self.dimensions)
            ranked = [(self._bound(node, query, summary, width, accounting), node) for node in candidates]
            ranked.sort(key=lambda pair: (-pair[0], self.nodes[pair[1]].start))
            frontier = [node for _, node in ranked[:beam_width]]
            accounting.frontier_width_by_level[level - 1] = len(frontier)
            accounting.node_expansions += len(candidates)
        ids = torch.tensor(sorted(self.nodes[node].start for node in frontier), dtype=torch.long)
        scores = self.keys[ids] @ query.float()
        accounting.leaf_tokens_evaluated = ids.numel()
        accounting.leaf_k_bytes_read = ids.numel() * self.dimensions * 4
        count = min(k, ids.numel())
        order = sorted(range(ids.numel()), key=lambda i: (-float(scores[i]), int(ids[i])))[:count]
        return SearchResult(ids[order], scores[order], accounting)

    def best_first_search(
        self,
        query: torch.Tensor,
        *,
        k: int = 8,
        summary: Summary = "ordered_prefix_plus_residual_bound",
        schedule: dict[int, int] | None = None,
        adaptive: bool = False,
        adaptive_policy: AdaptivePolicy = "ambiguity_refine",
        ambiguity_margin: float = 0.0,
    ) -> SearchResult:
        """Exact branch-and-bound for conservative summaries; mean/prefix forms are heuristic."""
        if adaptive_policy not in ("descend_immediately", "ambiguity_refine"):
            raise ValueError(f"unknown adaptive policy: {adaptive_policy}")
        if ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be nonnegative")
        accounting = SearchAccounting()
        heap: list[tuple[float, int, int, int]] = []
        default_adaptive_schedule = {0: 16, 1: 32, 2: 64}
        level_width = lambda level: (
            (schedule if schedule is not None else default_adaptive_schedule).get(level, self.dimensions)
            if adaptive
            else (schedule or {}).get(level, self.dimensions)
        )
        # The level schedule is the starting (minimum) width for adaptive
        # bounds, not an instruction to fully refine before descending.
        initial = level_width(self.max_level)
        heapq.heappush(heap, (-self._bound(self.root_id, query, summary, initial, accounting), self.nodes[self.root_id].start, self.root_id, initial))
        exact: list[tuple[float, int]] = []
        while heap:
            negative, _, node_id, width = heapq.heappop(heap)
            bound = -negative
            threshold = exact[0][0] if len(exact) >= k else float("-inf")
            if bound <= threshold:
                continue
            should_refine = (
                adaptive
                and width < self.dimensions
                and len(exact) >= k
                and adaptive_policy == "ambiguity_refine"
                and bound - threshold <= ambiguity_margin
            )
            if should_refine:
                next_width = next(value for value in (16, 32, 64, self.dimensions) if value > width)
                accounting.add_refinement(width, next_width)
                refined = self._bound(node_id, query, summary, next_width, accounting)
                # Re-evaluate only near the exact threshold, where a tighter
                # conservative bound can plausibly avoid an expansion.
                heapq.heappush(heap, (-refined, self.nodes[node_id].start, node_id, next_width))
                continue
            node = self.nodes[node_id]
            if node.is_leaf:
                score = float(torch.dot(self.keys[node.start], query.float()))
                if len(exact) < k:
                    heapq.heappush(exact, (score, node.start))
                elif score > exact[0][0]:
                    heapq.heapreplace(exact, (score, node.start))
                accounting.leaf_tokens_evaluated += 1
                accounting.leaf_k_bytes_read += self.dimensions * 4
                continue
            accounting.node_expansions += 1
            for child in node.children:
                width = level_width(self.nodes[child].level)
                child_bound = self._bound(child, query, summary, width, accounting)
                heapq.heappush(heap, (-child_bound, self.nodes[child].start, child, width))
        exact.sort(key=lambda pair: (-pair[0], pair[1]))
        return SearchResult(torch.tensor([i for _, i in exact]), torch.tensor([s for s, _ in exact]), accounting)


def accounting_ratios(accounting: SearchAccounting, token_count: int, dimensions: int = 128) -> dict[str, float | int]:
    """Comparable FP32-reference traffic ratios (dense FP16 K scan is 2 bytes/value)."""
    dense_fp16 = token_count * dimensions * 2
    flat_q8k4 = token_count * 80  # existing group-16 Q8xK4 key format
    total = accounting.metadata_bytes_read + accounting.leaf_k_bytes_read
    return {
        "metadata_bytes_read": accounting.metadata_bytes_read,
        "unique_metadata_bytes": accounting.unique_metadata_bytes,
        "repeated_redundant_metadata_bytes": accounting.repeated_metadata_bytes,
        "leaf_k_bytes_read": accounting.leaf_k_bytes_read,
        "estimated_total_routing_bytes": total,
        "coordinate_operations": accounting.coordinate_operations + accounting.leaf_tokens_evaluated * dimensions,
        "fraction_leaf_k_read": accounting.leaf_tokens_evaluated / token_count,
        "ratio_vs_dense_fp16_k_scan": total / dense_fp16,
        "ratio_vs_flat_cascadekv_scan": total / flat_q8k4,
    }
