"""Radix-tree summaries and reference hierarchical maximum-inner-product search.

This module deliberately models routing in FP32.  It is a correctness and
traffic-accounting oracle, not an FPGA implementation.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Literal

import torch

from cascadekv.fixed_point import q8k4_rtl_error_cushion, q8k4_rtl_score
from cascadekv.quantize import QuantizedTensor, quantize_symmetric

Summary = Literal[
    "mean_key", "full_box_bound", "ordered_prefix_box", "ordered_prefix_plus_residual_bound",
    "support_set_global", "support_set_per_prototype", "support_set_score",
    "support_set_k4_global", "support_set_k4_per_prototype",
    "support_set_q8k4_global", "support_set_q8k4_per_prototype",
    "support_set_fixed_q8k4",
]
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
    fetched_support_width: dict[tuple[int, int], int] = field(default_factory=dict, repr=False)
    fetched_support_scalars: set[tuple[int, int, int, bool]] = field(default_factory=set, repr=False)
    projected_support_metadata_bytes: int = 0

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

    def add_leaf_bound(self, node_id: int, level: int) -> None:
        """Record an exact singleton bound without charging node metadata."""
        self.nodes_evaluated_by_level[level] = self.nodes_evaluated_by_level.get(level, 0) + 1
        if node_id not in self.visited_node_ids:
            self.visited_node_ids.add(node_id)
            self.nodes_visited += 1

    def add_support_node(self, node_id: int, level: int, dimensions: int, representatives: int, *,
                         per_prototype: bool, tail: bool) -> None:
        """Charge progressive FP32 representatives and projected group-16 K4 sketches."""
        self.nodes_evaluated_by_level[level] = self.nodes_evaluated_by_level.get(level, 0) + 1
        if node_id not in self.visited_node_ids:
            self.visited_node_ids.add(node_id)
            self.nodes_visited += 1
        key = (node_id, representatives)
        old_width = self.fetched_support_width.get(key, 0)
        added_dimensions = max(0, dimensions - old_width)
        if added_dimensions:
            self.fetched_support_width[key] = dimensions
        coordinate_bytes = representatives * added_dimensions * 4
        # A 16-D group of a K representative is 8 bytes codes + 2 bytes scale.
        sketch_bytes = representatives * ((added_dimensions + 15) // 16) * 10
        scalar_key = (node_id, representatives, dimensions, per_prototype)
        scalar_count = representatives if per_prototype else 1
        # Prefix bounds also need one tail-norm scalar per node.
        if tail:
            scalar_count += 1
        scalar_bytes = 0 if scalar_key in self.fetched_support_scalars else scalar_count * 4
        self.fetched_support_scalars.add(scalar_key)
        added = coordinate_bytes + scalar_bytes
        self.metadata_bytes_read += added
        self.unique_metadata_bytes += added
        self.projected_support_metadata_bytes += sketch_bytes + scalar_bytes
        self.coordinate_operations += representatives * added_dimensions + scalar_count


@dataclass(frozen=True)
class SearchResult:
    ids: torch.Tensor
    scores: torch.Tensor
    accounting: SearchAccounting


class HierarchicalIndex:
    """Contiguous radix-``fanout`` tree with min/max and residual summaries."""

    def __init__(
        self, keys: torch.Tensor, ordering: torch.Tensor | list[int] | None = None, *, fanout: int = 16,
        support_set_sizes: tuple[int, ...] = (),
    ) -> None:
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
        self.support_set_sizes = tuple(sorted(set(support_set_sizes)))
        if any(size < 1 for size in self.support_set_sizes):
            raise ValueError("support-set sizes must be positive")
        # Original token IDs and all radii are retained for auditing.  Entries
        # are variable-length for nodes containing fewer than P keys.
        self._support_ids: dict[int, list[torch.Tensor]] = {}
        self._support_global: dict[tuple[int, int], torch.Tensor] = {}
        self._support_per_prototype: dict[tuple[int, int], list[torch.Tensor]] = {}
        self._support_tail: dict[tuple[int, int], torch.Tensor] = {}
        # These are deliberately real K4 encodings, rather than traffic-only
        # projections.  Each entry is variable-sized for degenerate nodes.
        self._support_k4: dict[tuple[int, int], QuantizedTensor] = {}
        self._support_k4_rhat: dict[tuple[int, int], torch.Tensor] = {}
        self._support_k4_global: dict[tuple[int, int], torch.Tensor] = {}
        self._support_k4_per_prototype: dict[tuple[int, int], list[torch.Tensor]] = {}
        self._key_norm_max = torch.empty(len(self.nodes), dtype=torch.float32)
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
            self._key_norm_max[node.id] = values.norm(dim=1).amax()
        for width in (16, 32, 64):
            if width < self.dimensions:
                self._radii[width] = torch.stack([
                    self.ordered_keys[n.start : n.end, width:].norm(dim=1).amax() for n in self.nodes
                ])
        for representatives in self.support_set_sizes:
            selections: list[torch.Tensor] = []
            for node in self.nodes:
                values = self.ordered_keys[node.start : node.end]
                # Nearest to the node mean, with original token ID as the tie-break.
                mean_distances = (values - values.mean(0)).square().sum(1)
                first = int(torch.argmin(mean_distances))
                selected = [first]
                min_distances = (values - values[first]).square().sum(1)
                min_distances[first] = float("-inf")
                while len(selected) < min(representatives, values.shape[0]):
                    # argmax is deterministic and returns the first local/original ID on ties.
                    candidate = int(torch.argmax(min_distances))
                    selected.append(candidate)
                    min_distances = torch.minimum(
                        min_distances, (values - values[candidate]).square().sum(1)
                    )
                    min_distances[candidate] = float("-inf")
                selections.append(torch.tensor([node.start + item for item in selected], dtype=torch.long))
            self._support_ids[representatives] = selections
            for width in (16, 32, 64, self.dimensions):
                global_radii, assigned_radii, tails = [], [], []
                for node, ids in zip(self.nodes, selections, strict=True):
                    values = self.ordered_keys[node.start : node.end]
                    if node.is_leaf:
                        # Leaf scores are evaluated exactly; no routing summary
                        # is stored or fetched for singleton nodes.
                        global_radii.append(torch.zeros(()))
                        assigned_radii.append(torch.zeros(1))
                        if width < self.dimensions:
                            tails.append(values[:, width:].norm(dim=1).amax())
                        continue
                    reps = self.ordered_keys[ids, :width]
                    distances = torch.cdist(values[:, :width], reps)
                    nearest_distance, assignment = distances.min(1)
                    global_radii.append(nearest_distance.max())
                    assigned_radii.append(torch.stack([
                        nearest_distance[assignment == prototype].amax()
                        if torch.any(assignment == prototype) else torch.zeros(())
                        for prototype in range(ids.numel())
                    ]))
                    if width < self.dimensions:
                        tails.append(values[:, width:].norm(dim=1).amax())
                self._support_global[representatives, width] = torch.stack(global_radii)
                # Kept as a list because degenerate nodes can have fewer than P representatives.
                self._support_per_prototype[representatives, width] = assigned_radii
                if width < self.dimensions:
                    self._support_tail[representatives, width] = torch.stack(tails)
            # Routing hardware stores quantized selected representatives.  The
            # assignment/radius is recomputed against their dequantized form
            # and always covers *original* FP keys.
            for node, ids in zip(self.nodes, selections, strict=True):
                reps = self.ordered_keys[ids]
                encoded = quantize_symmetric(reps, 4, 16)
                rhat = encoded.dequantize()
                self._support_k4[representatives, node.id] = encoded
                self._support_k4_rhat[representatives, node.id] = rhat
                values = self.ordered_keys[node.start : node.end]
                distances = torch.cdist(values, rhat)
                nearest, assignment = distances.min(1)
                self._support_k4_global[representatives, node.id] = nearest.max()
                self._support_k4_per_prototype[representatives, node.id] = [
                    nearest[assignment == p].amax() if torch.any(assignment == p) else torch.zeros(())
                    for p in range(rhat.shape[0])
                ]

    @property
    def max_level(self) -> int:
        return len(self.levels) - 1

    def support_ids(self, node_id: int, representatives: int) -> torch.Tensor:
        """Original constituent token IDs chosen by deterministic FPS."""
        return self._support_ids[representatives][node_id].clone()

    def quantized_support(self, node_id: int, representatives: int) -> QuantizedTensor:
        """Frozen group-16 K4 encoding physically stored for this node."""
        return self._support_k4[representatives, node_id]

    def quantized_support_reconstruction(self, node_id: int, representatives: int) -> torch.Tensor:
        return self._support_k4_rhat[representatives, node_id].clone()

    def _support_bound(self, node_id: int, query: torch.Tensor, width: int, representatives: int,
                       *, per_prototype: bool, conservative: bool) -> float:
        if representatives not in self._support_ids:
            raise ValueError(f"support set P={representatives} was not constructed")
        q = query.float()[self.ordering]
        ids = self._support_ids[representatives][node_id]
        reps = self.ordered_keys[ids, :width]
        scores = reps @ q[:width]
        if not conservative:
            return float(scores.max())
        qnorm = q[:width].norm()
        if per_prototype:
            radii = self._support_per_prototype[representatives, width][node_id]
            prefix = (scores + qnorm * radii).max()
        else:
            prefix = scores.max() + qnorm * self._support_global[representatives, width][node_id]
        if width < self.dimensions:
            prefix = prefix + q[width:].norm() * self._support_tail[representatives, width][node_id]
        return float(prefix)

    def _quantized_support_bound(self, node_id: int, query: torch.Tensor, representatives: int, *,
                                 per_prototype: bool, quantized_query: bool, fixed: bool) -> float:
        """Whole-vector K4 support bound against original FP Q/K scores.

        The Q8 variant adds ||q-qhat|| Kmax, so it remains conservative even
        though its prototype dot uses the deployed dequantized Q8 values.
        Fixed arithmetic uses the frozen integer RTL path.  Its per-prototype
        deterministic scale/saturation cushion makes this form conservative
        with respect to the dequantized Q8xK4 prototype dot.
        """
        q = query.float()[self.ordering]
        rhat = self._support_k4_rhat[representatives, node_id]
        if quantized_query:
            qhat = quantize_symmetric(q, 8, 16).dequantize()
            dot_q = qhat
            cushion = (q - qhat).norm() * self._key_norm_max[node_id]
        else:
            dot_q, cushion = q, torch.zeros(())
        scores = rhat @ dot_q
        if fixed:
            q8 = quantize_symmetric(q, 8, 16)
            encoded = self._support_k4[representatives, node_id]
            scores = q8k4_rtl_score(q8, encoded)
            # This is a proven, input-specific upper bound for the arithmetic
            # difference, rather than a fitted empirical epsilon.
            arithmetic_cushion = q8k4_rtl_error_cushion(q8, encoded)
        else:
            arithmetic_cushion = torch.zeros_like(scores)
        if per_prototype:
            radii = torch.stack(self._support_k4_per_prototype[representatives, node_id])
            value = (scores + arithmetic_cushion + dot_q.norm() * radii).max()
        else:
            value = (scores + arithmetic_cushion).max() + dot_q.norm() * self._support_k4_global[representatives, node_id]
        return float(value + cushion)

    def node_bound(self, node_id: int, query: torch.Tensor, *, summary: Summary = "ordered_prefix_plus_residual_bound", dimensions: int | None = None, representatives: int = 1) -> float:
        """Return a score estimate/bound. Only box-plus-residual is conservative."""
        q = query.float()[self.ordering]
        width = self.dimensions if dimensions is None else dimensions
        if not 1 <= width <= self.dimensions:
            raise ValueError("dimensions must lie within key dimensionality")
        if summary in ("support_set_global", "support_set_per_prototype", "support_set_score"):
            return self._support_bound(node_id, query, width, representatives,
                                       per_prototype=summary == "support_set_per_prototype",
                                       conservative=summary != "support_set_score")
        if summary in ("support_set_k4_global", "support_set_k4_per_prototype", "support_set_q8k4_global", "support_set_q8k4_per_prototype", "support_set_fixed_q8k4"):
            if width != self.dimensions:
                raise ValueError("quantized support bounds are whole-vector only")
            return self._quantized_support_bound(
                node_id, query, representatives,
                per_prototype=summary in ("support_set_k4_per_prototype", "support_set_q8k4_per_prototype"),
                quantized_query=summary in ("support_set_q8k4_global", "support_set_q8k4_per_prototype", "support_set_fixed_q8k4"),
                fixed=summary == "support_set_fixed_q8k4",
            )
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

    def validate_bounds(self, queries: torch.Tensor, *, dimensions: int = 128, summary: Summary = "ordered_prefix_plus_residual_bound", representatives: int = 1, tolerance: float = 1e-4) -> int:
        """Count node/query bound violations against every constituent exact score."""
        violations = 0
        for query in queries.float():
            exact = self.keys @ query
            for node in self.nodes:
                if self.node_bound(node.id, query, summary=summary, dimensions=dimensions, representatives=representatives) + tolerance < exact[node.start : node.end].max().item():
                    violations += 1
        return violations

    def _bound(self, node_id: int, query: torch.Tensor, summary: Summary, width: int, accounting: SearchAccounting,
               representatives: int = 1) -> float:
        if self.nodes[node_id].is_leaf and summary in (
            "support_set_global", "support_set_per_prototype", "support_set_score"
            , "support_set_k4_global", "support_set_k4_per_prototype", "support_set_q8k4_global", "support_set_q8k4_per_prototype", "support_set_fixed_q8k4"
        ):
            accounting.add_leaf_bound(node_id, self.nodes[node_id].level)
            return float(torch.dot(self.keys[self.nodes[node_id].start], query.float()))
        if summary in ("support_set_global", "support_set_per_prototype", "support_set_score", "support_set_k4_global", "support_set_k4_per_prototype", "support_set_q8k4_global", "support_set_q8k4_per_prototype", "support_set_fixed_q8k4"):
            accounting.add_support_node(
                node_id, self.nodes[node_id].level, width, representatives,
                per_prototype=summary in ("support_set_per_prototype", "support_set_k4_per_prototype", "support_set_q8k4_per_prototype"), tail=width < self.dimensions,
            )
        else:
            accounting.add_node(node_id, self.nodes[node_id].level, width, residual=summary == "ordered_prefix_plus_residual_bound")
        return self.node_bound(node_id, query, summary=summary, dimensions=width, representatives=representatives)

    def beam_search(self, query: torch.Tensor, *, beam_width: int = 16, k: int = 8, summary: Summary = "ordered_prefix_plus_residual_bound", schedule: dict[int, int] | None = None, representatives: int = 1) -> SearchResult:
        """Deterministic level-synchronous beam descent followed by exact leaf scoring."""
        if beam_width < 1 or k < 1:
            raise ValueError("beam_width and k must be positive")
        accounting = SearchAccounting()
        frontier = [self.root_id]
        for level in range(self.max_level, 0, -1):
            candidates = [child for parent in frontier for child in self.nodes[parent].children]
            width = (schedule or {}).get(level - 1, self.dimensions)
            ranked = [(self._bound(node, query, summary, width, accounting, representatives), node) for node in candidates]
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
        representatives: int = 1,
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
        heapq.heappush(heap, (-self._bound(self.root_id, query, summary, initial, accounting, representatives), self.nodes[self.root_id].start, self.root_id, initial))
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
                refined = self._bound(node_id, query, summary, next_width, accounting, representatives)
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
                child_bound = self._bound(child, query, summary, width, accounting, representatives)
                heapq.heappush(heap, (-child_bound, self.nodes[child].start, child, width))
        exact.sort(key=lambda pair: (-pair[0], pair[1]))
        return SearchResult(torch.tensor([i for _, i in exact]), torch.tensor([s for s, _ in exact]), accounting)


def accounting_ratios(accounting: SearchAccounting, token_count: int, dimensions: int = 128) -> dict[str, float | int]:
    """Comparable FP32-reference traffic ratios (dense FP16 K scan is 2 bytes/value)."""
    dense_fp16 = token_count * dimensions * 2
    flat_q8k4 = token_count * 80  # existing group-16 Q8xK4 key format
    total = accounting.metadata_bytes_read + accounting.leaf_k_bytes_read
    projected_total = accounting.projected_support_metadata_bytes + accounting.leaf_k_bytes_read
    return {
        "metadata_bytes_read": accounting.metadata_bytes_read,
        "unique_metadata_bytes": accounting.unique_metadata_bytes,
        "repeated_redundant_metadata_bytes": accounting.repeated_metadata_bytes,
        "leaf_k_bytes_read": accounting.leaf_k_bytes_read,
        "estimated_total_routing_bytes": total,
        "projected_q8k4_support_metadata_bytes": accounting.projected_support_metadata_bytes,
        "projected_total_bytes": projected_total,
        "projected_total_bytes_vs_dense_fp16": projected_total / dense_fp16,
        "coordinate_operations": accounting.coordinate_operations + accounting.leaf_tokens_evaluated * dimensions,
        "fraction_leaf_k_read": accounting.leaf_tokens_evaluated / token_count,
        "ratio_vs_dense_fp16_k_scan": total / dense_fp16,
        "ratio_vs_flat_cascadekv_scan": total / flat_q8k4,
    }


def radix_internal_node_count(token_count: int, fanout: int = 16) -> int:
    """Count non-leaf nodes for a contiguous, partially-filled radix tree."""
    if token_count < 1 or fanout < 2:
        raise ValueError("token_count must be positive and fanout at least two")
    count, total = token_count, 0
    while count > 1:
        count = (count + fanout - 1) // fanout
        total += count
    return total


def support_index_storage(token_count: int, representatives: int, *, dimensions: int = 128,
                          fanout: int = 16, radius_bytes: int = 4,
                          include_kmax: bool = True) -> dict[str, float | int]:
    """Stored-summary bytes, excluding singleton leaves (which have no summary)."""
    nodes = radix_internal_node_count(token_count, fanout)
    # K4 codes plus one FP16 scale for every group: 80 bytes at D=128.
    support = nodes * representatives * (dimensions // 2 + dimensions // 16 * 2)
    radii = nodes * representatives * radius_bytes
    kmax = nodes * radius_bytes if include_kmax else 0
    total = support + radii + kmax
    return {
        "internal_node_count": nodes, "support_vector_bytes": support,
        "radius_metadata_bytes": radii, "kmax_metadata_bytes": kmax,
        "total_index_bytes": total, "bytes_per_leaf_token": total / token_count,
        "percent_of_q8k4_leaf_sketch": 100 * total / (token_count * 80),
        "percent_of_fp16_authoritative_k": 100 * total / (token_count * dimensions * 2),
    }


def bottom_up_radius(parent: torch.Tensor, child_prototypes: torch.Tensor,
                     child_radii: torch.Tensor) -> torch.Tensor:
    """Triangle-inequality radius from child summary balls, no leaf rescan."""
    if child_prototypes.ndim != 2 or child_radii.ndim != 1 or child_prototypes.shape[0] != child_radii.numel():
        raise ValueError("child prototypes [C,D] and radii [C] must align")
    return ((child_prototypes - parent.float()).norm(dim=1) + child_radii.float()).amax()


def radix_completion_levels(token_count: int, fanout: int = 16) -> tuple[int, ...]:
    """Completed radix-summary levels after appending token number ``token_count``.

    Level one is a 16-token summary, level two a 256-token summary, etc.  A
    partial tail remains an ordinary leaf run and is intentionally not claimed
    as finalized online metadata.
    """
    if token_count < 0 or fanout < 2:
        raise ValueError("token_count must be nonnegative and fanout at least two")
    levels: list[int] = []
    block = fanout
    level = 1
    while token_count and token_count % block == 0:
        levels.append(level)
        block *= fanout
        level += 1
    return tuple(levels)


def radix_amortized_summary_updates(fanout: int = 16) -> float:
    """Finalized-summary writes per appended token, including all carry levels."""
    if fanout < 2:
        raise ValueError("fanout must be at least two")
    return 1.0 / (fanout - 1)


def bottom_up_p1_summaries(tree: HierarchicalIndex) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """P1 online radii made solely from child P1 balls, indexed by node ID.

    Parent representatives are selected from child representatives nearest to
    their centroid; each radius is the triangle-inequality envelope.  This is
    the precise no-old-leaf-rescan construction experiment.
    """
    if 1 not in tree._support_ids:
        raise ValueError("tree must have P1 support summaries")
    prototypes: dict[int, torch.Tensor] = {}
    radii: dict[int, torch.Tensor] = {}
    for level, node_ids in enumerate(tree.levels):
        for node_id in node_ids:
            node = tree.nodes[node_id]
            if level == 0:
                prototypes[node_id] = tree.ordered_keys[node.start]
                radii[node_id] = torch.zeros((), dtype=torch.float32)
                continue
            child_prototypes = torch.stack([prototypes[child] for child in node.children])
            center = child_prototypes.mean(0)
            choice = int(((child_prototypes - center).square().sum(1)).argmin())
            parent = child_prototypes[choice]
            prototypes[node_id] = parent
            radii[node_id] = bottom_up_radius(
                parent, child_prototypes, torch.stack([radii[child] for child in node.children])
            )
    return {node_id: (prototypes[node_id], radii[node_id]) for node_id in prototypes}


def bottom_up_p1_radii(tree: HierarchicalIndex) -> dict[int, torch.Tensor]:
    """Compatibility view of :func:`bottom_up_p1_summaries` containing radii."""
    return {node_id: radius for node_id, (_, radius) in bottom_up_p1_summaries(tree).items()}
