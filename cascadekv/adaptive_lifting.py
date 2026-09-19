"""Content-adaptive, causal lifting trees over contiguous token atoms.

This module intentionally contains no hardware policy: it is the FP32/K4
software reference used to decide whether an adaptive hierarchy is worthwhile.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Literal

import torch

from cascadekv.quantize import quantize_symmetric

MetricName = Literal[
    "k_l2", "query_cov_diag", "query_cov_lowrank", "ward_k_l2",
    "ward_query_cov_diag", "ward_query_cov_lowrank", "hybrid",
]
RepresentationName = Literal["lifting", "left", "right", "smaller", "two_centroid", "full_centroid"]


@dataclass
class LiftingNode:
    """A contiguous region.  ``detail`` is ``right.mean - left.mean``."""

    index: int
    start: int
    end: int
    count: int
    mean: torch.Tensor
    left: int | None = None
    right: int | None = None
    alpha: float | None = None
    detail: torch.Tensor | None = None
    merge_cost: float | None = None
    # Grouped second central moments are immutable node metadata.  A forest
    # may retain more than one group count (the gate needs G4 and G8) without
    # changing the lifting representation itself.
    grouped_m2: dict[int, torch.Tensor] | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass
class AdaptiveLiftingTree:
    nodes: list[LiftingNode]
    root: int
    atom_size: int = 16
    metric: str = "balanced_positional"
    balance_ratio: float | None = None
    balance_relaxations: int = 0
    projection: torch.Tensor | None = None
    metric_scales: dict[str, float] | None = None

    @property
    def leaves(self) -> list[LiftingNode]:
        return [node for node in self.nodes if node.is_leaf]

    @property
    def internal_nodes(self) -> list[LiftingNode]:
        return [node for node in self.nodes if not node.is_leaf]

    def reconstruct_means(self, *, k4: bool = False) -> dict[int, torch.Tensor]:
        """Reconstruct atom means solely from root scale and lifting details."""
        root = self.nodes[self.root]
        value = _stored_vector(root.mean, k4)
        out: dict[int, torch.Tensor] = {}

        def visit(index: int, scaling: torch.Tensor) -> None:
            node = self.nodes[index]
            if node.is_leaf:
                out[index] = scaling
                return
            assert node.left is not None and node.right is not None
            assert node.alpha is not None and node.detail is not None
            detail = _stored_vector(node.detail, k4)
            visit(node.left, scaling - node.alpha * detail)
            visit(node.right, scaling + (1.0 - node.alpha) * detail)

        visit(self.root, value)
        return out

    def reconstruction_error(self) -> float:
        reconstructed = self.reconstruct_means()
        return max((reconstructed[n.index] - n.mean).abs().max().item() for n in self.leaves)

    def leaf_depths(self) -> list[int]:
        depths: list[int] = []

        def walk(index: int, depth: int) -> None:
            node = self.nodes[index]
            if node.is_leaf:
                depths.append(depth)
            else:
                assert node.left is not None and node.right is not None
                walk(node.left, depth + 1)
                walk(node.right, depth + 1)

        walk(self.root, 0)
        return depths


@dataclass
class TreeRepresentation:
    """A query-time representation of one fixed ``AdaptiveLiftingTree``.

    Vectors here are deliberately the sole routing source: callers must not
    fall back to node means after K4 is requested.  That makes ablations of
    numerically different stored quantities honest.
    """

    tree: AdaptiveLiftingTree
    name: RepresentationName
    root: torch.Tensor
    vectors: dict[int, tuple[torch.Tensor, ...]]
    smaller_is_left: dict[int, bool]

    def child_means(self, index: int, parent: torch.Tensor, *, k4: bool) -> tuple[torch.Tensor, torch.Tensor]:
        node = self.tree.nodes[index]
        assert node.alpha is not None
        values = tuple(_stored_vector(value, k4) for value in self.vectors[index])
        alpha = node.alpha
        if self.name == "lifting":
            return reconstruct_pair(parent, values[0], alpha)
        if self.name == "left":
            left = values[0]
            return left, (parent - (1.0 - alpha) * left) / alpha
        if self.name == "right":
            right = values[0]
            return (parent - alpha * right) / (1.0 - alpha), right
        if self.name == "smaller":
            if self.smaller_is_left[index]:
                left = values[0]
                return left, (parent - (1.0 - alpha) * left) / alpha
            right = values[0]
            return (parent - alpha * right) / (1.0 - alpha), right
        # Both centroid variants read the two actual child centroid vectors.
        return values[0], values[1]

    @property
    def vectors_per_expansion(self) -> int:
        return 1 if self.name in ("lifting", "left", "right", "smaller") else 2


def build_tree_representation(tree: AdaptiveLiftingTree, name: RepresentationName) -> TreeRepresentation:
    """Materialize one stored-vector scheme without changing topology."""
    root = tree.nodes[tree.root].mean
    vectors: dict[int, tuple[torch.Tensor, ...]] = {}
    smaller: dict[int, bool] = {}
    for node in tree.internal_nodes:
        assert node.left is not None and node.right is not None and node.detail is not None
        left, right = tree.nodes[node.left].mean, tree.nodes[node.right].mean
        if name == "lifting":
            vectors[node.index] = (node.detail,)
        elif name == "left":
            vectors[node.index] = (left,)
        elif name == "right":
            vectors[node.index] = (right,)
        elif name == "smaller":
            is_left = tree.nodes[node.left].count <= tree.nodes[node.right].count
            smaller[node.index] = is_left
            vectors[node.index] = (left if is_left else right,)
        elif name in ("two_centroid", "full_centroid"):
            vectors[node.index] = (left, right)
        else:  # defensive for programmatic callers beyond the Literal type.
            raise ValueError(f"unknown representation: {name}")
    return TreeRepresentation(tree, name, root, vectors, smaller)


def recursive_child_score_errors(
    representation: TreeRepresentation, query: torch.Tensor, *, k4: bool = False
) -> list[float]:
    """Errors for child Q.K scores reached through the actual stored traversal.

    In particular, a non-root parent is the recursively reconstructed value,
    never the original FP32 node mean.  This is the metric used by the layer
    representation ablation and catches accumulated K4 error.
    """
    tree, q = representation.tree, query.float()
    errors: list[float] = []

    def visit(index: int, parent: torch.Tensor) -> None:
        node = tree.nodes[index]
        if node.is_leaf:
            return
        assert node.left is not None and node.right is not None
        left, right = representation.child_means(index, parent, k4=k4)
        for child, reconstructed in ((node.left, left), (node.right, right)):
            truth = tree.nodes[child].mean
            errors.append(abs(float(q @ reconstructed) - float(q @ truth)))
            visit(child, reconstructed)

    visit(tree.root, _stored_vector(representation.root, k4))
    return errors


def representation_reconstruction_error(representation: TreeRepresentation, *, k4: bool = False) -> float:
    """Maximum recursive vector reconstruction error across all descendants."""
    tree = representation.tree
    maximum = 0.0

    def visit(index: int, parent: torch.Tensor) -> None:
        nonlocal maximum
        node = tree.nodes[index]
        if node.is_leaf:
            maximum = max(maximum, float((parent - node.mean).abs().max()))
            return
        assert node.left is not None and node.right is not None
        left, right = representation.child_means(index, parent, k4=k4)
        visit(node.left, left); visit(node.right, right)

    visit(tree.root, _stored_vector(representation.root, k4))
    return maximum


def progressive_route_representation(
    representation: TreeRepresentation, query: torch.Tensor, token_budget: int, *, k4: bool = False
) -> dict[str, object]:
    """Best-first mean-score route using only a representation's stored vectors."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    tree, q = representation.tree, query.float()
    root_mean = _stored_vector(representation.root, k4)
    frontier: list[tuple[float, int, torch.Tensor]] = [(-float(q @ root_mean), tree.root, root_mean)]
    selected: list[int] = []
    tokens = expanded = vector_reads = 0
    while frontier and tokens < token_budget:
        _, index, mean = heapq.heappop(frontier)
        node = tree.nodes[index]
        if node.is_leaf:
            selected.append(index)
            tokens += node.count
            continue
        assert node.left is not None and node.right is not None
        left, right = representation.child_means(index, mean, k4=k4)
        heapq.heappush(frontier, (-float(q @ left), node.left, left))
        heapq.heappush(frontier, (-float(q @ right), node.right, right))
        expanded += 1
        vector_reads += representation.vectors_per_expansion
    token_ids = torch.cat([torch.arange(tree.nodes[i].start, tree.nodes[i].end) for i in selected]) if selected else torch.empty(0, dtype=torch.long)
    return {"leaf_indices": selected, "token_indices": token_ids, "candidate_tokens": tokens,
            "internal_nodes_expanded": expanded, "routing_vector_reads": vector_reads}


def lifting_pair(a: torch.Tensor, b: torch.Tensor, n_a: int, n_b: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return length-aware scaling, detail, and alpha for two adjacent regions."""
    if n_a <= 0 or n_b <= 0:
        raise ValueError("child lengths must be positive")
    alpha = n_b / (n_a + n_b)
    detail = b.float() - a.float()
    mean = a.float() + alpha * detail
    return mean, detail, alpha


def reconstruct_pair(mean: torch.Tensor, detail: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    return mean - alpha * detail, mean + (1.0 - alpha) * detail


def contiguous_subatom_means(
    keys: torch.Tensor, start: int, end: int, atom_size: int
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Return contiguous mean keys for a selected leaf's local refinement.

    This is intentionally a query-time primitive rather than a new tree
    construction metric: callers choose which already-selected leaf to split.
    """
    if keys.ndim != 2 or not (0 <= start < end <= keys.shape[0]) or atom_size <= 0:
        raise ValueError("invalid key span or atom_size")
    spans = [(offset, min(offset + atom_size, end)) for offset in range(start, end, atom_size)]
    return torch.stack([keys[left:right].float().mean(0) for left, right in spans]), spans


def query_cov_projection(calibration_queries: torch.Tensor, rank: int) -> torch.Tensor:
    """P with P.T @ P equal to the rank-truncated empirical query covariance."""
    if calibration_queries.ndim != 2 or calibration_queries.shape[0] == 0:
        raise ValueError("calibration_queries must be nonempty [samples, dimension]")
    covariance = calibration_queries.float().T @ calibration_queries.float() / calibration_queries.shape[0]
    values, vectors = torch.linalg.eigh(covariance)
    rank = min(rank, values.numel())
    values, vectors = values[-rank:].clamp_min(0), vectors[:, -rank:]
    return values.sqrt().unsqueeze(1) * vectors.T


def diagonal_query_cov_cost(sqrt_variance: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """Return d.T diag(E[q^2]) d without cross-dimension cancellation."""
    return (sqrt_variance * detail).square().sum()


def build_adaptive_tree(
    keys: torch.Tensor,
    *,
    metric: MetricName = "k_l2",
    calibration_queries: torch.Tensor | None = None,
    rank: int | None = None,
    balance_ratio: float | None = None,
    hybrid_lambda: float | None = None,
    atom_size: int = 16,
) -> AdaptiveLiftingTree:
    """Greedily agglomerate adjacent atoms, with stable leftmost tie breaking."""
    if keys.ndim != 2 or keys.shape[0] == 0 or atom_size <= 0:
        raise ValueError("keys must be nonempty [tokens, dimension] and atom_size positive")
    nodes = [
        LiftingNode(i, start, min(start + atom_size, keys.shape[0]), min(start + atom_size, keys.shape[0]) - start,
                    keys[start:min(start + atom_size, keys.shape[0])].float().mean(0))
        for i, start in enumerate(range(0, keys.shape[0], atom_size))
    ]
    projection: torch.Tensor | None = None
    is_diag = metric in ("query_cov_diag", "ward_query_cov_diag")
    is_lowrank = metric in ("query_cov_lowrank", "ward_query_cov_lowrank")
    if is_diag:
        if calibration_queries is None or calibration_queries.numel() == 0:
            raise ValueError("query covariance metrics require earlier calibration queries")
        projection = calibration_queries.float().square().mean(0).sqrt()
    elif is_lowrank:
        if calibration_queries is None or rank is None:
            raise ValueError("low-rank query covariance requires calibration queries and rank")
        projection = query_cov_projection(calibration_queries, rank)
    elif metric not in ("k_l2", "ward_k_l2", "hybrid"):
        raise ValueError(f"unknown metric: {metric}")
    if metric == "hybrid" and hybrid_lambda is None:
        raise ValueError("hybrid metric requires hybrid_lambda")

    active = {node.index for node in nodes}
    previous: dict[int, int | None] = {node.index: node.index - 1 if node.index else None for node in nodes}
    following: dict[int, int | None] = {
        node.index: node.index + 1 if node.index + 1 < len(nodes) else None for node in nodes
    }
    queue: list[tuple[float, int, int, int]] = []
    serial = itertools.count()

    def components(left: int, right: int) -> tuple[float, float, float]:
        detail = nodes[right].mean - nodes[left].mean
        l2 = float(detail.square().sum())
        attention = l2 if projection is None else float(
            diagonal_query_cov_cost(projection, detail) if is_diag else (projection @ detail).square().sum()
        )
        weight = nodes[left].count * nodes[right].count / (nodes[left].count + nodes[right].count)
        return l2, attention, weight

    # Initial adjacent costs are a stable, calibration-free unit conversion for
    # the exploratory hybrid; this avoids one unit dominating by magnitude.
    initial = [components(i, i + 1) for i in range(len(nodes) - 1)]
    scales = {
        "k_l2": max(torch.tensor([x[0] for x in initial]).median().item(), 1e-12),
        "attention_energy": max(torch.tensor([x[1] for x in initial]).median().item(), 1e-12),
    }

    def cost(left: int, right: int) -> float:
        l2, attention, weight = components(left, right)
        if metric == "hybrid":
            assert hybrid_lambda is not None
            return weight * (hybrid_lambda * l2 / scales["k_l2"] + (1 - hybrid_lambda) * attention / scales["attention_energy"])
        value = attention if projection is not None else l2
        return weight * value if metric.startswith("ward_") else value

    def push(left: int | None, right: int | None) -> None:
        if left is not None and right is not None:
            # serial makes fully equal entries deterministic even if IDs differ.
            heapq.heappush(queue, (cost(left, right), nodes[left].start, next(serial), left, right))

    for i in range(len(nodes) - 1):
        push(i, i + 1)
    relaxations = 0
    while len(active) > 1:
        deferred: list[tuple[float, int, int, int, int]] = []
        selected: tuple[int, int] | None = None
        while queue:
            _, _, _, left, right = heapq.heappop(queue)
            if left not in active or right not in active or following[left] != right:
                continue
            ratio = max(nodes[left].count, nodes[right].count) / min(nodes[left].count, nodes[right].count)
            if balance_ratio is None or ratio <= balance_ratio:
                selected = (left, right)
                break
            deferred.append((cost(left, right), nodes[left].start, next(serial), left, right))
        if selected is None:
            # No eligible adjacent merge remains.  Relax exactly one smallest
            # violating merge, rather than globally abandoning the constraint.
            if not deferred:
                raise RuntimeError("adjacency queue unexpectedly empty")
            deferred.sort()
            _, _, _, left, right = deferred.pop(0)
            selected = (left, right)
            relaxations += 1
        for item in deferred:
            heapq.heappush(queue, item)
        left, right = selected
        parent_mean, detail, alpha = lifting_pair(nodes[left].mean, nodes[right].mean, nodes[left].count, nodes[right].count)
        parent = LiftingNode(len(nodes), nodes[left].start, nodes[right].end, nodes[left].count + nodes[right].count,
                             parent_mean, left, right, alpha, detail, cost(left, right))
        nodes.append(parent)
        before, after = previous[left], following[right]
        active.remove(left)
        active.remove(right)
        active.add(parent.index)
        previous[parent.index], following[parent.index] = before, after
        if before is not None:
            following[before] = parent.index
        if after is not None:
            previous[after] = parent.index
        push(before, parent.index)
        push(parent.index, after)
    return AdaptiveLiftingTree(nodes, next(iter(active)), atom_size, metric, balance_ratio, relaxations, projection, scales)


def build_balanced_tree(keys: torch.Tensor, *, atom_size: int = 16) -> AdaptiveLiftingTree:
    """Near-balanced, content-independent contiguous control tree."""
    if keys.ndim != 2 or keys.shape[0] == 0:
        raise ValueError("keys must be nonempty [tokens, dimension]")
    leaves = [
        LiftingNode(i, start, min(start + atom_size, keys.shape[0]), min(start + atom_size, keys.shape[0]) - start,
                    keys[start:min(start + atom_size, keys.shape[0])].float().mean(0))
        for i, start in enumerate(range(0, keys.shape[0], atom_size))
    ]
    nodes = list(leaves)

    def join(indices: list[int]) -> int:
        if len(indices) == 1:
            return indices[0]
        middle = len(indices) // 2
        left, right = join(indices[:middle]), join(indices[middle:])
        mean, detail, alpha = lifting_pair(nodes[left].mean, nodes[right].mean, nodes[left].count, nodes[right].count)
        nodes.append(LiftingNode(len(nodes), nodes[left].start, nodes[right].end, nodes[left].count + nodes[right].count,
                                 mean, left, right, alpha, detail))
        return nodes[-1].index

    return AdaptiveLiftingTree(nodes, join([x.index for x in leaves]), atom_size)


@dataclass
class OnlineBuilderStats:
    """Per-atom construction work recorded by the causal reference builders."""

    cost_evaluations: list[int]
    vector_operations: list[int]
    merges: list[int]
    max_active_regions: list[int]
    temporary_vectors: list[int]
    grouped_scalar_additions: list[int] | None = None
    grouped_scalar_multiplies: list[int] | None = None
    grouped_scalar_metadata_writes: list[int] | None = None

    def __post_init__(self) -> None:
        self.grouped_scalar_additions = self.grouped_scalar_additions or []
        self.grouped_scalar_multiplies = self.grouped_scalar_multiplies or []
        self.grouped_scalar_metadata_writes = self.grouped_scalar_metadata_writes or []

    def summary(self) -> dict[str, dict[str, float]]:
        def values(items: list[int]) -> dict[str, float]:
            sample = torch.tensor(items, dtype=torch.float32)
            return {"mean": float(sample.mean()), "p95": float(torch.quantile(sample, .95)), "max": float(sample.max())}
        return {"merge_cost_evaluations": values(self.cost_evaluations), "vector_arithmetic_operations": values(self.vector_operations),
                "merges": values(self.merges), "max_active_regions": values(self.max_active_regions),
                "stored_temporary_vectors": values(self.temporary_vectors),
                "grouped_scalar_additions": values(self.grouped_scalar_additions),
                "grouped_scalar_multiplies": values(self.grouped_scalar_multiplies),
                "grouped_scalar_metadata_writes": values(self.grouped_scalar_metadata_writes)}


@dataclass
class StreamingLiftingForest:
    """Persistent, append-only atom builder state.

    Nodes are never changed after creation.  The small ``roots`` list is the
    only mutable state and deliberately remains a forest at query time.
    """

    mode: Literal["binary_counter", "local_greedy", "causal_threshold_p50"]
    atom_size: int = 8
    window: int = 4
    balance_ratio: float = 4.0
    warmup_atoms: int = 128
    group_counts: tuple[int, ...] = (4, 8)
    nodes: list[LiftingNode] | None = None
    roots: list[int] | None = None
    threshold: float | None = None
    warmup_costs: list[float] | None = None
    stats: OnlineBuilderStats | None = None

    def __post_init__(self) -> None:
        self.nodes = [] if self.nodes is None else self.nodes
        self.roots = [] if self.roots is None else self.roots
        self.warmup_costs = [] if self.warmup_costs is None else self.warmup_costs
        self.stats = self.stats or OnlineBuilderStats([], [], [], [], [])

    @staticmethod
    def _grouped_m2(values: torch.Tensor, mean: torch.Tensor, groups: int) -> torch.Tensor:
        if mean.numel() % groups:
            raise ValueError("group count must divide key dimension")
        return (values.float() - mean.float()).square().reshape(-1, groups, mean.numel() // groups).sum((0, 2))

    @staticmethod
    def _merge_grouped_m2(left: LiftingNode, right: LiftingNode, mean: torch.Tensor, groups: int) -> torch.Tensor:
        assert left.grouped_m2 is not None and right.grouped_m2 is not None
        width = mean.numel() // groups
        correction = (left.count * (left.mean.float() - mean).square()
                      + right.count * (right.mean.float() - mean).square())
        return left.grouped_m2[groups] + right.grouped_m2[groups] + correction.reshape(groups, width).sum(1)

    def _cost(self, left: int, right: int) -> float:
        assert self.nodes is not None
        return float((self.nodes[left].mean - self.nodes[right].mean).square().sum())

    def _eligible(self, left: int, right: int) -> bool:
        assert self.nodes is not None
        return max(self.nodes[left].count, self.nodes[right].count) / min(self.nodes[left].count, self.nodes[right].count) <= self.balance_ratio

    def _join(self, at: int, work: list[int]) -> None:
        assert self.nodes is not None and self.roots is not None
        left, right = self.roots[at:at + 2]
        mean, detail, alpha = lifting_pair(self.nodes[left].mean, self.nodes[right].mean, self.nodes[left].count, self.nodes[right].count)
        parent_m2 = {groups: self._merge_grouped_m2(self.nodes[left], self.nodes[right], mean, groups)
                     for groups in self.group_counts}
        self.nodes.append(LiftingNode(len(self.nodes), self.nodes[left].start, self.nodes[right].end,
                                      self.nodes[left].count + self.nodes[right].count, mean, left, right,
                                      alpha, detail, self._cost(left, right), parent_m2))
        self.roots[at:at + 2] = [self.nodes[-1].index]
        work[1] += 4 * self.nodes[-1].mean.numel()
        work[2] += 1
        # Chan correction: two child terms and the parent accumulation per
        # group.  Count scalar work separately from vector operations.
        # These counters describe the physical metadata configuration retained
        # by this forest.  A G=8 summary is eight scalars, not one entry in
        # ``group_counts``.  Consumers wanting an isolated G protocol derive
        # its counters from the raw per-atom merge counts.
        total_groups = sum(self.group_counts)
        work[3] += 3 * total_groups
        work[4] += 2 * total_groups
        work[5] += total_groups

    def append_atom(self, atom: torch.Tensor) -> None:
        """Append exactly one completed atom; no query can trigger a merge."""
        if atom.ndim != 2 or atom.shape[0] == 0 or atom.shape[0] > self.atom_size:
            raise ValueError("atom must be a nonempty completed [<=atom_size, dimension] tensor")
        assert self.nodes is not None and self.roots is not None and self.stats is not None and self.warmup_costs is not None
        start = self.nodes[self.roots[-1]].end if self.roots else 0
        mean = atom.float().mean(0)
        leaf_m2 = {groups: self._grouped_m2(atom, mean, groups) for groups in self.group_counts}
        self.nodes.append(LiftingNode(len(self.nodes), start, start + atom.shape[0], atom.shape[0], mean,
                                      grouped_m2=leaf_m2))
        self.roots.append(self.nodes[-1].index)
        # Vector arithmetic includes leaf mean formation plus the coordinate
        # squared-difference work used to form leaf grouped M2.  The grouped
        # counters separately charge group reductions/writes.
        total_groups = sum(self.group_counts)
        work = [0, atom.shape[1] + 2 * atom.numel(), 0,
                total_groups, total_groups, total_groups]
        if self.mode == "binary_counter":
            while len(self.roots) >= 2 and self.nodes[self.roots[-1]].count == self.nodes[self.roots[-2]].count:
                self._join(len(self.roots) - 2, work)
        else:
            warmup = self.mode == "causal_threshold_p50" and len(self.stats.merges) < self.warmup_atoms
            while True:
                begin = max(0, len(self.roots) - self.window)
                pairs = [(self._cost(self.roots[i], self.roots[i + 1]), i) for i in range(begin, len(self.roots) - 1) if self._eligible(self.roots[i], self.roots[i + 1])]
                work[0] += len(pairs)
                if warmup:
                    self.warmup_costs.extend(value for value, _ in pairs)
                forced = len(self.roots) > self.window
                if not pairs:
                    if forced:
                        self._join(begin, work); continue
                    break
                value, at = min(pairs, key=lambda item: (item[0], self.nodes[self.roots[item[1]]].start))
                accept = self.mode == "local_greedy" or warmup or forced or (self.threshold is not None and value <= self.threshold)
                if not accept:
                    break
                self._join(at, work)
                if (self.mode == "local_greedy" or warmup) and not forced:
                    break
            if self.mode == "causal_threshold_p50" and len(self.stats.merges) + 1 == self.warmup_atoms:
                self.threshold = float(torch.tensor(self.warmup_costs).median()) if self.warmup_costs else 0.0
        self.stats.cost_evaluations.append(work[0]); self.stats.vector_operations.append(work[1])
        self.stats.merges.append(work[2]); self.stats.max_active_regions.append(len(self.roots))
        self.stats.temporary_vectors.append(len(self.roots))
        self.stats.grouped_scalar_additions.append(work[3])
        self.stats.grouped_scalar_multiplies.append(work[4])
        self.stats.grouped_scalar_metadata_writes.append(work[5])

    def append_keys(self, keys: torch.Tensor) -> None:
        for start in range(0, keys.shape[0], self.atom_size):
            self.append_atom(keys[start:min(start + self.atom_size, keys.shape[0])])

    def snapshot(self) -> tuple[tuple[int, int, int], ...]:
        assert self.nodes is not None and self.roots is not None
        return tuple((node, self.nodes[node].start, self.nodes[node].end) for node in self.roots)


def progressive_route_forest(forest: StreamingLiftingForest, query: torch.Tensor, token_budget: int, *, k4: bool = False) -> dict[str, object]:
    """Route a forest with one shared best-first frontier, without mutation."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    assert forest.nodes is not None and forest.roots is not None
    q, frontier = query.float(), []
    for root in forest.roots:
        mean = _stored_vector(forest.nodes[root].mean, k4)
        heapq.heappush(frontier, (-float(q @ mean), root, mean))
    selected: list[int] = []; tokens = expanded = 0
    while frontier and tokens < token_budget:
        _, index, mean = heapq.heappop(frontier); node = forest.nodes[index]
        if node.is_leaf:
            if tokens + node.count <= token_budget:
                selected.append(index); tokens += node.count
            continue
        assert node.left is not None and node.right is not None and node.detail is not None and node.alpha is not None
        left, right = reconstruct_pair(mean, _stored_vector(node.detail, k4), node.alpha)
        heapq.heappush(frontier, (-float(q @ left), node.left, left)); heapq.heappush(frontier, (-float(q @ right), node.right, right)); expanded += 1
    ids = torch.cat([torch.arange(forest.nodes[i].start, forest.nodes[i].end) for i in selected]) if selected else torch.empty(0, dtype=torch.long)
    return {"leaf_indices": selected, "token_indices": ids, "candidate_tokens": tokens, "internal_nodes_expanded": expanded,
            "routing_vector_reads": expanded, "active_root_vector_reads": len(forest.roots)}


def build_online_tree(
    keys: torch.Tensor, *, mode: Literal["binary_counter", "local_greedy", "threshold"], atom_size: int = 8,
    window: int = 8, threshold: float | None = None, balance_ratio: float = 4.0,
) -> tuple[AdaptiveLiftingTree, OnlineBuilderStats]:
    """Build a contiguous tree without inspecting atoms that have not arrived.

    ``local_greedy`` only merges pairs in the newest active window; regions to
    its left are never reconsidered. ``threshold`` uses the same bounded
    causal stack and deterministic size-pressure merges for progress.
    """
    if keys.ndim != 2 or keys.shape[0] == 0 or atom_size <= 0:
        raise ValueError("keys must be nonempty [tokens, dimension]")
    if mode == "threshold" and threshold is None:
        raise ValueError("threshold mode requires a calibrated threshold")
    if window < 2:
        raise ValueError("window must be at least two")
    nodes: list[LiftingNode] = []
    stack: list[int] = []
    costs, ops, merges, active, temporary = [], [], [], [], []

    def cost(left: int, right: int) -> float:
        return float((nodes[left].mean - nodes[right].mean).square().sum())

    def eligible(left: int, right: int) -> bool:
        return max(nodes[left].count, nodes[right].count) / min(nodes[left].count, nodes[right].count) <= balance_ratio

    def join(at: int, work: list[int]) -> None:
        left, right = stack[at], stack[at + 1]
        mean, detail, alpha = lifting_pair(nodes[left].mean, nodes[right].mean, nodes[left].count, nodes[right].count)
        node = LiftingNode(len(nodes), nodes[left].start, nodes[right].end, nodes[left].count + nodes[right].count,
                           mean, left, right, alpha, detail, cost(left, right))
        nodes.append(node); stack[at:at + 2] = [node.index]
        work[1] += 4  # two weighted vector updates plus detail and mean bookkeeping
        work[2] += 1

    for start in range(0, keys.shape[0], atom_size):
        end = min(start + atom_size, keys.shape[0])
        nodes.append(LiftingNode(len(nodes), start, end, end - start, keys[start:end].float().mean(0)))
        stack.append(nodes[-1].index)
        work = [0, 1, 0]  # cost evals, vector ops, merges
        if mode == "binary_counter":
            # Equal-span carry operations are a streaming binary counter.
            while len(stack) >= 2 and nodes[stack[-1]].count == nodes[stack[-2]].count:
                join(len(stack) - 2, work)
        else:
            # Only the recent suffix is mutable. Choose the lowest-cost eligible
            # pair, stable by left position, and either satisfy the threshold or
            # make room at the active-window boundary.
            while True:
                begin = max(0, len(stack) - window)
                pairs = [(cost(stack[i], stack[i + 1]), i) for i in range(begin, len(stack) - 1) if eligible(stack[i], stack[i + 1])]
                work[0] += len(pairs)
                forced = len(stack) > window
                if not pairs:
                    if forced:
                        # Deterministic relaxed merge at the boundary ensures bounded state.
                        join(begin, work)
                        continue
                    break
                value, at = min(pairs, key=lambda item: (item[0], nodes[stack[item[1]]].start))
                if mode == "local_greedy" or forced or value <= float(threshold):
                    join(at, work)
                    # Local greedy makes one voluntary decision per arrival;
                    # threshold mode may consume multiple qualifying pairs.
                    if mode == "local_greedy" and not forced:
                        break
                    continue
                break
        costs.append(work[0]); ops.append(work[1]); merges.append(work[2]); active.append(len(stack)); temporary.append(len(stack))
    # Finalization occurs only after the completed prefix; it is deterministic
    # and does not affect the per-arrival causal work accounting.
    while len(stack) > 1:
        pairs = [(cost(stack[i], stack[i + 1]), i) for i in range(len(stack) - 1) if eligible(stack[i], stack[i + 1])]
        at = min(pairs, key=lambda item: (item[0], nodes[stack[item[1]]].start))[1] if pairs else 0
        join(at, [0, 0, 0])
    return AdaptiveLiftingTree(nodes, stack[0], atom_size, f"online_{mode}", balance_ratio), OnlineBuilderStats(costs, ops, merges, active, temporary)


def _stored_vector(vector: torch.Tensor, k4: bool) -> torch.Tensor:
    return quantize_symmetric(vector, 4, 16).dequantize() if k4 else vector.float()


def subtree_uncertainty_summaries(
    tree: AdaptiveLiftingTree, calibration_queries: torch.Tensor
) -> dict[str, dict[int, float]]:
    """Derive normalized one-scalar uncertainty summaries for internal nodes.

    This is deliberately separate from ``AdaptiveLiftingTree``: the lifting
    representation remains root scaling plus details.  The four returned maps
    are reference metadata which a future implementation may choose to store.
    Descendants include the node's own detail, so all four quantities can be
    maintained by a bottom-up builder with count/sum/max accumulators.
    """
    if calibration_queries.ndim != 2 or calibration_queries.shape[0] == 0:
        raise ValueError("calibration_queries must be nonempty [samples, dimension]")
    covariance = calibration_queries.float().T @ calibration_queries.float() / calibration_queries.shape[0]
    raw: dict[str, dict[int, float]] = {name: {} for name in ("local", "norm", "max", "attn")}
    aggregate: dict[int, tuple[int, float, float, float]] = {}

    def visit(index: int) -> tuple[int, float, float, float]:
        node = tree.nodes[index]
        if node.is_leaf:
            return 0, 0.0, 0.0, 0.0
        assert node.left is not None and node.right is not None and node.detail is not None
        left, right = visit(node.left), visit(node.right)
        norm_sq = float(node.detail.float().square().sum())
        attn_sq = float(node.detail.float() @ covariance @ node.detail.float())
        count = 1 + left[0] + right[0]
        norm_sum, attn_sum = norm_sq + left[1] + right[1], attn_sq + left[2] + right[2]
        max_norm = max(norm_sq**0.5, left[3], right[3])
        aggregate[index] = (count, norm_sum, attn_sum, max_norm)
        raw["local"][index] = norm_sq**0.5
        raw["norm"][index] = (norm_sum / count) ** 0.5
        raw["max"][index] = max_norm
        raw["attn"][index] = max(attn_sum / count, 0.0) ** 0.5
        return aggregate[index]

    visit(tree.root)
    # RMS scaling is robust to a zero median and does not let a near-zero
    # calibration median explode.  Each tree/KV-head gets its own scale.
    normalized: dict[str, dict[int, float]] = {}
    for name, values in raw.items():
        scale = (sum(value * value for value in values.values()) / max(len(values), 1)) ** 0.5
        scale = max(scale, 1e-12)
        normalized[name] = {index: value / scale for index, value in values.items()}
    return normalized


def progressive_route(
    tree: AdaptiveLiftingTree,
    query: torch.Tensor,
    token_budget: int,
    *,
    k4: bool = False,
    uncertainty: dict[int, float] | None = None,
    beta: float = 0.0,
    span_scale: bool = False,
    parent_detail_gamma: float = 0.0,
) -> dict[str, object]:
    """Best-first routing with optional scalar uncertainty or parent-detail priority.

    One detail vector is still read only when an internal node is expanded.
    ``uncertainty`` is indexed only by internal nodes; leaves have no
    unresolved detail and therefore require no metadata read.
    """
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    root = tree.nodes[tree.root]
    q = query.float()
    q_norm = float(q.norm())
    metadata_nodes_read: set[int] = set()

    def priority(index: int, mean: torch.Tensor, parent_detail_score: float = 0.0) -> float:
        node = tree.nodes[index]
        value = float(q @ mean)
        if uncertainty is not None and not node.is_leaf and beta:
            metadata_nodes_read.add(index)
            factor = (node.count / tree.atom_size) ** 0.5 if span_scale else 1.0
            value += beta * q_norm * uncertainty[index] * factor
        return value + parent_detail_gamma * abs(parent_detail_score)

    root_mean = _stored_vector(root.mean, k4)
    frontier: list[tuple[float, int, torch.Tensor]] = [(-priority(root.index, root_mean), root.index, root_mean)]
    selected: list[int] = []
    candidate_tokens = 0
    expanded = 0
    while frontier and candidate_tokens < token_budget:
        _, index, mean = heapq.heappop(frontier)
        node = tree.nodes[index]
        if node.is_leaf:
            selected.append(index)
            candidate_tokens += node.count
            continue
        assert node.left is not None and node.right is not None and node.detail is not None and node.alpha is not None
        detail = _stored_vector(node.detail, k4)
        left_mean, right_mean = reconstruct_pair(mean, detail, node.alpha)
        parent_detail_score = float(q @ detail)
        heapq.heappush(frontier, (-priority(node.left, left_mean, parent_detail_score), node.left, left_mean))
        heapq.heappush(frontier, (-priority(node.right, right_mean, parent_detail_score), node.right, right_mean))
        expanded += 1
    token_ids = torch.cat([torch.arange(tree.nodes[i].start, tree.nodes[i].end) for i in selected]) if selected else torch.empty(0, dtype=torch.long)
    # Scores are retained for loss analysis; these are the regions left
    # unrefined because the token budget was reached.
    final_frontier = [(-item_priority, index) for item_priority, index, _ in frontier]
    frontier_states = [( -item_priority, index, mean) for item_priority, index, mean in frontier]
    return {"leaf_indices": selected, "token_indices": token_ids, "internal_detail_vectors_read": expanded,
            "internal_nodes_expanded": expanded, "microchunks_selected": len(selected), "candidate_tokens": candidate_tokens,
            "frontier": final_frontier, "frontier_states": frontier_states,
            "uncertainty_metadata_nodes_read": len(metadata_nodes_read)}
