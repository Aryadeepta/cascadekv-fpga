"""Progressive Q8 x K4 candidate routing with exact per-stage accounting.

The partial scores in this module are *unscaled accumulated contributions*.
This is important: a candidate that survives a stage carries its accumulator
forward, rather than having an earlier prefix re-read or re-scaled.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from cascadekv.quantize import QuantizedTensor, quantize_symmetric

GROUP_SIZE = 16
HEAD_DIM = 128
INT4_K_BYTES_PER_VECTOR = 80  # 64 packed INT4 bytes + eight FP16 scales.
FP16_K_BYTES_PER_VECTOR = 256
INT4_ROUTER_FP16_RERANK_STORAGE_BYTES_PER_VECTOR = INT4_K_BYTES_PER_VECTOR + FP16_K_BYTES_PER_VECTOR
STAGE_GROUPS = (1, 1, 2, 4)
STAGE_DIMENSIONS = (16, 32, 64, 128)


@dataclass(frozen=True)
class CascadeSchedule:
    """Numbers retained after 16, 32, and 64 dimensions before final INT4 scoring."""

    name: str
    survivors: tuple[int, int, int]

    def counts(self, candidate_count: int, final_k: int = 32) -> tuple[int, int, int]:
        if candidate_count < final_k:
            raise ValueError("candidate_count must be at least final_k")
        counts: list[int] = []
        previous = candidate_count
        for requested in self.survivors:
            count = min(previous, max(final_k, requested))
            counts.append(count)
            previous = count
        return tuple(counts)  # type: ignore[return-value]


@dataclass(frozen=True)
class StageAccounting:
    candidate_count_entering: int
    candidate_count_exiting: int
    newly_read_k_coordinate_bytes: int
    newly_read_k_scale_bytes: int
    integer_macs: int


@dataclass(frozen=True)
class CascadeResult:
    stage_ids: tuple[torch.Tensor, ...]
    stage_scores: tuple[torch.Tensor, ...]
    final_scores: torch.Tensor
    accounting: tuple[StageAccounting, ...]
    # Diagnostic only: this assumes an additional high-precision K cache.
    oracle_fp32_rerank_scores: torch.Tensor | None = None

    @property
    def final_ids(self) -> torch.Tensor:
        return self.stage_ids[-1]

    def final_top_ids(self, count: int) -> torch.Tensor:
        """Top IDs from the final deployable 128-D Q8xK4 score."""
        return select_top_ids(self.final_ids, self.final_scores, count)[0]


def fraction_schedule(name: str, fractions: tuple[float, float, float], candidate_count: int) -> CascadeSchedule:
    """Make deterministic ceil-rounded survivor targets for a particular context."""
    if len(fractions) != 3 or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("three fractions in (0, 1] are required")
    return CascadeSchedule(name, tuple(int(torch.ceil(torch.tensor(value * candidate_count)).item()) for value in fractions))


def select_top_ids(ids: torch.Tensor, scores: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select by score, breaking ties by lower original key ID deterministically."""
    if ids.ndim != 1 or scores.ndim != 1 or ids.numel() != scores.numel():
        raise ValueError("ids and scores must be aligned rank-one tensors")
    if not 0 < count <= ids.numel():
        raise ValueError("count must lie in [1, number of candidates]")
    # Stable sorting establishes ID order first, then score order for ties.
    by_id = torch.argsort(ids, stable=True)
    by_score = torch.argsort(scores[by_id], descending=True, stable=True)[:count]
    indices = by_id[by_score]
    return ids[indices], scores[indices]


class Q8K4Cascade:
    """Prepared per-query Q8 x K4 group-16 contributions and cascade runner."""

    def __init__(self, q: torch.Tensor, keys: torch.Tensor, ordering: torch.Tensor | list[int]) -> None:
        if q.ndim != 1 or keys.ndim != 2 or q.numel() != HEAD_DIM or keys.shape[1] != HEAD_DIM:
            raise ValueError("q must be [128] and keys must be [candidates, 128]")
        order = torch.as_tensor(ordering, dtype=torch.long, device=q.device)
        if order.numel() != HEAD_DIM or not torch.equal(order.sort().values.cpu(), torch.arange(HEAD_DIM)):
            raise ValueError("ordering must be a 128-coordinate permutation")
        self.q = q.float()
        self.keys = keys.float()
        self.order = order
        ordered_q, ordered_keys = self.q[order], self.keys[:, order]
        self.q_quantized: QuantizedTensor = quantize_symmetric(ordered_q, 8, GROUP_SIZE)
        self.k_quantized: QuantizedTensor = quantize_symmetric(ordered_keys, 4, GROUP_SIZE)
        q_codes = self.q_quantized.values.to(torch.int32)
        k_codes = self.k_quantized.values.to(torch.int32)
        terms = []
        for group in range(HEAD_DIM // GROUP_SIZE):
            start = group * GROUP_SIZE
            integer_dot = (k_codes[:, start : start + GROUP_SIZE] * q_codes[start : start + GROUP_SIZE]).sum(
                dim=-1, dtype=torch.int32
            )
            terms.append(integer_dot.float() * self.q_quantized.scales[group] * self.k_quantized.scales[:, group])
        self.group_terms = torch.stack(terms, dim=0)

    def direct_prefix_scores(self, dimensions: int) -> torch.Tensor:
        if dimensions not in STAGE_DIMENSIONS:
            raise ValueError("dimensions must be one of 16, 32, 64, 128")
        return self.group_terms[: dimensions // GROUP_SIZE].sum(dim=0)

    def accounting_for_schedule(self, schedule: CascadeSchedule, *, final_k: int = 32) -> tuple[StageAccounting, ...]:
        """Return traffic/MAC rows without executing any score sorting."""
        retained = schedule.counts(self.keys.shape[0], final_k)
        entering = self.keys.shape[0]
        rows: list[StageAccounting] = []
        for stage, groups in enumerate(STAGE_GROUPS):
            exiting = retained[stage] if stage < 3 else entering
            rows.append(StageAccounting(
                entering, exiting, entering * groups * 8, entering * groups * 2,
                entering * groups * GROUP_SIZE,
            ))
            entering = exiting
        return tuple(rows)

    def run(self, schedule: CascadeSchedule, *, final_k: int = 32, oracle_fp32_rerank: bool = False) -> CascadeResult:
        """Run the deployable INT4-only cascade.

        ``oracle_fp32_rerank`` is explicitly diagnostic: it is evaluated only
        over final survivors and is never included in deployable accounting.
        """
        total = self.keys.shape[0]
        retained = schedule.counts(total, final_k)
        ids = torch.arange(total, device=self.keys.device)
        scores = torch.zeros(total, device=self.keys.device)
        stage_ids: list[torch.Tensor] = []
        stage_scores: list[torch.Tensor] = []
        accounting = list(self.accounting_for_schedule(schedule, final_k=final_k))
        starts = (0, 1, 2, 4)
        for stage, (start, groups) in enumerate(zip(starts, STAGE_GROUPS)):
            scores = scores + self.group_terms[start : start + groups, ids].sum(dim=0)
            exiting = retained[stage] if stage < 3 else ids.numel()
            selected_ids, selected_scores = select_top_ids(ids, scores, exiting)
            ids, scores = selected_ids, selected_scores
            stage_ids.append(ids)
            stage_scores.append(scores)
        oracle_scores = (self.keys[ids] * self.q).sum(dim=-1) if oracle_fp32_rerank else None
        return CascadeResult(tuple(stage_ids), tuple(stage_scores), scores, tuple(accounting), oracle_scores)


def random_cascade(candidate_count: int, schedule: CascadeSchedule, *, seed: int, final_k: int = 32) -> tuple[torch.Tensor, ...]:
    """Nested, deterministic random survivor IDs for the matched-pruning baseline."""
    counts = schedule.counts(candidate_count, final_k)
    generator = torch.Generator().manual_seed(seed)
    ids = torch.arange(candidate_count)
    result = []
    for count in counts:
        ids = ids[torch.randperm(ids.numel(), generator=generator)[:count]]
        result.append(ids)
    result.append(ids)
    return tuple(result)


def accounting_totals(accounting: Iterable[StageAccounting], candidate_count: int) -> dict[str, float | int]:
    rows = tuple(accounting)
    k_bytes = sum(row.newly_read_k_coordinate_bytes + row.newly_read_k_scale_bytes for row in rows)
    integer_macs = sum(row.integer_macs for row in rows)
    dense_int4_bytes = candidate_count * INT4_K_BYTES_PER_VECTOR
    dense_fp16_bytes = candidate_count * FP16_K_BYTES_PER_VECTOR
    dense_macs = candidate_count * HEAD_DIM
    return {
        "total_k_bytes": k_bytes, "dense_int4_k_bytes": dense_int4_bytes, "dense_fp16_k_bytes": dense_fp16_bytes,
        "int4_bytes_avoided_percent": 100 * (1 - k_bytes / dense_int4_bytes),
        "fp16_bytes_avoided_percent": 100 * (1 - k_bytes / dense_fp16_bytes),
        "total_quantized_macs": integer_macs,
        "dense_128d_macs": dense_macs,
        "macs_avoided_percent": 100 * (1 - integer_macs / dense_macs),
    }


def router_fp16_rerank_totals(
    accounting: Iterable[StageAccounting], candidate_count: int, final_survivors: int
) -> dict[str, float | int]:
    """Account for INT4 routing through 64-D, followed by FP16 K reranking.

    The final (128-D) INT4 stage is deliberately excluded: Architecture B
    fetches authoritative FP16 K vectors for the 64-D survivors instead.
    """
    routing_rows = tuple(accounting)[:3]
    router_k_bytes = sum(row.newly_read_k_coordinate_bytes + row.newly_read_k_scale_bytes for row in routing_rows)
    router_macs = sum(row.integer_macs for row in routing_rows)
    rerank_k_bytes = final_survivors * FP16_K_BYTES_PER_VECTOR
    total_k_bytes = router_k_bytes + rerank_k_bytes
    dense_fp16_k_bytes = candidate_count * FP16_K_BYTES_PER_VECTOR
    return {
        "router_int4_k_bytes": router_k_bytes,
        "rerank_fp16_k_bytes": rerank_k_bytes,
        "total_k_bytes_with_rerank": total_k_bytes,
        "dense_fp16_k_bytes": dense_fp16_k_bytes,
        "traffic_avoided_vs_dense_fp16_percent": 100 * (1 - total_k_bytes / dense_fp16_k_bytes),
        "router_integer_macs": router_macs,
        "rerank_high_precision_macs": final_survivors * HEAD_DIM,
    }
