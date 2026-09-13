"""Progressive partial-dot scores and lightweight ranking metrics."""

from __future__ import annotations

import torch


def full_dot_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute query/key dot products, leaving the key sequence axis intact."""
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key dimensions must match")
    return (q.unsqueeze(-2) * k).sum(dim=-1)


def progressive_dot_scores(
    q: torch.Tensor, k: torch.Tensor, dimensions: tuple[int, ...] = (16, 32, 64, 128)
) -> dict[int, torch.Tensor]:
    """Estimate full scores from nested prefixes of a rotated representation."""
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim:
        raise ValueError("query and key dimensions must match")
    if any(dimension <= 0 or dimension > head_dim for dimension in dimensions):
        raise ValueError(f"dimensions must lie in [1, {head_dim}]")
    return {
        dimension: (head_dim / dimension)
        * (q[..., :dimension].unsqueeze(-2) * k[..., :dimension]).sum(dim=-1)
        for dimension in dimensions
    }


def ordered_progressive_dot_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    ordering: torch.Tensor | list[int],
    dimensions: tuple[int, ...] = (16, 32, 64, 128),
    *,
    unbiased: bool = True,
) -> dict[int, torch.Tensor]:
    """Nested partial scores via one ordered contribution stream.

    A cumulative sum is the FPGA-relevant accumulator; requesting a wider
    dimension only consumes its next coordinate contributions.  With the
    default ``unbiased`` readout, each partial accumulator is multiplied by
    ``head_dim / dimension``.  The 128-D output is always the exact dot score.
    """
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim:
        raise ValueError("query and key dimensions must match")
    order = torch.as_tensor(ordering, device=q.device, dtype=torch.long)
    if order.ndim != 1 or order.numel() != head_dim or not torch.equal(
        order.sort().values.cpu(), torch.arange(head_dim)
    ):
        raise ValueError("ordering must be a permutation of [0, head_dim)")
    if any(dimension <= 0 or dimension > head_dim for dimension in dimensions):
        raise ValueError(f"dimensions must lie in [1, {head_dim}]")
    contributions = q.index_select(-1, order).unsqueeze(-2) * k.index_select(-1, order)
    cumulative = contributions.cumsum(-1)
    return {
        dimension: cumulative[..., dimension - 1] * (head_dim / dimension if unbiased else 1.0)
        for dimension in dimensions
    }


def _flatten_pairs(
    approximate: torch.Tensor, full: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if approximate.shape != full.shape:
        raise ValueError("score tensors must have equal shapes")
    if approximate.ndim == 0:
        raise ValueError("score tensors need a sequence dimension")
    return (
        approximate.reshape(-1, approximate.shape[-1]).float(),
        full.reshape(-1, full.shape[-1]).float(),
    )


def pearson_correlation(approximate: torch.Tensor, full: torch.Tensor) -> float:
    """Mean per-row Pearson correlation over the final (key) dimension."""
    approximate, full = _flatten_pairs(approximate, full)
    a = approximate - approximate.mean(dim=-1, keepdim=True)
    b = full - full.mean(dim=-1, keepdim=True)
    denominator = a.square().sum(-1).sqrt() * b.square().sum(-1).sqrt()
    valid = denominator > 0
    if not valid.any():
        return 0.0
    return ((a * b).sum(-1)[valid] / denominator[valid]).mean().clamp(-1, 1).item()


def spearman_correlation(approximate: torch.Tensor, full: torch.Tensor) -> float:
    """Spearman correlation using ordinal ranks (sufficient for score ties being rare)."""
    approximate, full = _flatten_pairs(approximate, full)
    a_rank = approximate.argsort(dim=-1).argsort(dim=-1).float()
    b_rank = full.argsort(dim=-1).argsort(dim=-1).float()
    return pearson_correlation(a_rank, b_rank)


def top_k_recall(approximate: torch.Tensor, full: torch.Tensor, k: int) -> float:
    """Fraction of exact top-k keys recovered by approximate top-k selection."""
    approximate, full = _flatten_pairs(approximate, full)
    if not 0 < k <= full.shape[-1]:
        raise ValueError(f"k must be in [1, {full.shape[-1]}]")
    approximate_indices = approximate.topk(k, dim=-1).indices
    full_indices = full.topk(k, dim=-1).indices
    matches = (approximate_indices.unsqueeze(-1) == full_indices.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean().item()


def attention_mass_recall(approximate: torch.Tensor, full: torch.Tensor, k: int) -> float:
    """Dense full-softmax probability mass retained by approximate top-k keys."""
    approximate, full = _flatten_pairs(approximate, full)
    if not 0 < k <= full.shape[-1]:
        raise ValueError(f"k must be in [1, {full.shape[-1]}]")
    indices = approximate.topk(k, dim=-1).indices
    return torch.softmax(full, dim=-1).gather(-1, indices).sum(-1).mean().item()


def relative_attention_mass_recall(approximate: torch.Tensor, full: torch.Tensor, k: int) -> float:
    """Mass of approximate top-k relative to the mass of exact top-k.

    The ratio removes variation caused purely by the concentration of a sample's
    dense attention distribution.  The exact ranking therefore has value one.
    """
    approximate, full = _flatten_pairs(approximate, full)
    if not 0 < k <= full.shape[-1]:
        raise ValueError(f"k must be in [1, {full.shape[-1]}]")
    probabilities = torch.softmax(full, dim=-1)
    approximate_mass = probabilities.gather(-1, approximate.topk(k, dim=-1).indices).sum(-1)
    exact_mass = probabilities.gather(-1, full.topk(k, dim=-1).indices).sum(-1)
    # Exact top-k maximizes mass, so numerical noise is the only reason this
    # could very slightly exceed one.
    return (approximate_mass / exact_mass).mean().clamp(0, 1).item()


def has_nontrivial_top_k(candidate_count: int, k: int) -> bool:
    """Whether a top-k result has at least four times as many candidates as k."""
    return candidate_count >= 4 * k


def normalized_score_mse(approximate: torch.Tensor, full: torch.Tensor, epsilon: float = 1e-12) -> float:
    """Mean per-sample MSE normalized by that full-score row's variance."""
    approximate, full = _flatten_pairs(approximate, full)
    mse = (approximate - full).square().mean(-1)
    variance = full.var(-1, unbiased=False)
    return (mse / (variance + epsilon)).mean().item()


def gqa_kv_head_for_query(query_head: int, query_heads: int, kv_heads: int) -> int:
    """Map a Qwen-style grouped-query query head to its shared KV head."""
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query head count must be a positive multiple of KV head count")
    if not 0 <= query_head < query_heads:
        raise ValueError(f"query_head must be in [0, {query_heads})")
    return query_head // (query_heads // kv_heads)
