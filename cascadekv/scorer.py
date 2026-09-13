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
