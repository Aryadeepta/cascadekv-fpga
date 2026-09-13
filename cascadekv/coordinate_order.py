"""Calibration-derived, layer-shared coordinate orderings for partial dots.

All functions take a matrix of paired coordinate contributions, with one row
per calibration query/key pair and ``contributions[i, j] = q_j * k_j``.
This deliberately makes the calibration API independent of model capture and
prevents an ordering routine from reaching evaluation tensors.
"""

from __future__ import annotations

import torch


def validate_permutation(order: torch.Tensor | list[int], head_dim: int) -> None:
    """Raise if ``order`` is not a permutation of the coordinate range."""
    values = torch.as_tensor(order, dtype=torch.long).cpu()
    if values.ndim != 1 or values.numel() != head_dim:
        raise ValueError(f"ordering must contain exactly {head_dim} coordinates")
    if not torch.equal(values.sort().values, torch.arange(head_dim)):
        raise ValueError("ordering must be a permutation of [0, head_dim)")


def _check_contributions(contributions: torch.Tensor) -> torch.Tensor:
    if contributions.ndim != 2 or contributions.shape[0] == 0 or contributions.shape[1] == 0:
        raise ValueError("contributions must be a non-empty [pairs, head_dim] matrix")
    return contributions.float()


def contribution_energy(contributions: torch.Tensor) -> torch.Tensor:
    """Order by descending empirical ``E[(q_j * k_j)^2]``.

    Stable sorting makes ties deterministic, preserving a reproducible
    permutation even on degenerate calibration inputs.
    """
    contributions = _check_contributions(contributions)
    return contributions.square().mean(0).argsort(descending=True, stable=True)


def covariance_with_full_score(contributions: torch.Tensor) -> torch.Tensor:
    """Order by absolute Pearson covariance with the complete dot score.

    Precisely, coordinates are ranked by
    ``abs(cov(c_j, s)) / sqrt(var(c_j) * var(s) + eps)``, where
    ``s = sum_j c_j`` and population moments are used.  This is the absolute
    correlation, a scale-aware statistic that does not merely favor large
    coordinate magnitudes.
    """
    contributions = _check_contributions(contributions)
    centered = contributions - contributions.mean(0, keepdim=True)
    score = contributions.sum(1)
    score_centered = score - score.mean()
    covariance = (centered * score_centered.unsqueeze(1)).mean(0)
    variance = centered.square().mean(0)
    score_variance = score_centered.square().mean()
    statistic = covariance.abs() / (variance * score_variance + 1e-12).sqrt()
    return statistic.argsort(descending=True, stable=True)


def greedy_block_ordering(contributions: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Residual-greedy ordering, emitted as contiguous hardware-sized blocks.

    This avoids an infeasible search over all 16-coordinate combinations. At
    every individual selection it chooses the remaining coordinate with the
    greatest empirical reduction in unscaled residual score MSE:
    ``2 E[residual * c_j] - E[c_j^2]``.  The selected coordinates are simply
    grouped in successive ``block_size`` chunks in the resulting permutation.
    """
    contributions = _check_contributions(contributions)
    _, head_dim = contributions.shape
    if block_size <= 0 or head_dim % block_size:
        raise ValueError("block_size must be positive and divide head_dim")
    residual = contributions.sum(1).clone()
    remaining = torch.ones(head_dim, dtype=torch.bool, device=contributions.device)
    energies = contributions.square().mean(0)
    selected: list[int] = []
    for _ in range(head_dim):
        reductions = 2 * (contributions * residual.unsqueeze(1)).mean(0) - energies
        reductions = reductions.masked_fill(~remaining, -torch.inf)
        coordinate = int(reductions.argmax().item())
        selected.append(coordinate)
        residual -= contributions[:, coordinate]
        remaining[coordinate] = False
    order = torch.tensor(selected, dtype=torch.long)
    validate_permutation(order, head_dim)
    return order
