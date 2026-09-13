"""Symmetric integer Q/K quantization and progressive integer-dot scoring.

The scorer deliberately multiplies integer codes and accumulates them in
``int32`` before applying the corresponding Q and K scales.  It is therefore
not a fake-quantization float dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

QMAX = {4: 7, 8: 127}


@dataclass(frozen=True)
class QuantizedTensor:
    """Signed codes plus one symmetric scale per consecutive coordinate group."""

    values: torch.Tensor
    scales: torch.Tensor
    bits: int
    group_size: int

    @property
    def qmax(self) -> int:
        return QMAX[self.bits]

    def dequantize(self) -> torch.Tensor:
        groups = self.values.shape[-1] // self.group_size
        expanded = self.scales.repeat_interleave(self.group_size, dim=-1)
        assert expanded.shape[-1] == groups * self.group_size
        return self.values.float() * expanded


@dataclass(frozen=True)
class QuantizationDiagnostics:
    clipping_rate: float
    saturation_rate: float
    mean_absolute_error: float
    mean_squared_error: float


def quantize_symmetric(x: torch.Tensor, bits: int, group_size: int, epsilon: float = 1e-12) -> QuantizedTensor:
    """Quantize final-axis blocks using max-abs symmetric signed codes.

    Zero blocks get scale one, which makes their codes and reconstruction zero
    without a divide-by-zero special case.  A scale is intentionally retained
    for every group even when it is zero-valued in the source tensor.
    """
    if bits not in QMAX:
        raise ValueError("bits must be 4 or 8")
    if x.shape[-1] == 0 or group_size <= 0 or x.shape[-1] % group_size:
        raise ValueError("group_size must be positive and divide the final dimension")
    qmax = QMAX[bits]
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    max_abs = grouped.abs().amax(dim=-1)
    scales = torch.where(max_abs > epsilon, max_abs / qmax, torch.ones_like(max_abs))
    codes = torch.round(grouped / scales.unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int8)
    return QuantizedTensor(codes.reshape_as(x), scales, bits, group_size)


def quantization_diagnostics(x: torch.Tensor, quantized: QuantizedTensor) -> QuantizationDiagnostics:
    """Return code-endpoint incidence and reconstruction error diagnostics."""
    if x.shape != quantized.values.shape:
        raise ValueError("source and quantized tensor shapes must match")
    reconstructed = quantized.dequantize()
    expanded_scales = quantized.scales.repeat_interleave(quantized.group_size, dim=-1)
    unclamped_codes = torch.round(x.float() / expanded_scales)
    return QuantizationDiagnostics(
        clipping_rate=(unclamped_codes.abs() > quantized.qmax).float().mean().item(),
        saturation_rate=(quantized.values.abs() == quantized.qmax).float().mean().item(),
        mean_absolute_error=(reconstructed - x.float()).abs().mean().item(),
        mean_squared_error=(reconstructed - x.float()).square().mean().item(),
    )


def quantized_ordered_progressive_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    ordering: torch.Tensor | list[int],
    *,
    q_bits: int,
    k_bits: int,
    group_size: int,
    dimensions: tuple[int, ...] = (16, 32, 64, 128),
    unbiased: bool = True,
) -> tuple[dict[int, torch.Tensor], QuantizedTensor, QuantizedTensor]:
    """Score energy-reordered vectors through cumulative scaled integer dots."""
    if q.ndim != 1 or k.ndim != 2 or q.numel() != k.shape[-1]:
        raise ValueError("q must be [D] and k must be [keys, D]")
    head_dim = q.numel()
    if any(d <= 0 or d > head_dim for d in dimensions):
        raise ValueError("dimensions must lie within the head dimension")
    order = torch.as_tensor(ordering, dtype=torch.long, device=q.device)
    if order.ndim != 1 or order.numel() != head_dim or not torch.equal(
        order.sort().values.cpu(), torch.arange(head_dim)
    ):
        raise ValueError("ordering must be a head-dimension permutation")
    ordered_q, ordered_k = q.index_select(0, order), k.index_select(1, order)
    q_quantized = quantize_symmetric(ordered_q, q_bits, group_size)
    k_quantized = quantize_symmetric(ordered_k, k_bits, group_size)
    # Products and sums are integer operations; scaling starts only after each
    # hardware-sized dot accumulator has completed.
    q_codes = q_quantized.values.to(torch.int32)
    k_codes = k_quantized.values.to(torch.int32)
    terms: list[torch.Tensor] = []
    for start in range(0, head_dim, group_size):
        integer_dot = (k_codes[:, start : start + group_size] * q_codes[start : start + group_size]).sum(
            dim=-1, dtype=torch.int32
        )
        group = start // group_size
        terms.append(integer_dot.float() * q_quantized.scales[group] * k_quantized.scales[:, group])
    cumulative = torch.stack(terms, dim=-1).cumsum(dim=-1)

    def score_at(dimension: int) -> torch.Tensor:
        whole_groups, remainder = divmod(dimension, group_size)
        if remainder == 0:
            score = cumulative[:, whole_groups - 1]
        else:
            start = whole_groups * group_size
            integer_dot = (k_codes[:, start : start + remainder] * q_codes[start : start + remainder]).sum(
                dim=-1, dtype=torch.int32
            )
            partial = integer_dot.float() * q_quantized.scales[whole_groups] * k_quantized.scales[:, whole_groups]
            score = (cumulative[:, whole_groups - 1] if whole_groups else 0.0) + partial
        return score * (head_dim / dimension if unbiased else 1.0)

    return {dimension: score_at(dimension) for dimension in dimensions}, q_quantized, k_quantized
