"""Small, deterministic signed power-of-two fixed-point utilities.

These helpers model the arithmetic boundary proposed for the routing datapath:
integer dot products multiply a quantized scale product, then accumulate in a
wider signed score register.  Values are intentionally represented as integer
codes plus an explicit fractional-bit count rather than pretending that the
hardware performs floating point arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FixedFormat:
    """Signed Q(I-F).F format, where ``integer_bits`` includes the sign bit."""

    total_bits: int
    integer_bits: int

    def __post_init__(self) -> None:
        if not 1 <= self.integer_bits <= self.total_bits:
            raise ValueError("integer_bits must be in [1, total_bits]")

    @property
    def fractional_bits(self) -> int:
        return self.total_bits - self.integer_bits

    @property
    def minimum_code(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def maximum_code(self) -> int:
        return (1 << (self.total_bits - 1)) - 1


@dataclass(frozen=True)
class FixedConversion:
    codes: torch.Tensor
    format: FixedFormat
    clipping_rate: float

    def dequantize(self) -> torch.Tensor:
        return self.codes.float() / float(1 << self.format.fractional_bits)


def round_nearest_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    """Round ties away from zero, a simple deterministic hardware contract."""
    return value.sign() * torch.floor(value.abs() + 0.5)


def float_to_fixed(value: torch.Tensor, format: FixedFormat) -> FixedConversion:
    """Quantize floats to signed fixed point with saturating round-to-nearest."""
    unbounded = round_nearest_away_from_zero(value.float() * (1 << format.fractional_bits))
    clipping = (unbounded < format.minimum_code) | (unbounded > format.maximum_code)
    return FixedConversion(
        codes=unbounded.clamp(format.minimum_code, format.maximum_code).to(torch.int64),
        format=format,
        clipping_rate=clipping.float().mean().item() if clipping.numel() else 0.0,
    )


def fixed_multiply(left: FixedConversion, right: FixedConversion, output: FixedFormat) -> FixedConversion:
    """Multiply two fixed values and requantize to ``output`` with saturation."""
    product = left.codes * right.codes
    shift = left.format.fractional_bits + right.format.fractional_bits - output.fractional_bits
    scaled = product.float() / float(1 << shift) if shift >= 0 else product * (1 << -shift)
    return float_to_fixed(scaled / float(1 << output.fractional_bits), output)


def integer_dot_times_scale(
    integer_dot: torch.Tensor, scale: FixedConversion, output: FixedFormat
) -> FixedConversion:
    """Model ``int dot * fixed scale -> fixed contribution`` without float math."""
    shift = output.fractional_bits - scale.format.fractional_bits
    raw = integer_dot.to(torch.int64) * scale.codes
    if shift >= 0:
        raw = raw * (1 << shift)
        overflow = (raw < output.minimum_code) | (raw > output.maximum_code)
        codes = raw.clamp(output.minimum_code, output.maximum_code)
    else:
        rounded = round_nearest_away_from_zero(raw.float() / float(1 << -shift)).to(torch.int64)
        overflow = (rounded < output.minimum_code) | (rounded > output.maximum_code)
        codes = rounded.clamp(output.minimum_code, output.maximum_code)
    return FixedConversion(codes, output, overflow.float().mean().item() if overflow.numel() else 0.0)


def fixed_accumulate(contributions: FixedConversion, output: FixedFormat) -> FixedConversion:
    """Cumulatively add final-axis contributions with saturating accumulator width."""
    shift = output.fractional_bits - contributions.format.fractional_bits
    raw = contributions.codes * (1 << shift) if shift >= 0 else round_nearest_away_from_zero(
        contributions.codes.float() / float(1 << -shift)
    ).to(torch.int64)
    cumulative = raw.cumsum(dim=-1)
    overflow = (cumulative < output.minimum_code) | (cumulative > output.maximum_code)
    return FixedConversion(cumulative.clamp(output.minimum_code, output.maximum_code), output,
                           overflow.float().mean().item() if overflow.numel() else 0.0)
