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


def unsigned_scale_codes(value: torch.Tensor, fractional_bits: int) -> tuple[torch.Tensor, float]:
    """Encode nonnegative scales in an unsigned 16-bit power-of-two format."""
    if not 0 <= fractional_bits <= 16:
        raise ValueError("fractional_bits must be in [0, 16]")
    unbounded = torch.floor(value.float() * (1 << fractional_bits) + 0.5).to(torch.int64)
    saturated = (unbounded < 0) | (unbounded > 0xFFFF)
    return unbounded.clamp(0, 0xFFFF), saturated.float().mean().item() if value.numel() else 0.0


def factorized_scale_product_codes(
    q_scale_code: torch.Tensor, k_scale_code: torch.Tensor, shift: int = 13
) -> tuple[torch.Tensor, float]:
    """Return saturated positive Q4.12 codes from two unsigned scale codes.

    U0.16 times U7.9 has 25 fractional bits, hence ``shift=13`` converts its
    integer product to Q4.12 using nonnegative round-to-nearest.
    """
    if shift < 1:
        raise ValueError("shift must be positive for this rounding contract")
    wide = q_scale_code.to(torch.int64) * k_scale_code.to(torch.int64)
    rounded = (wide + (1 << (shift - 1))) >> shift
    saturated = rounded > 0x7FFF
    return rounded.clamp(0, 0x7FFF), saturated.float().mean().item() if rounded.numel() else 0.0


# These constants intentionally mirror rtl/{scale_product,progressive_dot}.sv.
SCORE_MIN = -(1 << 23)
SCORE_MAX = (1 << 23) - 1


def scale_product_rtl_codes(q_scale: torch.Tensor, k_scale: torch.Tensor) -> torch.Tensor:
    """Bit-exact U0.16 x U7.9 -> signed Q4.12 RTL scale-product codes."""
    q_code, _ = unsigned_scale_codes(q_scale, 16)
    k_code, _ = unsigned_scale_codes(k_scale, 9)
    return factorized_scale_product_codes(q_code, k_code)[0]


def progressive_dot_rtl_update(
    q_codes: torch.Tensor, k_codes: torch.Tensor, scale_product_code: torch.Tensor, score_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One exact `progressive_dot.sv` update (dot, contribution, score).

    Inputs may be batched, with the final dimension containing the 16 signed
    lanes.  This is integer-only: there is no fake dequantized dot anywhere in
    the datapath model.
    """
    dot = (q_codes.to(torch.int64) * k_codes.to(torch.int64)).sum(dim=-1)
    contribution = (dot * scale_product_code.to(torch.int64)) << 1
    contribution = contribution.clamp(SCORE_MIN, SCORE_MAX)
    score = (score_in.to(torch.int64) + contribution).clamp(SCORE_MIN, SCORE_MAX)
    return dot, contribution, score


def q8k4_rtl_score_codes(q8: object, k4: object) -> torch.Tensor:
    """Return exact eight-group Q11.13 scores for Q8 and K4 encodings.

    ``q8`` is one 128-D group-16 Q8 tensor and ``k4`` is ``[N,128]`` K4.
    Scales are first encoded in the actual U0.16/U7.9 ports, then every group
    takes the same saturating path as an RTL progressive-dot instance.
    """
    # Kept duck-typed to avoid a circular dependency on quantize.py.
    q_values, q_scales = q8.values, q8.scales
    k_values, k_scales = k4.values, k4.scales
    if q_values.ndim != 1 or k_values.ndim != 2 or q_values.numel() != k_values.shape[1]:
        raise ValueError("expected one Q vector and [N,D] K vectors")
    group_size = q8.group_size
    if group_size != 16 or k4.group_size != 16 or q_values.numel() % 16:
        raise ValueError("the frozen RTL path requires group-16 vectors")
    score = torch.zeros(k_values.shape[0], dtype=torch.int64, device=k_values.device)
    for group, start in enumerate(range(0, q_values.numel(), 16)):
        q_group = q_values[start : start + 16].expand(k_values.shape[0], -1)
        k_group = k_values[:, start : start + 16]
        scale = scale_product_rtl_codes(q_scales[group], k_scales[:, group])
        _, _, score = progressive_dot_rtl_update(q_group, k_group, scale, score)
    return score


def q8k4_rtl_score(q8: object, k4: object) -> torch.Tensor:
    """Exact RTL score codes converted from signed Q11.13 to real units."""
    return q8k4_rtl_score_codes(q8, k4).float() / float(1 << 13)


def q8k4_rtl_error_cushion(q8: object, k4: object) -> torch.Tensor:
    """A deterministic one-sided cushion for dequantized Q8xK4 dot error.

    It bounds ``qhat @ rhat - rtl_score`` for each row.  The first term bounds
    the independently rounded U0.16/U7.9 scale products using exact integer
    dot magnitudes.  The second accounts for RTL contribution/accumulator
    saturation.  It is deliberately per query/prototype, so no unproven
    global statistical margin is presented as a safety guarantee.
    """
    q_values, q_scales = q8.values, q8.scales
    k_values, k_scales = k4.values, k4.scales
    n = k_values.shape[0]
    linear_hardware = torch.zeros(n, dtype=torch.float64, device=k_values.device)
    scale_error = torch.zeros_like(linear_hardware)
    for group, start in enumerate(range(0, q_values.numel(), 16)):
        dots = (q_values[start : start + 16].to(torch.int64).expand(n, -1)
                * k_values[:, start : start + 16].to(torch.int64)).sum(dim=-1)
        actual = q_scales[group].double() * k_scales[:, group].double()
        hardware = scale_product_rtl_codes(q_scales[group], k_scales[:, group]).double() / 4096.0
        linear_hardware += dots.double() * hardware
        scale_error += dots.abs().double() * (actual - hardware).abs()
    rtl = q8k4_rtl_score_codes(q8, k4).double() / 8192.0
    # Saturation can only be safely handled by paying the observed difference
    # from the corresponding unsaturated hardware sum.
    return (scale_error + (linear_hardware - rtl).abs()).float()
