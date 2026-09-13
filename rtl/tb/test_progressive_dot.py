"""Bit-exact cocotb checks for the combinational progressive score update."""

from __future__ import annotations

import random

import cocotb
import torch
from cocotb.triggers import Timer

from cascadekv.fixed_point import (
    FixedFormat,
    fixed_accumulate,
    float_to_fixed,
    integer_dot_times_scale,
)
from cascadekv.quantize import quantize_symmetric

SCALE_FORMAT = FixedFormat(16, 4)  # Q4.12
SCORE_FORMAT = FixedFormat(24, 11)  # Q11.13
SCORE_MIN = -(1 << 23)
SCORE_MAX = (1 << 23) - 1


def pack_lanes(values: list[int], width: int) -> int:
    return sum((value & ((1 << width) - 1)) << (index * width) for index, value in enumerate(values))


def saturate(value: int) -> tuple[int, bool]:
    return max(SCORE_MIN, min(SCORE_MAX, value)), value < SCORE_MIN or value > SCORE_MAX


def reference(q: list[int], k: list[int], scale_code: int, score_code: int) -> tuple[int, int, int, bool, bool]:
    dot = sum(q_i * k_i for q_i, k_i in zip(q, k, strict=True))
    # dot * Q4.12 has 12 fractional bits; left-shift once for Q11.13.
    contribution, contribution_saturated = saturate((dot * scale_code) << 1)
    score_out, score_saturated = saturate(score_code + contribution)
    return dot, contribution, score_out, contribution_saturated, score_saturated


async def check_update(dut: object, q: list[int], k: list[int], scale_code: int, score_code: int) -> None:
    dut.q_flat.value = pack_lanes(q, 8)
    dut.k_flat.value = pack_lanes(k, 4)
    dut.scale_product.value = scale_code & 0xFFFF
    dut.score_in.value = score_code & 0xFFFFFF
    await Timer(1, unit="ns")
    expected = reference(q, k, scale_code, score_code)
    actual = (
        dut.integer_dot.value.to_signed(),
        dut.contribution.value.to_signed(),
        dut.score_out.value.to_signed(),
        int(dut.contribution_saturated.value),
        int(dut.score_saturated.value),
    )
    assert actual == expected, (q, k, scale_code, score_code, expected, actual)


@cocotb.test()
async def directed_extrema_and_random_updates(dut: object) -> None:
    cases = [
        ([0] * 16, [0] * 16, 0, 0),
        ([1] * 16, [1] * 16, 4096, 0),
        ([1] * 16, [-1] * 16, 4096, 0),
        ([1] * 16, [1] * 16, 2048, 100 << 13),
        ([1] * 16, [-1] * 16, 2048, -(100 << 13)),
        ([1] * 16, [-1] * 16, 4096, 16 << 13),
        ([127] * 16, [7] * 16, 4096, 0),
        ([-128] * 16, [-8] * 16, 4096, 0),
        ([127, -128] * 8, [7, -8] * 8, 0, 12345),
        ([127] * 16, [7] * 16, 4096, SCORE_MAX),
        ([127] * 16, [-8] * 16, 4096, SCORE_MIN),
        ([127] * 16, [7] * 16, 32767, 0),
        ([-128] * 16, [7] * 16, 32767, 0),
    ]
    for case in cases:
        await check_update(dut, *case)

    rng = random.Random(0xCA5CADE)
    for _ in range(1000):
        await check_update(
            dut,
            [rng.randint(-128, 127) for _ in range(16)],
            [rng.randint(-8, 7) for _ in range(16)],
            rng.randint(-32768, 32767),
            rng.randint(SCORE_MIN, SCORE_MAX),
        )


@cocotb.test()
async def quantizer_and_fixed_point_helpers_match_rtl(dut: object) -> None:
    """Use frozen Q8xK4 quantization plus characterization helper arithmetic."""
    generator = torch.Generator().manual_seed(20260913)
    q_source, k_source = torch.randn(16, generator=generator), torch.randn(16, generator=generator)
    q_quantized = quantize_symmetric(q_source, bits=8, group_size=16)
    k_quantized = quantize_symmetric(k_source, bits=4, group_size=16)
    scale = float_to_fixed(q_quantized.scales * k_quantized.scales, SCALE_FORMAT)
    q, k, scale_code = q_quantized.values.tolist(), k_quantized.values.tolist(), int(scale.codes.item())
    dot = sum(q_i * k_i for q_i, k_i in zip(q, k, strict=True))
    expected_contribution = integer_dot_times_scale(torch.tensor([dot]), scale, SCORE_FORMAT)
    expected_score = fixed_accumulate(expected_contribution, SCORE_FORMAT)
    await check_update(dut, q, k, scale_code, 0)
    assert expected_contribution.codes.tolist() == [reference(q, k, scale_code, 0)[1]]
    assert expected_score.codes.tolist() == [reference(q, k, scale_code, 0)[2]]


@cocotb.test()
async def eight_group_progressive_sequence_matches_python(dut: object) -> None:
    """Check exact 16/32/64/128-D checkpoints through successive RTL updates."""
    generator = torch.Generator().manual_seed(616128)
    q_source, k_source = torch.randn(128, generator=generator), torch.randn(128, generator=generator)
    q_quantized = quantize_symmetric(q_source, bits=8, group_size=16)
    k_quantized = quantize_symmetric(k_source, bits=4, group_size=16)
    scales = float_to_fixed(q_quantized.scales * k_quantized.scales, SCALE_FORMAT)
    dots = []
    for group in range(8):
        start = group * 16
        dots.append(sum(int(a) * int(b) for a, b in zip(q_quantized.values[start:start + 16], k_quantized.values[start:start + 16], strict=True)))
    expected_contributions = integer_dot_times_scale(torch.tensor(dots), scales, SCORE_FORMAT)
    expected_scores = fixed_accumulate(
        type(expected_contributions)(expected_contributions.codes.reshape(1, 8), SCORE_FORMAT, expected_contributions.clipping_rate),
        SCORE_FORMAT,
    ).codes[0].tolist()

    score = 0
    for group in range(8):
        start = group * 16
        q = q_quantized.values[start:start + 16].tolist()
        k = k_quantized.values[start:start + 16].tolist()
        await check_update(dut, q, k, int(scales.codes[group]), score)
        score = int(dut.score_out.value.to_signed())
        if group in (0, 1, 3, 7):
            assert score == expected_scores[group], (group, score, expected_scores[group])
