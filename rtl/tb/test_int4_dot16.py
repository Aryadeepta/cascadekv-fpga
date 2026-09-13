"""Cocotb verification for the 16-lane signed INT8 x INT4 dot primitive."""

from __future__ import annotations

import random

import cocotb
import torch
from cocotb.triggers import Timer

from cascadekv.quantize import quantize_symmetric


def pack_lanes(values: list[int], width: int) -> int:
    """Pack signed lane values using the primitive's least-significant-first convention."""
    mask = (1 << width) - 1
    return sum((value & mask) << (index * width) for index, value in enumerate(values))


async def check_vector(dut: object, q: list[int], k: list[int]) -> None:
    dut.q_flat.value = pack_lanes(q, 8)
    dut.k_flat.value = pack_lanes(k, 4)
    await Timer(1, unit="ns")
    expected = sum(q_i * k_i for q_i, k_i in zip(q, k, strict=True))
    actual = dut.dot.value.to_signed()
    assert actual == expected, (q, k, expected, actual)


@cocotb.test()
async def directed_and_random_vectors(dut: object) -> None:
    cases = [
        ([0] * 16, [0] * 16),
        ([1] * 16, [1] * 16),
        ([1 if i % 2 == 0 else -1 for i in range(16)], [-1 if i % 2 == 0 else 1 for i in range(16)]),
        ([127] * 16, [7] * 16),
        ([-127] * 16, [-7] * 16),
        ([127, -128, 127, -128] * 4, [7, -8, -8, 7] * 4),
        ([-128] * 16, [-8] * 16),
        ([127] * 16, [-8] * 16),
    ]
    for q, k in cases:
        await check_vector(dut, q, k)

    rng = random.Random(0xCA5CADE)
    for _ in range(1000):
        await check_vector(
            dut,
            [rng.randint(-128, 127) for _ in range(16)],
            [rng.randint(-8, 7) for _ in range(16)],
        )


@cocotb.test()
async def quantized_values_match_python_integer_dot(dut: object) -> None:
    """Exercise codes made by the frozen software quantizer, without applying scales."""
    generator = torch.Generator().manual_seed(20260913)
    q_source = torch.randn(16, generator=generator)
    k_source = torch.randn(16, generator=generator)
    q_codes = quantize_symmetric(q_source, bits=8, group_size=16).values.tolist()
    k_codes = quantize_symmetric(k_source, bits=4, group_size=16).values.tolist()
    await check_vector(dut, q_codes, k_codes)
