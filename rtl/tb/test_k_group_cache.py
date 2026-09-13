"""Bit-exact cocotb tests for the ordered INT4 K-group routing cache."""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


async def reset(dut: object) -> None:
    dut.write_valid.value = 0
    dut.request_valid.value = 0
    dut.response_ready.value = 0
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def write_entry(dut: object, candidate: int, group: int, k_flat: int, scale: int) -> None:
    await FallingEdge(dut.clk)
    dut.write_candidate_id.value = candidate
    dut.write_group_index.value = group
    dut.write_k_flat.value = k_flat
    dut.write_scale.value = scale & 0xFFFF
    dut.write_valid.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.write_valid.value = 0


async def read_entry(dut: object, candidate: int, group: int) -> tuple[int, int, int, int]:
    """Accept a request then consume its next-cycle response."""
    await FallingEdge(dut.clk)
    assert int(dut.request_ready.value) == 1
    dut.candidate_id.value = candidate
    dut.group_index.value = group
    dut.request_valid.value = 1
    dut.response_ready.value = 0
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.request_valid.value = 0
    await ReadOnly()
    assert int(dut.response_valid.value) == 1
    actual = (
        int(dut.response_candidate_id.value),
        int(dut.response_group_index.value),
        int(dut.response_k_flat.value),
        int(dut.response_scale.value) & 0xFFFF,
    )
    await FallingEdge(dut.clk)
    dut.response_ready.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.response_valid.value) == 0
    await FallingEdge(dut.clk)
    dut.response_ready.value = 0
    return actual


@cocotb.test()
async def single_groups_candidates_boundaries_and_overwrite(dut: object) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    await write_entry(dut, 3, 5, 0xFEDC_BA98_7654_3210, -1234)
    assert await read_entry(dut, 3, 5) == (3, 5, 0xFEDC_BA98_7654_3210, 0xFB2E)

    # Every group of one candidate must have a separate location.
    for group in range(8):
        await write_entry(dut, 17, group, 0x1100_0000_0000_0000 + group, group * 101)
    for group in range(8):
        assert await read_entry(dut, 17, group) == (17, group, 0x1100_0000_0000_0000 + group, group * 101)

    # Same group across candidates must not alias.
    for candidate in (1, 2, 31, 128, 254):
        await write_entry(dut, candidate, 6, candidate * 0x0101_0101_0101_0101, -candidate)
    for candidate in (1, 2, 31, 128, 254):
        assert await read_entry(dut, candidate, 6) == (
            candidate, 6, candidate * 0x0101_0101_0101_0101, (-candidate) & 0xFFFF
        )

    boundaries = ((0, 0), (0, 7), (255, 0), (255, 7))
    for index, (candidate, group) in enumerate(boundaries):
        await write_entry(dut, candidate, group, 0xABC0_0000_0000_0000 + index, -32768 + index)
    for index, (candidate, group) in enumerate(boundaries):
        assert await read_entry(dut, candidate, group) == (
            candidate, group, 0xABC0_0000_0000_0000 + index, (-32768 + index) & 0xFFFF
        )

    await write_entry(dut, 3, 5, 0x1111_2222_3333_4444, 11)
    await write_entry(dut, 3, 5, 0x9999_AAAA_BBBB_CCCC, -22)
    assert await read_entry(dut, 3, 5) == (3, 5, 0x9999_AAAA_BBBB_CCCC, 0xFFEA)


@cocotb.test()
async def signed_nibbles_random_reads_ordering_and_backpressure(dut: object) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    signed_lanes = [-8, -7, -1, 0, 1, 7] * 2 + [-8, 7, -1, 1]
    signed_payload = sum((lane & 0xF) << (lane_index * 4) for lane_index, lane in enumerate(signed_lanes))
    await write_entry(dut, 44, 2, signed_payload, -1)
    assert await read_entry(dut, 44, 2) == (44, 2, signed_payload, 0xFFFF)

    rng = random.Random(0xCACEC0DE)
    memory: dict[tuple[int, int], tuple[int, int]] = {}
    for _ in range(400):
        candidate, group = rng.randrange(256), rng.randrange(8)
        value = (rng.getrandbits(64), rng.randrange(1 << 16))
        memory[candidate, group] = value
        await write_entry(dut, candidate, group, *value)

    # At least 1000 deterministic randomized reads, checked bit-exactly.
    keys = list(memory)
    for read_number in range(1000):
        candidate, group = rng.choice(keys)
        k_flat, scale = memory[candidate, group]
        assert await read_entry(dut, candidate, group) == (candidate, group, k_flat, scale), read_number

    # A stalled response must retain payload and its original request tags.
    candidate, group = keys[0]
    k_flat, scale = memory[candidate, group]
    await FallingEdge(dut.clk)
    dut.candidate_id.value = candidate
    dut.group_index.value = group
    dut.request_valid.value = 1
    dut.response_ready.value = 0
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.request_valid.value = 0
    await ReadOnly()
    expected = (candidate, group, k_flat, scale)
    for _ in range(3):
        assert int(dut.response_valid.value) == 1
        assert (
            int(dut.response_candidate_id.value), int(dut.response_group_index.value),
            int(dut.response_k_flat.value), int(dut.response_scale.value) & 0xFFFF,
        ) == expected
        assert int(dut.request_ready.value) == 0
        await RisingEdge(dut.clk)
        await ReadOnly()
    await FallingEdge(dut.clk)
    dut.response_ready.value = 1
    await RisingEdge(dut.clk)
