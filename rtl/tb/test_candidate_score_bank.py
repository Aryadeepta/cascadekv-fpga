"""Bit-exact persistent-score tests for candidate_score_bank."""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

SCORE_MIN = -(1 << 23)
SCORE_MAX = (1 << 23) - 1


def pack_lanes(values: list[int], width: int) -> int:
    return sum((value & ((1 << width) - 1)) << (index * width) for index, value in enumerate(values))


def reference(q: list[int], k: list[int], scale: int, previous: int) -> tuple[int, bool, bool]:
    contribution_raw = (sum(a * b for a, b in zip(q, k, strict=True)) * scale) << 1
    contribution = max(SCORE_MIN, min(SCORE_MAX, contribution_raw))
    contribution_saturated = contribution != contribution_raw
    total = previous + contribution
    score = max(SCORE_MIN, min(SCORE_MAX, total))
    return score, contribution_saturated, score != total


async def reset(dut: object) -> None:
    dut.update_valid.value = 0
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def update(
    dut: object, candidate_id: int, q: list[int], k: list[int], scale: int, clear: bool
) -> tuple[int, int, bool, bool]:
    """Issue one idle-only request and return its edge-N+2 result."""
    # A preceding call observes its result in ReadOnly; move to a writable
    # phase before driving the next request.
    await FallingEdge(dut.clk)
    assert int(dut.update_ready.value) == 1
    dut.candidate_id.value = candidate_id
    dut.q_flat.value = pack_lanes(q, 8)
    dut.k_flat.value = pack_lanes(k, 4)
    dut.scale_product.value = scale & 0xFFFF
    dut.clear_before_update.value = int(clear)
    dut.update_valid.value = 1
    await RisingEdge(dut.clk)  # N: accepted
    await ReadOnly()
    assert int(dut.update_ready.value) == 0
    await FallingEdge(dut.clk)
    dut.update_valid.value = 0
    await RisingEdge(dut.clk)  # N+1: arithmetic result captured
    await RisingEdge(dut.clk)  # N+2: write/result
    await ReadOnly()
    assert int(dut.result_valid.value) == 1
    assert int(dut.update_ready.value) == 1
    return (
        int(dut.result_candidate_id.value),
        dut.result_score.value.to_signed(),
        bool(dut.contribution_saturated.value),
        bool(dut.score_saturated.value),
    )


@cocotb.test()
async def update_ready_only_accepts_idle_requests(dut: object) -> None:
    """A request held during a busy interval must not become queued work."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    assert int(dut.update_ready.value) == 1

    q, k, scale = vectors(123)
    await FallingEdge(dut.clk)
    dut.candidate_id.value = 9
    dut.q_flat.value = pack_lanes(q, 8)
    dut.k_flat.value = pack_lanes(k, 4)
    dut.scale_product.value = scale & 0xFFFF
    dut.clear_before_update.value = 1
    dut.update_valid.value = 1
    await RisingEdge(dut.clk)  # Candidate 9 is accepted.
    await ReadOnly()
    assert int(dut.update_ready.value) == 0

    # Present candidate 10 while the bank is busy.  It is deliberately not
    # held until ready; therefore it must be ignored rather than queued.
    await FallingEdge(dut.clk)
    dut.candidate_id.value = 10
    dut.update_valid.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.update_ready.value) == 0
    await FallingEdge(dut.clk)
    dut.update_valid.value = 0

    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.result_valid.value) == 1
    assert int(dut.result_candidate_id.value) == 9
    assert int(dut.update_ready.value) == 1

    # There is no deferred result for the request presented while not ready.
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.result_valid.value) == 0
    assert int(dut.update_ready.value) == 1


def vectors(seed: int) -> tuple[list[int], list[int], int]:
    rng = random.Random(seed)
    return (
        [rng.randint(-128, 127) for _ in range(16)],
        [rng.randint(-8, 7) for _ in range(16)],
        rng.randint(-32768, 32767),
    )


async def checked_update(
    dut: object, memory: dict[int, int], candidate_id: int, seed: int, clear: bool
) -> None:
    q, k, scale = vectors(seed)
    previous = 0 if clear else memory.get(candidate_id, 0)
    expected_score, expected_contribution_sat, expected_score_sat = reference(q, k, scale, previous)
    actual = await update(dut, candidate_id, q, k, scale, clear)
    assert actual == (candidate_id, expected_score, expected_contribution_sat, expected_score_sat)
    memory[candidate_id] = expected_score


@cocotb.test()
async def candidate_state_sequences_and_saturation(dut: object) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    memory: dict[int, int] = {}

    # Independent candidates and a 16 -> 32 progressive update.
    await checked_update(dut, memory, 3, 3, True)
    await checked_update(dut, memory, 19, 19, True)
    await checked_update(dut, memory, 7, 70, True)
    first_score = memory[7]
    await checked_update(dut, memory, 7, 71, False)
    assert memory[7] != first_score or first_score in (SCORE_MIN, SCORE_MAX)
    assert 3 in memory and 19 in memory

    # Interleaved candidate streams through their second coordinate groups.
    for candidate_id, seed, clear in ((11, 110, True), (42, 420, True), (11, 111, False), (42, 421, False)):
        await checked_update(dut, memory, candidate_id, seed, clear)

    # 16 -> 32 -> 64 checkpoints: group 0, group 1, then groups 2 and 3.
    for candidate_id in (1, 5, 13, 31):
        for group in range(4):
            await checked_update(dut, memory, candidate_id, candidate_id * 100 + group, group == 0)

    # clear_before_update must discard an established value.
    await checked_update(dut, memory, 7, 999, True)

    # Saturation flags must accompany this specific candidate's result.
    q, k, scale = [127] * 16, [7] * 16, 32767
    score, contribution_sat, score_sat = reference(q, k, scale, 0)
    assert await update(dut, 55, q, k, scale, True) == (55, score, contribution_sat, score_sat)


@cocotb.test()
async def deterministic_randomized_score_memory(dut: object) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    rng = random.Random(0xCA5CADE)
    memory: dict[int, int] = {}
    for index in range(1000):
        candidate_id = rng.randrange(64)
        # A never-reset memory has no defined contents until a stage-16
        # clear establishes a candidate's first score.
        clear = candidate_id not in memory or rng.randrange(5) == 0
        q = [rng.randint(-128, 127) for _ in range(16)]
        k = [rng.randint(-8, 7) for _ in range(16)]
        scale = rng.randint(-32768, 32767)
        previous = 0 if clear else memory.get(candidate_id, 0)
        expected_score, expected_contribution_sat, expected_score_sat = reference(q, k, scale, previous)
        actual = await update(dut, candidate_id, q, k, scale, clear)
        assert actual == (candidate_id, expected_score, expected_contribution_sat, expected_score_sat), index
        memory[candidate_id] = expected_score
