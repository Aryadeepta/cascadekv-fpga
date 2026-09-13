"""Ready/valid and ordering tests for cascade_stage_controller."""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


def expected_work(stage: int, count: int, survivors: list[int]) -> list[tuple[int, int, int]]:
    groups = ((0,), (1,), (2, 3), (4, 5, 6, 7))[stage]
    ids = list(range(count)) if stage == 0 else survivors[:count]
    return [(candidate, group, int(stage == 0)) for candidate in ids for group in groups]


async def reset(dut: object) -> None:
    dut.start.value = 0
    dut.work_ready.value = 0
    dut.survivor_write_valid.value = 0
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def load_survivors(dut: object, survivors: list[int]) -> None:
    for index, candidate_id in enumerate(survivors):
        await FallingEdge(dut.clk)
        dut.survivor_write_index.value = index
        dut.survivor_write_candidate_id.value = candidate_id
        dut.survivor_write_valid.value = 1
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.survivor_write_valid.value = 0


async def start_stage(dut: object, stage: int, count: int) -> None:
    await FallingEdge(dut.clk)
    dut.stage.value = stage
    dut.stage_candidate_count.value = count
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0


async def collect_work(
    dut: object, rng: random.Random | None = None
) -> list[tuple[int, int, int]]:
    """Collect one stage, checking that stalled payloads never change."""
    actual: list[tuple[int, int, int]] = []
    stall_payload: tuple[int, int, int] | None = None
    for _ in range(10000):
        await FallingEdge(dut.clk)
        ready = 1 if rng is None else int(rng.randrange(4) != 0)
        dut.work_ready.value = ready
        valid = int(dut.work_valid.value)
        payload = (
            int(dut.work_candidate_id.value),
            int(dut.work_group_index.value),
            int(dut.work_clear_before_update.value),
        )
        if valid and ready:
            actual.append(payload)
        if valid and not ready:
            if stall_payload is None:
                stall_payload = payload
            else:
                assert payload == stall_payload
        else:
            stall_payload = None

        await RisingEdge(dut.clk)
        await ReadOnly()
        # This was a stalled transfer, so the producer cannot have advanced.
        if valid and not ready:
            assert int(dut.work_valid.value) == 1
            assert (
                int(dut.work_candidate_id.value),
                int(dut.work_group_index.value),
                int(dut.work_clear_before_update.value),
            ) == payload
        if int(dut.done.value):
            assert int(dut.busy.value) == 0
            assert int(dut.work_valid.value) == 0
            return actual
    raise AssertionError("stage did not complete")


@cocotb.test()
async def ordered_dense_and_survivor_stages(dut: object) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    survivors = [19, 2, 87, 5]
    await load_survivors(dut, survivors)

    await start_stage(dut, 0, 8)
    assert await collect_work(dut) == expected_work(0, 8, survivors)

    await start_stage(dut, 1, len(survivors))
    assert await collect_work(dut) == expected_work(1, len(survivors), survivors)

    await start_stage(dut, 2, len(survivors))
    assert await collect_work(dut) == expected_work(2, len(survivors), survivors)

    await start_stage(dut, 3, len(survivors))
    assert await collect_work(dut) == expected_work(3, len(survivors), survivors)


@cocotb.test()
async def backpressure_and_original_ids(dut: object) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    survivors = [201, 4, 113, 72]
    await load_survivors(dut, survivors)
    await start_stage(dut, 3, len(survivors))
    assert await collect_work(dut, random.Random(0xC05CADE)) == expected_work(3, len(survivors), survivors)


@cocotb.test()
async def zero_candidates_and_randomized_reference(dut: object) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    dut.work_ready.value = 1
    await start_stage(dut, 2, 0)
    await ReadOnly()
    assert int(dut.done.value) == 1
    assert int(dut.busy.value) == 0
    assert int(dut.work_valid.value) == 0

    rng = random.Random(0x5CA1E)
    for scenario in range(100):
        stage = rng.randrange(4)
        count = rng.randrange(0, 17)
        survivors = [rng.randrange(256) for _ in range(count)]
        await load_survivors(dut, survivors)
        await start_stage(dut, stage, count)
        actual = await collect_work(dut, rng) if count else []
        if count == 0:
            await ReadOnly()
            assert int(dut.done.value) == 1
        assert actual == expected_work(stage, count, survivors), scenario
