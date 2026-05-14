# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import time
from typing import Any, Iterator

import pytest

from thinkingbox.common.ordered_parallel_executor import (
    Worker,
    WorkResult,
    iter_map_parallel_ordered,
)


class DummyWorker(Worker[dict[str, Any], str]):
    """
    Work item shape:
      {
        "id": int | str,                 # identifier included in the result string
        "delay": float = 0.0,            # seconds to await before producing output
        "raise_exc": bool = False,       # if True, raise RuntimeError instead of returning
        "error_result": bool = False,    # if True, return (True, result) but do not raise
        "payload": str | None = None,    # custom payload string; default f"done-{id}"
      }
    """

    async def work(self, work: dict[str, Any]) -> WorkResult[str]:
        await asyncio.sleep(work.get("delay", 0.0))
        if work.get("raise_exc", False):
            raise RuntimeError(f"boom at {work.get('id')}")
        payload = work.get("payload", f"done-{work.get('id')}")
        is_error = work.get("error_result", False)
        return WorkResult(result=payload, is_system_error=is_error)

    def get_error_result(self, work: dict[str, Any]) -> str:
        return f"error-{work.get('id')}"


def _iter(items: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    # Helper to provide a real iterator as required by the executor API
    return iter(items)


@pytest.mark.asyncio
async def test_order_preserved_despite_parallelism():
    # Arrange: deliberately set delays so that tasks complete out-of-order,
    # but the executor must yield strictly in input order.
    COUNT = 40
    items = [
        {"id": 0, "delay": 0.05},
        {"id": 1, "delay": 0.01},
        {"id": 2, "delay": 0.00},
        {"id": 3, "delay": 0.02},
    ]
    for i in range(4, COUNT):
        items.append({"id": i, "delay": 0.001})

    worker = DummyWorker()

    # Act
    aiter = iter_map_parallel_ordered(
        it=_iter(items),
        max_parallelism=3,
        max_input_queue=10,
        max_output_queue=3,
        worker=worker,
    )
    results = [r async for r in aiter]
    expected_results = [f"done-{i}" for i in range(COUNT)]

    # Assert: yielded strictly in input order
    assert results == expected_results


@pytest.mark.asyncio
async def test_exception_stops_early():
    # Arrange: index 2 raises; executor should yield items 0 and 1, then raise.
    items = [
        {"id": 0, "delay": 0.0},
        {"id": 1, "delay": 0.0},
        {"id": 2, "delay": 0.0, "raise_exc": True},  # first failure in order
        {
            "id": 3,
            "delay": 0.0,
        },  # would be computed but never yielded due to strict order
    ]
    for i in range(4, 40):
        items.append({"id": i, "delay": 0.0})
    worker = DummyWorker()

    async def collect_until_exc():
        out = []
        aiter = iter_map_parallel_ordered(
            it=_iter(items),
            max_parallelism=4,
            max_input_queue=10,
            max_output_queue=10,
            worker=worker,
        )
        try:
            async for r in aiter:
                out.append(r)
        except Exception as e:
            return out, e
        return out, None

    # Act
    results, exc = await collect_until_exc()

    # Assert
    assert results == ["done-0", "done-1"]
    assert isinstance(exc, RuntimeError)
    assert "boom at 2" in str(exc)


@pytest.mark.asyncio
async def test_error_results_are_yielded_not_raised():
    # Arrange: some tasks mark their result as "error_result=True"
    items = [
        {"id": 0, "error_result": False},
        {"id": 1, "error_result": True},  # should still be yielded
        {"id": 2, "error_result": False},
    ]
    worker = DummyWorker()

    # Act
    aiter = iter_map_parallel_ordered(
        it=_iter(items),
        max_parallelism=2,
        max_input_queue=10,
        max_output_queue=2,
        worker=worker,
    )
    results = [r async for r in aiter]

    # Assert: all results yielded in order, regardless of error_result flag
    assert results == ["done-0", "done-1", "done-2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,err_msg_part",
    [
        (
            dict(max_parallelism=0, max_input_queue=1, max_output_queue=1),
            "max_parallelism",
        ),
        (
            dict(max_parallelism=1, max_input_queue=0, max_output_queue=1),
            "max_input_queue",
        ),
        (
            dict(max_parallelism=1, max_input_queue=1, max_output_queue=0),
            "max_output_queue",
        ),
    ],
)
async def test_invalid_parameters_raise_valueerror(kwargs, err_msg_part):
    worker = DummyWorker()
    items = [{"id": 0}]
    # Note: async generator body (and validation) starts on first iteration
    with pytest.raises(ValueError) as ei:
        aiter = iter_map_parallel_ordered(
            it=_iter(items),
            worker=worker,
            **kwargs,
        )
        # Trigger execution to hit validation
        await anext(aiter)
    assert err_msg_part in str(ei.value)


@pytest.mark.asyncio
async def test_watchdog_cancels_stuck_task():
    """
    A single stuck task should be canceled by the watchdog and yield an error result
    produced by get_error_result, within a short time window.
    """

    class HangingWorker(Worker[int, str]):
        async def work(self, work: int) -> WorkResult[str]:
            if work == 0:
                # long hang; watchdog must cancel this
                await asyncio.sleep(300.0)
            return WorkResult(result=f"done-{work}")

        def get_error_result(self, work: int) -> str:
            # Mark as error result so stats/errors path is exercised
            return f"error-{work}"

    worker = HangingWorker()
    items = list(range(4))

    start = time.monotonic()
    aiter = iter_map_parallel_ordered(
        it=iter(items),
        max_parallelism=1,
        max_input_queue=1,
        max_output_queue=1,
        worker=worker,
        watchdog_timeout=0.4,  # small timeout to trigger quickly
    )

    results = [r async for r in aiter]
    elapsed = time.monotonic() - start

    # The hung task should have been canceled and replaced by error result
    assert results == [
        "error-0",
        "done-1",
        "done-2",
        "done-3",
    ]
    # Should complete fast (adaptive sleep keeps this under ~1s)
    assert elapsed < 5.0, f"Watchdog took too long: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_watchdog_disabled():
    """When watchdog_timeout=None, long tasks should complete normally."""

    class SlowWorker(Worker[int, str]):
        async def work(self, work: int) -> WorkResult[str]:
            if work == 0:
                await asyncio.sleep(10.0)  # Longer than typical test
            return WorkResult(result=f"done-{work}")

        def get_error_result(self, work: int) -> str:
            return f"error-{work}"

    worker = SlowWorker()
    items = [0, 1, 2]

    aiter = iter_map_parallel_ordered(
        it=iter(items),
        max_parallelism=1,
        max_input_queue=1,
        max_output_queue=1,
        worker=worker,
        watchdog_timeout=None,  # Disabled
    )

    results = [r async for r in aiter]

    # All tasks complete normally (not canceled)
    assert results == ["done-0", "done-1", "done-2"]
