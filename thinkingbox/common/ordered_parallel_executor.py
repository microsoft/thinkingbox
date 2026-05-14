# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import contextlib
import copy
import dataclasses
import enum
import logging
import os
import signal
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Generic, Iterator, NamedTuple, TypeVar

from thinkingbox.common.utils import enter_async_context_if_not_none

logger = logging.getLogger()


Work = TypeVar("Work")
Result = TypeVar("Result")


class WorkResult(NamedTuple, Generic[Result]):
    result: Result
    is_system_error: bool = False
    is_correct: bool = False


class Worker(ABC, Generic[Work, Result]):
    @abstractmethod
    async def work(self, work: Work) -> WorkResult[Result]: ...

    @abstractmethod
    def get_error_result(self, work: Work) -> Result: ...


class Status(enum.Enum):
    OK = "ok"
    ERROR = "error"


PID = os.getpid()


class SignalEvent:
    def __init__(self, signal_name: str):
        self._flag = False
        self._installed = False
        self._signal_name = signal_name

    def handler(self):
        self._flag = True

    def install(self):
        if self._installed:
            return
        self._installed = True
        sig = getattr(signal, self._signal_name)
        if sig is None:
            return
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR1, self.handler)

    def get(self):
        value = self._flag
        if value:
            self._flag = False
        return value


@dataclasses.dataclass
class ExecutorProgress:
    started: int = 0
    finished: int = 0
    errors: int = 0
    correct: int = 0
    queued_out: int = 0


@dataclasses.dataclass
class ExecutorResult:
    status: Status
    is_error_result: bool = False
    is_correct: bool = False
    result: Any = None
    exception: BaseException | None = None


def report_fatal_error():
    logger.critical("Error in executor:\n" + traceback.format_exc())


_g_sig_cancel_pending = SignalEvent("SIGUSR1")


class ExecutorContext:
    def __init__(
        self,
        worker: Worker[Work, Result],
        max_parallelism: int,
        max_input_queue: int,
        max_output_queue: int,
        print_lock: asyncio.Lock | None = None,
        watchdog_timeout: float | None = None,
    ):
        self.worker = worker
        self.max_parallelism = max_parallelism
        self.max_output_queue = max_output_queue
        self.print_lock = print_lock
        self.progress_every_seconds = 30.0
        self.start_time = time.monotonic()
        self.produced_count = 0
        self.in_q: asyncio.Queue = asyncio.Queue(maxsize=max_input_queue)
        self.out_lock = asyncio.Lock()
        self.out_buf: dict[int, ExecutorResult] = {}
        self.out_sem = asyncio.Semaphore(self.max_output_queue)
        self.done_reading = asyncio.Event()
        self.done_processing = asyncio.Event()
        self.stop_all = asyncio.Event()

        # if no results are produced in watchdog_timeout time,
        # terminate all the pending tasks and return errors rows,
        # and continue with decoding others
        self.watchdog_timeout = watchdog_timeout
        if self.watchdog_timeout is None:
            self.watchdog_interval = 5.0
        else:
            self.watchdog_interval = 5.0 if (watchdog_timeout > 5.0) else 0.25
        self.last_event = time.monotonic()
        self.last_started_idx = -1
        self.terminate_task_before_idx = 0

        self.stats = ExecutorProgress()

    def _watchdog(self, reset: bool = False):
        t = time.monotonic()
        if reset:
            self.last_event = t
            return

        # if no activity for a while and timeout is set, or signal received, clear the queue and reset
        elapsed = t - self.last_event
        if (
            self.watchdog_timeout is not None and elapsed > self.watchdog_timeout
        ) or _g_sig_cancel_pending.get():
            self.terminate_task_before_idx = self.last_started_idx + 1
            self.last_event = t

    async def _check_work_task(self, idx: int, task: asyncio.Task):
        # check on a work task, cancel it if signaled to
        interval = self.watchdog_interval
        while True:
            if task.done():
                return
            if idx < self.terminate_task_before_idx:
                task.cancel()
                return
            self._watchdog(reset=False)
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                return

    async def _do_work(self, idx: int, work: Work):
        # wait for task to complete while idx >= self.terminate_task_before_idx
        # otherwise cancel it
        if idx > self.last_started_idx:
            self.last_started_idx = idx

        task = asyncio.create_task(self.worker.work(work), name=f"item-{idx}")
        checker = asyncio.create_task(self._check_work_task(idx, task))
        await checker
        try:
            result = await task
            self._watchdog(reset=True)
            return result
        except asyncio.CancelledError:
            logger.error(
                "Cancelled task (blocked) %d:\n%s", idx, traceback.format_exc()
            )
            return WorkResult(
                result=self.worker.get_error_result(work), is_system_error=True
            )

    async def producer_task(self, it: Iterator[Work]):
        try:
            for idx, work in enumerate(it):
                if self.stop_all.is_set():
                    break
                if not self.stop_all.is_set():
                    await self.in_q.put((idx, work))
                    self.produced_count = idx + 1
                else:
                    break
        finally:
            self.done_reading.set()
            if not self.stop_all.is_set():
                # Send termination sentinels if not stopping
                # When stopping (errors or Ctrl-C), do not wait to fill the blocking input queue.
                # all tasks are cancelled, so it is not necessary, and will also cause a deadlock
                for _ in range(self.max_parallelism):
                    await self.in_q.put(None)

    async def worker_task(self, index: int):
        while not self.stop_all.is_set():
            # reserve a place in the output queue before getting the
            # next item from the input queue, to ensure that each item drawn
            # is guaranteed to make it to the output queue when done.
            # Otherwise there can be a deadlock
            await self.out_sem.acquire()
            try:
                item = await asyncio.wait_for(self.in_q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                self.out_sem.release()
                continue
            except asyncio.CancelledError:
                self.out_sem.release()
                break

            try:
                if item is None:
                    # Sentinel value received, indicating producer is done.
                    self.out_sem.release()
                    return

                idx, work = item
                try:
                    self.stats.started += 1
                    worker_result = await self._do_work(idx, work)
                    assert isinstance(
                        worker_result, WorkResult
                    ), "Work result must be a WorkResult"
                    result = ExecutorResult(
                        status=Status.OK,
                        is_error_result=worker_result.is_system_error,
                        is_correct=worker_result.is_correct,
                        result=worker_result.result,
                    )
                except asyncio.CancelledError:
                    # The task was cancelled during execution. Re-raise to ensure
                    # proper cleanup by the asyncio loop. The outer finally
                    # will handle task_done().
                    raise
                except Exception as e:
                    report_fatal_error()
                    result = ExecutorResult(
                        status=Status.ERROR,
                        exception=e,
                    )
                    self.stop_all.set()

                async with self.out_lock:
                    assert idx not in self.out_buf, "this should not happen"
                    self.out_buf[idx] = result
                    out_count = len(self.out_buf)

                self.stats.queued_out = out_count
                self.stats.finished += 1
                if (result.status is not Status.OK) or result.is_error_result:
                    self.stats.errors += 1
                if result.is_correct:
                    self.stats.correct += 1
            finally:
                self.in_q.task_done()

    async def progress_task(self):
        next_event = 0.0
        while not (self.stop_all.is_set() or self.done_processing.is_set()):
            await asyncio.sleep(0.1)
            if self.done_processing.is_set():
                break
            if time.monotonic() < next_event:
                continue
            stats = copy.copy(self.stats)
            t = time.monotonic()
            elapsed = t - self.start_time
            async with enter_async_context_if_not_none(self.print_lock):
                acc = stats.correct / stats.finished if stats.finished > 0 else 0.0
                logger.info(
                    "Progress: t=%ds, processing=%d, done=%d (err=%d acc=%.2f), outq=%d/%d",
                    int(elapsed),
                    stats.started - stats.finished,
                    stats.finished,
                    stats.errors,
                    acc,
                    stats.queued_out,
                    self.max_output_queue,
                )
                if (t - self.last_event) > 120.0:
                    logger.warning(
                        "No new results in the last 120 seconds,"
                        + " use 'kill -SIGUSR1 %d' to cancel pending work and unblock now",
                        PID,
                    )
            next_event = time.monotonic() + self.progress_every_seconds

    async def run_iter(self, it: Iterator[Work]):
        _g_sig_cancel_pending.install()
        # Launch tasks
        prod = asyncio.create_task(self.producer_task(it), name="producer")
        workers = [
            asyncio.create_task(self.worker_task(i), name=f"worker-{i}")
            for i in range(self.max_parallelism)
        ]
        progress = asyncio.create_task(self.progress_task(), name="progress")

        next_idx = 0
        try:
            while True:
                to_yield: list[ExecutorResult] = []
                async with self.out_lock:
                    # collect next element if any
                    if next_idx in self.out_buf:
                        result = self.out_buf.pop(next_idx)
                        out_count = len(self.out_buf)
                        to_yield.append(result)
                        next_idx += 1
                if to_yield:
                    self.stats.queued_out = out_count

                # if none, wait a bit
                should_sleep = not to_yield

                if to_yield:
                    # yield the new element or raise exception
                    try:
                        result = to_yield[0]
                        if result.status is Status.OK:
                            yield result.result
                        else:
                            raise result.exception
                    finally:
                        # increment the semaphore accordingly
                        self.out_sem.release()
                else:
                    if self.done_reading.is_set() and next_idx >= self.produced_count:
                        await asyncio.gather(*workers, return_exceptions=True)
                        await prod
                        self.done_processing.set()
                        await progress
                        return

                if should_sleep:
                    await asyncio.sleep(0.1)
        except (Exception, asyncio.CancelledError):
            report_fatal_error()
            self.stop_all.set()
            # Cancel all tasks
            for t in workers:
                t.cancel()
            prod.cancel()
            progress.cancel()
            # Wait for cancellation to complete
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.gather(*workers, return_exceptions=True)
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await prod
            raise
        finally:
            # Cleanup remaining tasks
            if not prod.done():
                prod.cancel()
            for t in workers:
                if not t.done():
                    t.cancel()
            if not progress.done():
                progress.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.gather(*workers, return_exceptions=True)
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await prod
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await progress


async def iter_map_parallel_ordered(
    it: Iterator[Work],
    max_parallelism: int,
    max_input_queue: int,
    max_output_queue: int,
    worker: Worker[Work, Result],
    print_lock: asyncio.Lock | None = None,
    watchdog_timeout: float | None = None,
) -> AsyncIterator[Result]:
    """
    Run all work items from `it` through `worker.work` with:
      - at most `max_parallelism` concurrent executions,
      - at most `max_input_queue` items buffered after being read from `it`,
      - at most `max_output_queue` completed-but-not-yet-yielded results buffered,
      - yields result objects strictly in input order.
    If provided, `print_lock` is acquired during log prints
    """
    if max_parallelism <= 0:
        raise ValueError("max_parallelism must be > 0")
    if max_input_queue <= 0:
        raise ValueError("max_input_queue must be > 0")
    if max_output_queue < max_parallelism:
        raise ValueError("max_output_queue must be >= max_parallelism")

    ctx = ExecutorContext(
        worker=worker,
        max_parallelism=max_parallelism,
        max_input_queue=max_input_queue,
        max_output_queue=max_output_queue,
        print_lock=print_lock,
        watchdog_timeout=watchdog_timeout,
    )
    async for res in ctx.run_iter(it):
        yield res


async def iter_map_sequential(
    it: Iterator[Work],
    worker: Worker[Work, Result],
) -> AsyncIterator[Result]:
    start_time = time.monotonic()
    next_event = 0.0
    finished = 0
    errors = 0
    correct = 0
    progress_every_seconds = 30.0
    for work in it:
        work_result = await worker.work(work)
        finished += 1
        if work_result.is_system_error:
            errors += 1
        if work_result.is_correct:
            correct += 1
        if time.monotonic() >= next_event:
            elapsed = time.monotonic() - start_time
            acc = correct / finished if finished > 0 else 0.0
            logger.info(
                "Progress: t=%ds, done=%d (err=%d acc=%.2f)",
                int(elapsed),
                finished,
                errors,
                acc,
            )
            next_event = time.monotonic() + progress_every_seconds
        yield work_result.result
