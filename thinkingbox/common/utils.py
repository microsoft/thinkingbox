# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import contextlib
import logging
import time
from abc import ABC, abstractmethod
from typing import IO, Awaitable, Iterator, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T")


@contextlib.asynccontextmanager
async def enter_async_context_if_not_none(obj):
    if obj is None:
        yield None
    else:
        async with obj as v:
            yield v


def syncify(coro: Awaitable[T]) -> T:
    """
    Runs an awaitable coroutine in a synchronous context and returns its result.
    This is useful for calling async functions from synchronous code.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Necessary if running in a thread with no event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class CredentialBase(ABC):
    @abstractmethod
    async def get_token(self) -> str: ...

    @abstractmethod
    async def invalidate(self) -> None: ...


class ApiKeyCredential(CredentialBase):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_token(self):
        return self.api_key

    async def invalidate(self):
        pass


class ThinkingBoxError(Exception):
    pass


def raise_for_status_with_error(response: httpx.Response):
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as status_exc:
        error_message = ""
        try:
            error_message = status_exc.response.json()["error"]
        except (ValueError, TypeError, KeyError):
            error_message = str(status_exc.response.text)
        raise ThinkingBoxError(error_message) from status_exc


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("thinkingbox.backoff").setLevel(logging.ERROR)
    # Suppress verbose Azure SDK logging
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.identity._internal.get_token_mixin").setLevel(
        logging.WARNING
    )
    # Suppress other common Azure SDK verbose loggers
    logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)


class Timers:
    def __init__(self):
        self._timings: dict[str, float] = {}

    def ensure(self, key: str):
        if key not in self._timings:
            self._timings[key] = 0.0

    def _update_one(self, key: str, value: float, accumulate: bool = True):
        if accumulate:
            value += self._timings.get(key, 0)
        self._timings[key] = value

    @contextlib.contextmanager
    def measure(self, key: str, accumulate: bool = True):
        t_beg = time.monotonic()
        try:
            yield
        finally:
            t_elapsed = time.monotonic() - t_beg
            self._update_one(key, t_elapsed, accumulate=accumulate)

    @property
    def timings(self) -> dict[str, float]:
        return self._timings.copy()

    def pop(self) -> dict[str, float]:
        out: dict[str, float] = self._timings.copy()
        self.zero()
        return out

    def update(self, other: dict[str, float], accumulate: bool = True):
        for key, value in other.items():
            self._update_one(key, value, accumulate=accumulate)

    def merge_into(self, target: dict, accumulate: bool = True):
        for key, value in self._timings.items():
            if accumulate:
                value += target.get(key, 0)
            target[key] = value

    def zero(self):
        for k in self._timings:
            self._timings[k] = 0.0


def iter_nonempty_lines(file: IO[str]) -> Iterator[str]:
    for line in file:
        line = line.strip()
        if line:
            yield line


ModelT = TypeVar("ModelT", bound=BaseModel)


def iter_validate_jsonl(file: IO[str], model: type[ModelT]) -> Iterator[ModelT]:
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError(repr(model))
    for line in iter_nonempty_lines(file):
        yield model.model_validate_json(line)


class ErrorInfo(BaseModel):
    type: str = "None"
    message: str = ""
    tb: str = ""
