#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import enum
import os
import sys
from pathlib import Path
from typing import Annotated

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("test_tool")


class TestToolContext(BaseModel):
    path: Path | None = None
    effects: list[dict] = Field(default_factory=list)
    session_id: str = ""

    def set_file(self, path: str | Path):
        self.path = Path(path)

    def read_file(self) -> list[str]:
        if self.path is None:
            raise RuntimeError("not initialized")
        with open(self.path, "r", encoding="utf-8") as f:
            return [x for x in f.read().splitlines() if x]

    def write_file(self, content: list[str]):
        if self.path is None:
            raise RuntimeError("not initialized")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    def delete_file(self):
        self.path.unlink()


# MCP session id -> TestToolContext
_g_sessions: dict[str, TestToolContext] = {}


@mcp.tool(name="__reserved__init")
async def initialize(session_id: str, config: dict, ctx: Context) -> dict:
    if not session_id:
        raise RuntimeError("session_id not present")
    print(
        f"initialize: mcp-session={ctx.session_id} tb-session={session_id}",
        file=sys.stderr,
    )

    assert ctx.session_id not in _g_sessions
    _g_sessions[ctx.session_id] = TestToolContext(session_id=session_id)

    s = _g_sessions[ctx.session_id]
    s.set_file(config["file"])
    s.write_file([session_id])
    return {"status": "ok"}


@mcp.tool(name="__reserved__geteffects")
async def geteffects(session_id: str, ctx: Context) -> dict:
    print(
        f"geteffects: mcp-session={ctx.session_id} tb-session={session_id}",
        file=sys.stderr,
    )
    s = _g_sessions[ctx.session_id]
    if session_id != s.session_id:
        raise RuntimeError(
            f"Wrong session_id, expected {s.session_id!r}, received {session_id!r}"
        )
    return {"effects": s.effects}


@mcp.tool(name="__reserved__teardown")
async def teardown(session_id: str, ctx: Context) -> dict:
    print(
        f"teardown: mcp-session={ctx.session_id} tb-session={session_id}",
        file=sys.stderr,
    )
    s = _g_sessions[ctx.session_id]
    if session_id != s.session_id:
        raise RuntimeError(
            f"Wrong session_id, expected {s.session_id!r}, received {session_id!r}"
        )
    s.delete_file()
    del _g_sessions[ctx.session_id]
    return {"status": "ok"}


@mcp.tool(name="append")
async def append(
    content: Annotated[str, Field(description="new contents")],
    ctx: Context,
) -> dict:
    s = _g_sessions[ctx.session_id]
    print(
        f"append: mcp-session={ctx.session_id} tb-context.session={s.session_id}",
        file=sys.stderr,
    )
    lines = s.read_file()
    s.write_file(lines + [content])
    s.effects.append({"tool": "append", "content": content})
    return {"status": "ok"}


@mcp.tool(name="get_info")
def get_info() -> dict:
    obj = {
        "TEST_TOOL": os.environ.get("TEST_TOOL"),
        "cwd": os.getcwd(),
    }
    return obj


@mcp.tool(name="add")
def add(
    a: Annotated[int, Field(description="a number")],
    b: Annotated[int, Field(description="another number")],
) -> dict:
    return {"result": a + b}


class MyEnum(enum.Enum):
    A = "a"
    B = "b"


@mcp.tool()
def tool_with_enum(
    value: Annotated[MyEnum, Field(description="enum")],
) -> dict:
    return {"result": value.value}


@mcp.tool(name="slow_operation")
async def slow_operation(
    duration: Annotated[float, Field(description="seconds to sleep")],
    ctx: Context,
) -> dict:
    """Sleeps for the specified duration to test timeouts."""
    import asyncio

    await asyncio.sleep(duration)
    return {"status": "completed", "duration": duration}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=10111)
    p.add_argument(
        "--transport",
        type=str,
        choices=["streamable-http", "sse", "stdio"],
        default="streamable-http",
    )
    args = p.parse_args()

    transport_kwargs = {}
    if args.transport in ("streamable-http", "sse"):
        transport_kwargs["host"] = "127.0.0.1"
        transport_kwargs["port"] = args.port

    mcp.run(
        transport=args.transport,
        show_banner=False,
        **transport_kwargs,
    )


if __name__ == "__main__":
    main()
