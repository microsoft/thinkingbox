# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from thinkingbox.common.mcp_proxy_client import MCPProxyContext

MCP_SERVER_PORT = 10611
TOOLS = [
    "get_info",
    "append",
    "tool_with_enum",
]


@contextlib.asynccontextmanager
async def get_mcp_client(
    url: str,
    transport: str,
    timeout: float,
    headers: dict[str, str] | None = None,
):
    if transport == "sse":
        async with sse_client(
            url=url,
            headers=headers,
            timeout=timeout,
        ) as (r, w):
            async with ClientSession(r, w) as client:
                await client.initialize()
                yield client
        return

    assert transport == "streamable-http"
    http_client = httpx.AsyncClient(
        headers=headers, timeout=timeout, follow_redirects=True
    )
    async with streamable_http_client(
        url=url,
        http_client=http_client,
    ) as (r, w, _):
        async with ClientSession(r, w) as client:
            await client.initialize()
            yield client


async def wait_for_mcp_server(
    url: str,
    transport: str,
    timeout: float = 30.0,
    interval: float = 0.5,
):
    deadline = time.time() + timeout
    # TODO
    while True:
        try:
            async with get_mcp_client(url, transport, 3.0) as client:
                _ = await client.list_tools()
            return
        except Exception:
            # Connection failures are expected until the proxy is ready.
            pass
        if time.time() >= deadline:
            raise RuntimeError(f"Timeout trying to connect to {url}")
        await asyncio.sleep(interval)


@contextlib.asynccontextmanager
async def mcp_test_server_context(port: int, transport: str):
    here = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(here / "mcp_test_tool.py"),
        "--port",
        str(port),
        "--transport",
        transport,
    ]
    path = "mcp" if (transport == "streamable-http") else "sse"
    env = os.environ.copy()
    proc = await asyncio.create_subprocess_exec(
        *command,
        env=env,
        stdout=None,
        stderr=None,
    )
    try:
        await asyncio.sleep(1.0)
        await wait_for_mcp_server(
            f"http://127.0.0.1:{port}/{path}/",
            transport,
            timeout=30.0,
        )
        yield
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=5)


def read_file_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [x for x in f.read().splitlines() if x]


async def run_test(
    session: MCPProxyContext,
    server_name: str,
    test_file: Path,
    mcp_client_for_tool_calls: ClientSession | None = None,
):
    # init correctly called and creates the test file
    # with the correct session id passed to it
    lines = read_file_lines(test_file)
    assert lines == [session.session_id]

    # append line correctly called (file changed)
    if mcp_client_for_tool_calls is None:
        await session.call_tool("append", content="another line")
    else:
        await mcp_client_for_tool_calls.call_tool(
            "append", arguments={"content": "another line"}
        )

    lines = read_file_lines(test_file)
    assert lines == [session.session_id, "another line"]

    # geteffects correctly called and returns effects
    effects = await session.get_effects()
    assert effects[server_name]["effects"] == [
        {"tool": "append", "content": "another line"},
    ]

    # can call invisible tools through the reserved tool
    result = await session.call_tool(
        "__reserved__server_tool",
        server_name=server_name,
        tool_name="add",
        arguments={
            "a": 33,
            "b": 9,
        },
    )
    assert json.loads(result) == {"result": 42}


@pytest.mark.asyncio
async def test_transport_http(tmp_path, session_proxy):
    test_file = tmp_path / "file.txt"
    server_name = "test_tool_http"
    cfg = {
        server_name: {
            "file": str(test_file),
        },
    }
    async with mcp_test_server_context(MCP_SERVER_PORT, "streamable-http"):
        async with session_proxy.get(cfg, TOOLS) as session:
            await run_test(session, server_name, test_file)

        # teardown correctly called (deletes the test file)
        assert not test_file.exists()


@pytest.mark.asyncio
async def test_transport_sse(tmp_path, session_proxy):
    test_file = tmp_path / "file.txt"
    server_name = "test_tool_sse"
    cfg = {
        server_name: {
            "file": str(test_file),
        },
    }
    async with mcp_test_server_context(MCP_SERVER_PORT, "sse"):
        async with session_proxy.get(cfg, TOOLS) as session:
            await run_test(session, server_name, test_file)

        # teardown correctly called (deletes the test file)
        assert not test_file.exists()


@pytest.mark.asyncio
async def test_transport_stdio(tmp_path, session_proxy):
    test_file = tmp_path / "file.txt"
    server_name = "test_tool_process"
    cfg = {
        server_name: {
            "file": str(test_file),
        },
    }
    async with session_proxy.get(cfg, TOOLS) as session:
        session: MCPProxyContext

        tools = await session.list_tools()
        tool_with_enum = [tool for tool in tools if tool.name == "tool_with_enum"]
        value_schema = tool_with_enum[0].input_schema["properties"]["value"]
        # verify that Enum was dereferenced (fastmcp >=3 includes description in schema)
        assert value_schema == {
            "enum": ["a", "b"],
            "type": "string",
            "description": "enum",
        }
        # call tool with enum
        result = await session.call_tool("tool_with_enum", value="a")
        assert json.loads(result) == {"result": "a"}

        # check env and cwd from config work
        response = await session.call_tool("get_info")
        result = json.loads(response)
        assert result["TEST_TOOL"] == "hello"
        assert Path(result["cwd"]).name == "tests"

        await run_test(session, server_name, test_file)

    # teardown correctly called (deletes the test file)
    assert not test_file.exists()


@pytest.mark.asyncio
async def test_mcp_interface(tmp_path, session_proxy):
    test_file = tmp_path / "file.txt"
    server_name = "test_tool_process"
    cfg = {
        server_name: {
            "file": str(test_file),
        },
    }
    async with session_proxy.get(cfg, TOOLS) as session:
        session: MCPProxyContext

        # Now connect to the MCP interface
        async with get_mcp_client(
            url=session.client.endpoint.rstrip("/") + "/mcp",
            transport="streamable-http",
            timeout=60.0,
            headers={
                "X-TB-Session-Id": session.session_id,
            },
        ) as real_mcp_client:
            tools = await real_mcp_client.list_tools()
            tool_with_enum = [
                tool for tool in tools.tools if tool.name == "tool_with_enum"
            ]
            value_schema = tool_with_enum[0].inputSchema["properties"]["value"]
            # verify that Enum was dereferenced (fastmcp >=3 includes description in schema)
            assert value_schema == {
                "enum": ["a", "b"],
                "type": "string",
                "description": "enum",
            }
            # call tool with enum
            result = await real_mcp_client.call_tool(
                "tool_with_enum", arguments={"value": "a"}
            )
            assert result.structuredContent == {"result": "a"}

            await run_test(
                session,
                server_name,
                test_file,
                mcp_client_for_tool_calls=real_mcp_client,
            )

    # teardown correctly called (deletes the test file)
    assert not test_file.exists()


@contextlib.asynccontextmanager
async def session_proxy_process(port: int, api_key: str):
    """Start a session proxy subprocess with auth enabled."""
    here = Path(__file__).resolve().parent
    servers_path = here / "servers.yaml"
    command = [
        sys.executable,
        "-m",
        "thinkingbox.cli.main",
        "mcp-start",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--servers",
        str(servers_path),
        "--require-auth",
        "--api-key",
        api_key,
    ]
    proc = await asyncio.create_subprocess_exec(
        *command,
        env=os.environ.copy(),
        stdout=None,
        stderr=None,
    )
    try:
        # Wait for the server to be ready via /health (no auth required)
        deadline = time.time() + 60.0
        while True:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    r = await client.get(f"http://127.0.0.1:{port}/health")
                    if r.status_code == 200:
                        break
            except httpx.HTTPError:
                # Connection failures are expected until the proxy is ready.
                pass
            if time.time() >= deadline:
                raise RuntimeError("Timeout waiting for session proxy to start")
            await asyncio.sleep(0.5)
        yield proc
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=5)


def assert_http_status_unauthorized_in_exception_group(exc: Exception):
    if isinstance(exc, httpx.HTTPStatusError):
        assert exc.response.status_code == 401
    elif callable(subgroup := getattr(exc, "subgroup", None)):
        errs = subgroup(httpx.HTTPStatusError)
        assert errs is not None, f"No HTTPStatusError in group: {exc}"
        assert errs.exceptions[0].response.status_code == 401
    else:
        raise exc


@pytest.mark.asyncio
async def test_session_proxy_with_auth():
    port = MCP_SERVER_PORT
    api_key = "test-secret"
    session_id = "auth-test-session"
    base_url = f"http://127.0.0.1:{port}"
    payload = {
        "session_id": session_id,
        "server_config": {"test_notepad": {"text": "some"}},
        "available_tools": ["read_notepad", "write_notepad"],
    }

    async with session_proxy_process(port, api_key):
        async with httpx.AsyncClient() as client:
            # No Authorization header -> 401
            r = await client.post(f"{base_url}/session_create", json=payload)
            assert r.status_code == 401

            # Wrong key -> 401
            r = await client.post(
                f"{base_url}/session_create",
                json=payload,
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert r.status_code == 401

            # Correct key -> succeeds
            r = await client.post(
                f"{base_url}/session_create",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            r.raise_for_status()

            # Can call a tool
            r = await client.post(
                f"{base_url}/call_tool",
                json={
                    "session_id": session_id,
                    "tool_name": "write_notepad",
                    "arguments": {"text": "I changed it"},
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            r.raise_for_status()

            # MCP endpoint without Bearer token -> 401
            try:
                async with get_mcp_client(
                    f"{base_url}/mcp/",
                    transport="streamable-http",
                    timeout=20.0,
                    headers={
                        "X-TB-Session-Id": session_id,
                    },
                ) as _:
                    pass
                pytest.fail("Expected HTTPStatusError")
            except Exception as exc:
                assert_http_status_unauthorized_in_exception_group(exc)

            # MCP endpoint with wrong key -> 401
            try:
                async with get_mcp_client(
                    f"{base_url}/mcp/",
                    transport="streamable-http",
                    timeout=20.0,
                    headers={
                        "X-TB-Session-Id": session_id,
                        "Authorization": "Bearer wrong-key",
                    },
                ) as _:
                    pass
                pytest.fail("Expected HTTPStatusError")
            except Exception as exc:
                assert_http_status_unauthorized_in_exception_group(exc)

            # MCP endpoint with correct key
            async with get_mcp_client(
                f"{base_url}/mcp/",
                transport="streamable-http",
                timeout=20.0,
                headers={
                    "X-TB-Session-Id": session_id,
                    "Authorization": f"Bearer {api_key}",
                },
            ) as mcp_client:
                # Can call a tool
                result = await mcp_client.call_tool("read_notepad", arguments={})
                assert result.structuredContent["text"] == "I changed it"
