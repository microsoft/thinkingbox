# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib

import pytest_asyncio

from thinkingbox.common.mcp_proxy_client import MCPProxyClient


class MCPProxyClientFactory:
    def __init__(self):
        pass

    @contextlib.asynccontextmanager
    async def get(
        self,
        server_config: dict[str, dict],
        tools: list[str],
        endpoint: str = "http://127.0.0.1:7111",
        timeout: float = 120.0,
    ):
        async with MCPProxyClient.session_context(
            endpoint=endpoint,
            server_config=server_config,
            available_tools=tools,
            timeout=timeout,
        ) as mcp:
            yield mcp


@pytest_asyncio.fixture(scope="module")
async def session_proxy():
    # Setup code
    yield MCPProxyClientFactory()
