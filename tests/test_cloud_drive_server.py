# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

import pytest

from thinkingbox.common.mcp_proxy_client import MCPProxyClient

TOOLS = [
    "get_metadata",
    "upload_file",
    "delete_file",
    "get_text_content",
    "list_files",
    "search_files",
]
SERVER_CONFIG = {
    "cloud_drive": {
        "files": [
            {
                "path": "Documents/file.txt",
                "metadata": {
                    "created": "2025-04-09T19:38:07",
                    "modified": "2025-04-09T19:38:07",
                },
                "text_content": "some text",
            }
        ],
    }
}


@pytest.mark.asyncio
async def test_get_metadata(session_proxy):
    async with session_proxy.get(SERVER_CONFIG, TOOLS) as session:
        session: MCPProxyClient
        response = await session.call_tool("get_metadata", path="Documents/file.txt")
        print(response)
        result = json.loads(response)
        assert result["status"] == "ok"
        assert result["metadata"]["created"] == "2025-04-09T19:38:07"
        assert result["metadata"]["modified"] == "2025-04-09T19:38:07"


@pytest.mark.asyncio
async def test_upload_file(session_proxy):
    async with session_proxy.get(SERVER_CONFIG, TOOLS) as session:
        session: MCPProxyClient
        response = await session.call_tool(
            "upload_file",
            path="Documents/new_file.txt",
            text_content="new content",
            overwrite=False,
        )
        result = json.loads(response)
        assert result["status"] == "ok"

        # Verify the file was uploaded
        response = await session.call_tool(
            "get_metadata", path="Documents/new_file.txt"
        )
        result = json.loads(response)
        assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_delete_file(session_proxy):
    async with session_proxy.get(SERVER_CONFIG, TOOLS) as session:
        session: MCPProxyClient
        response = await session.call_tool("delete_file", path="Documents/file.txt")
        result = json.loads(response)
        assert result["status"] == "ok"

        # Verify the file was deleted
        response = await session.call_tool("get_metadata", path="Documents/file.txt")
        with pytest.raises(json.decoder.JSONDecodeError):
            json.loads(response)


@pytest.mark.asyncio
async def test_get_text_content(session_proxy):
    async with session_proxy.get(SERVER_CONFIG, TOOLS) as session:
        session: MCPProxyClient
        response = await session.call_tool(
            "get_text_content", path="Documents/file.txt"
        )
        result = json.loads(response)
        assert result["status"] == "ok"
        assert result["text_content"] == "some text"


@pytest.mark.asyncio
async def test_list_files(session_proxy):
    async with session_proxy.get(SERVER_CONFIG, TOOLS) as session:
        session: MCPProxyClient
        response = await session.call_tool("list_files", prefix="Documents")
        result = json.loads(response)
        assert result["status"] == "ok"
        assert len(result["files"]) == 1
        assert result["files"][0]["path"] == "Documents/file.txt"


@pytest.mark.asyncio
async def test_find_files(session_proxy):
    async with session_proxy.get(SERVER_CONFIG, TOOLS) as session:
        session: MCPProxyClient
        response = await session.call_tool("search_files", pattern="Documents/*.txt")
        result = json.loads(response)
        assert result["status"] == "ok"
        assert len(result["files"]) == 1
        assert result["files"][0]["path"] == "Documents/file.txt"
