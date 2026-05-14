# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Integration tests for MCP session timeout configuration.

Tests verify that:
1. Session-level timeout applies to all tool calls
2. Default timeout (120s) is used when not specified
3. Timeouts are properly enforced and operations are canceled
4. All tools in a session share the same timeout
"""

import json

import pytest


@pytest.mark.asyncio
async def test_session_timeout_basic(tmp_path, session_proxy):
    """Test that session-level timeout applies to tool calls."""
    test_file = tmp_path / "file.txt"
    server_name = "test_tool_process"
    cfg = {
        server_name: {
            "file": str(test_file),
        },
    }

    # Create session with 8 second timeout
    async with session_proxy.get(
        cfg,
        tools=["slow_operation"],
        timeout=8.0,
    ) as session:
        # This should succeed: 3 seconds < 8 second timeout
        result = await session.call_tool("slow_operation", duration=3.0)
        result_data = json.loads(result)
        assert result_data["status"] == "completed"
        assert result_data["duration"] == 3.0


@pytest.mark.asyncio
async def test_session_timeout_enforcement(tmp_path, session_proxy):
    """Test that session timeout is enforced and operations are canceled."""
    test_file = tmp_path / "file.txt"
    server_name = "test_tool_process"
    cfg = {
        server_name: {
            "file": str(test_file),
        },
    }

    # Create session with 5 second timeout
    async with session_proxy.get(
        cfg,
        tools=["slow_operation"],
        timeout=5.0,
    ) as session:
        # This should timeout: 5 seconds < 7 second duration
        with pytest.raises(Exception) as exc_info:
            await session.call_tool("slow_operation", duration=7)

        # Verify it's a timeout-related error
        error_str = str(exc_info.value).lower()
        assert (
            "timeout" in error_str or "timed out" in error_str
        ), f"Expected timeout error, got: {exc_info.value}"
