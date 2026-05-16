# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the TUI /user command functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thinkingbox.cli.tui_main import TUI
from thinkingbox.common.chat_types import Text
from thinkingbox.common.config_types import (
    AgentConfig,
    ConfigFile,
    CustomSessionConfig,
    HydratedTestCase,
    ScenarioConfig,
)


def create_mock_config(user_model: CustomSessionConfig | None = None) -> ConfigFile:
    """Create a mock ConfigFile for testing."""
    return ConfigFile(
        mcp_proxy="http://127.0.0.1:7111",
        agent_model=CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=[],
        ),
        judge_model=CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=[],
        ),
        user_model=user_model,
    )


def create_mock_test_case(user_context: str = "") -> HydratedTestCase:
    """Create a mock HydratedTestCase for testing."""
    return HydratedTestCase(
        uid="test:test_case",
        agent=AgentConfig(
            system_instructions="You are a helpful assistant.",
            builtin_tools=[],
        ),
        scenario=ScenarioConfig(
            world_state={},
            tools=[],
        ),
        query="Hello",
        test_code="",
        user_context=user_context,
    )


def create_mock_agent() -> MagicMock:
    """Create a mock AgentSession."""
    agent = MagicMock()
    agent.conversation = MagicMock()
    agent.conversation.messages = [
        Text(role="user", content="Hello"),
        Text(role="assistant", content="Hi there! How can I help you?"),
    ]
    return agent


class TestGenerateUserMessage:
    """Tests for the generate_user_message method."""

    @pytest.mark.asyncio
    async def test_user_context_show_empty(self):
        """Test /user context shows (none) when context is empty."""
        config = create_mock_config()
        tc = create_mock_test_case(user_context="")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint") as mock_pprint:
            result = await tui.generate_user_message(agent, ["context"])

        assert result is None
        # Check that pprint was called with context info
        calls = [str(call) for call in mock_pprint.call_args_list]
        assert any("user context" in str(call).lower() for call in calls)
        assert any("none" in str(call).lower() for call in calls)

    @pytest.mark.asyncio
    async def test_user_context_show_with_content(self):
        """Test /user context shows the current context."""
        config = create_mock_config()
        tc = create_mock_test_case(user_context="I am a customer looking for help.")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint") as mock_pprint:
            result = await tui.generate_user_message(agent, ["context"])

        assert result is None
        calls = [str(call) for call in mock_pprint.call_args_list]
        assert any("I am a customer looking for help" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_user_context_set(self):
        """Test /user context <text> sets the context."""
        config = create_mock_config()
        tc = create_mock_test_case(user_context="")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint") as mock_pprint:
            result = await tui.generate_user_message(
                agent, ["context", "New", "user", "context"]
            )

        assert result is None
        assert tui.user_context == "New user context"
        calls = [str(call) for call in mock_pprint.call_args_list]
        assert any("updated" in str(call).lower() for call in calls)

    @pytest.mark.asyncio
    async def test_user_no_user_model(self):
        """Test /user without user_model configured shows error."""
        config = create_mock_config(user_model=None)
        tc = create_mock_test_case(user_context="Some context")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint") as mock_pprint:
            result = await tui.generate_user_message(agent, [])

        assert result is None
        calls = [str(call) for call in mock_pprint.call_args_list]
        assert any("user_model not configured" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_user_no_user_context(self):
        """Test /user without user_context shows error."""
        user_model = CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=[[Text(role="assistant", content="Generated response")]],
        )
        config = create_mock_config(user_model=user_model)
        tc = create_mock_test_case(user_context="")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint") as mock_pprint:
            result = await tui.generate_user_message(agent, [])

        assert result is None
        calls = [str(call) for call in mock_pprint.call_args_list]
        assert any("no user_context set" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_user_generates_message(self):
        """Test /user successfully generates a user message."""
        user_model = CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=[[Text(role="assistant", content="What time is my flight?")]],
        )
        config = create_mock_config(user_model=user_model)
        tc = create_mock_test_case(
            user_context="User has a flight booked for tomorrow."
        )
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint"):
            result = await tui.generate_user_message(agent, [])

        assert result is not None
        assert isinstance(result, Text)
        assert result.role == "user"
        assert result.metadata.get("is_user_llm") is True

    @pytest.mark.asyncio
    async def test_user_generate_error_returns_none(self):
        """Test /user returns None and prints error when generate() raises."""
        user_model = CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=[],
        )
        config = create_mock_config(user_model=user_model)
        tc = create_mock_test_case(user_context="Some context")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with (
            patch("thinkingbox.cli.tui_main.pprint") as mock_pprint,
            patch.object(tui, "user_model", create=True),
            patch("thinkingbox.cli.tui_main.UserSimulator") as mock_sim_cls,
        ):
            mock_sim = mock_sim_cls.return_value
            mock_sim.generate = AsyncMock(
                side_effect=RuntimeError("LLM connection failed")
            )
            result = await tui.generate_user_message(agent, [])

        assert result is None
        calls = [str(call) for call in mock_pprint.call_args_list]
        assert any("failed to generate user response" in str(call) for call in calls)
        assert any("LLM connection failed" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_user_empty_content_returns_none(self):
        """Test /user returns None with warning when LLM returns empty content."""
        user_model = CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=[],
        )
        config = create_mock_config(user_model=user_model)
        tc = create_mock_test_case(user_context="Some context")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with (
            patch("thinkingbox.cli.tui_main.pprint") as mock_pprint,
            patch.object(tui, "user_model", create=True),
            patch("thinkingbox.cli.tui_main.UserSimulator") as mock_sim_cls,
        ):
            mock_sim = mock_sim_cls.return_value
            mock_sim.generate = AsyncMock(return_value=Text(role="user", content="   "))
            result = await tui.generate_user_message(agent, [])

        assert result is None
        calls = [str(call) for call in mock_pprint.call_args_list]
        assert any("empty response" in str(call) for call in calls)


class TestHandleCommand:
    """Tests for handle_command return types."""

    @pytest.mark.asyncio
    async def test_quit_returns_exit_tuple(self):
        """Test /quit returns (True, None)."""
        config = create_mock_config()
        tc = create_mock_test_case()
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint"):
            should_exit, msg = await tui.handle_command(agent, "quit", [])

        assert should_exit is True
        assert msg is None

    @pytest.mark.asyncio
    async def test_unrecognized_command_returns_continue_tuple(self):
        """Test unrecognized command returns (False, None)."""
        config = create_mock_config()
        tc = create_mock_test_case()
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint"):
            should_exit, msg = await tui.handle_command(agent, "unknown_cmd", [])

        assert should_exit is False
        assert msg is None

    @pytest.mark.asyncio
    async def test_user_command_returns_message(self):
        """Test /user returns (False, Text) when successful."""
        user_model = CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=[[Text(role="assistant", content="Follow-up question")]],
        )
        config = create_mock_config(user_model=user_model)
        tc = create_mock_test_case(user_context="User context here")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint"):
            should_exit, msg = await tui.handle_command(agent, "user", [])

        assert should_exit is False
        assert msg is not None
        assert isinstance(msg, Text)

    @pytest.mark.asyncio
    async def test_user_context_command_returns_none(self):
        """Test /user context returns (False, None)."""
        config = create_mock_config()
        tc = create_mock_test_case(user_context="Some context")
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint"):
            should_exit, msg = await tui.handle_command(agent, "user", ["context"])

        assert should_exit is False
        assert msg is None

    @pytest.mark.asyncio
    async def test_help_command_returns_continue_tuple(self):
        """Test /help returns (False, None) and lists commands."""
        config = create_mock_config()
        tc = create_mock_test_case()
        tui = TUI(config, tc)
        agent = create_mock_agent()

        with patch("thinkingbox.cli.tui_main.pprint") as mock_pprint:
            should_exit, msg = await tui.handle_command(agent, "help", [])

        assert should_exit is False
        assert msg is None
        calls = [str(call) for call in mock_pprint.call_args_list]
        assert any("/quit" in str(call) for call in calls)
        assert any("/user" in str(call) for call in calls)
        assert any("/help" in str(call) for call in calls)


class TestTUIInitialization:
    """Tests for TUI initialization."""

    def test_tui_initializes_user_model_when_configured(self):
        """Test TUI creates user_model when configured."""
        user_model = CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=[],
        )
        config = create_mock_config(user_model=user_model)
        tc = create_mock_test_case()

        tui = TUI(config, tc)

        assert tui.user_model is not None

    def test_tui_user_model_none_when_not_configured(self):
        """Test TUI has user_model=None when not configured."""
        config = create_mock_config(user_model=None)
        tc = create_mock_test_case()

        tui = TUI(config, tc)

        assert tui.user_model is None

    def test_tui_loads_user_context_from_test_case(self):
        """Test TUI loads user_context from test case."""
        config = create_mock_config()
        tc = create_mock_test_case(user_context="Test context from test case")

        tui = TUI(config, tc)

        assert tui.user_context == "Test context from test case"

    def test_tui_empty_user_context_when_none(self):
        """Test TUI has empty user_context when test case has None."""
        config = create_mock_config()
        tc = create_mock_test_case(user_context="")

        tui = TUI(config, tc)

        assert tui.user_context == ""
