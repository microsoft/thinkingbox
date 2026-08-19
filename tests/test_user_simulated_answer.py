# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from tests.mock_session import MockSession
from thinkingbox.common.chat_types import Text
from thinkingbox.common.user_simulated_answer import UserSimulator


def _create_conversation_pattern(num_turns: int) -> list:
    """Create a test conversation similar to test_chat_types.py"""
    messages = [Text(role="system", content="You are a helpful assistant")]

    for i in range(num_turns):
        messages.append(Text(role="user", content=f"User message {i+1}"))
        messages.append(
            Text(
                role="assistant",
                content=f"Assistant response {i+1}",
                metadata={"tag": "text"},
            )
        )

    return messages


@pytest.mark.asyncio
async def test_user_simulator_basic_response():
    """Test that UserSimulator generates a basic user response."""
    # Mock LLM that returns a simple text response
    mock_llm = MockSession(
        completions=[
            [Text(role="assistant", content="I need more details about your order")]
        ]
    )

    simulator = UserSimulator(llm=mock_llm)
    chat_history = _create_conversation_pattern(1)
    user_context = "User wants to track order #12345"

    result = await simulator.generate(chat_history, user_context)

    assert isinstance(result, Text)
    assert result.role == "user"
    assert result.content == "I need more details about your order"
    assert len(simulator.history) == 1


@pytest.mark.asyncio
async def test_user_simulator_cleans_role_prefix():
    """Test that UserSimulator removes role prefixes from responses."""
    # Mock LLM that returns response with "User:" prefix
    mock_llm = MockSession(
        completions=[[Text(role="assistant", content="User: My order number is 12345")]]
    )

    simulator = UserSimulator(llm=mock_llm)
    chat_history = _create_conversation_pattern(1)
    user_context = "User has order #12345"

    result = await simulator.generate(chat_history, user_context)

    assert result.content == "My order number is 12345"
    assert not result.content.startswith("User:")


@pytest.mark.asyncio
async def test_user_simulator_removes_quotes():
    """Test that UserSimulator removes surrounding quotes from responses."""
    # Mock LLM that returns quoted response
    mock_llm = MockSession(
        completions=[
            [Text(role="assistant", content='"What is the status of my order?"')]
        ]
    )

    simulator = UserSimulator(llm=mock_llm)
    chat_history = _create_conversation_pattern(1)
    user_context = "User wants to know order status"

    result = await simulator.generate(chat_history, user_context)

    assert result.content == "What is the status of my order?"
    assert not result.content.startswith('"')
    assert not result.content.endswith('"')


@pytest.mark.asyncio
async def test_user_simulator_empty_context():
    """Test that UserSimulator handles empty user context."""
    mock_llm = MockSession(
        completions=[[Text(role="assistant", content="I have a question")]]
    )

    simulator = UserSimulator(llm=mock_llm)
    chat_history = _create_conversation_pattern(2)
    user_context = ""

    result = await simulator.generate(chat_history, user_context)

    assert isinstance(result, Text)
    assert result.role == "user"
    assert result.content == "I have a question"


@pytest.mark.asyncio
async def test_user_simulator_fallback_to_dont_know():
    """Test that UserSimulator returns 'I don't know.' for empty response."""
    # Mock LLM that returns empty content
    mock_llm = MockSession(completions=[[Text(role="assistant", content="   ")]])

    simulator = UserSimulator(llm=mock_llm)
    chat_history = _create_conversation_pattern(1)
    user_context = "User has a complex question"

    result = await simulator.generate(chat_history, user_context)

    assert result.content == "I don't know."


def test_user_simulator_can_end_conversation_enabled():
    """Test UserSimulator can be created with can_end_conversation=True."""
    mock_llm = MockSession(completions=[])

    # Should not raise NotImplementedError
    simulator = UserSimulator(llm=mock_llm, can_end_conversation=True)

    assert simulator.can_end_conversation is True
    assert simulator.message_config is not None
    assert simulator.message_config.tag_done == "<DONE>"


def test_user_simulator_can_end_conversation_disabled():
    """Test UserSimulator defaults to can_end_conversation=False."""
    mock_llm = MockSession(completions=[])

    simulator = UserSimulator(llm=mock_llm)

    assert simulator.can_end_conversation is False
    assert simulator.message_config.tag_done == "<DONE>"


@pytest.mark.asyncio
async def test_user_simulator_detects_done_marker():
    """Test that UserSimulator detects <DONE> marker when enabled."""
    mock_llm = MockSession(
        completions=[[Text(role="assistant", content="Thanks for your help! <DONE>")]]
    )

    simulator = UserSimulator(llm=mock_llm, can_end_conversation=True)
    chat_history = _create_conversation_pattern(1)
    user_context = "Get weather information for Seattle"

    result = await simulator.generate(chat_history, user_context)

    assert result.role == "user"
    assert result.content == "Thanks for your help! <DONE>"
    assert result.metadata.get("is_done") is True


@pytest.mark.asyncio
async def test_user_simulator_ignores_done_when_disabled():
    """Test that <DONE> marker is ignored when can_end_conversation=False."""
    mock_llm = MockSession(
        completions=[[Text(role="assistant", content="Thanks! <DONE>")]]
    )

    simulator = UserSimulator(llm=mock_llm, can_end_conversation=False)
    chat_history = _create_conversation_pattern(1)
    user_context = "Simple task"

    result = await simulator.generate(chat_history, user_context)

    assert result.content == "Thanks! <DONE>"
    assert "is_done" not in result.metadata  # Metadata should not be set


@pytest.mark.asyncio
async def test_user_simulator_done_marker_must_be_at_end():
    """Test that <DONE> in middle of message doesn't trigger termination."""
    mock_llm = MockSession(
        completions=[
            [
                Text(
                    role="assistant",
                    content="<DONE> with that, but I have more questions",
                )
            ]
        ]
    )

    simulator = UserSimulator(llm=mock_llm, can_end_conversation=True)
    chat_history = _create_conversation_pattern(1)
    user_context = "Complex task"

    result = await simulator.generate(chat_history, user_context)

    assert "<DONE>" in result.content
    assert not result.content.endswith("<DONE>")  # Not at end
    assert result.metadata.get("is_done") is not True  # Should not trigger


@pytest.mark.asyncio
async def test_user_simulator_sanitizes_done_marker():
    """Test that <DONE> in conversation history is sanitized."""
    mock_llm = MockSession(
        completions=[[Text(role="assistant", content="What's the status?")]]
    )

    # Conversation history contains <DONE>
    chat_history = [
        Text(role="user", content="Tell me about task <DONE>"),
        Text(role="assistant", content="The task marked <DONE> is complete"),
    ]

    simulator = UserSimulator(llm=mock_llm, can_end_conversation=True)
    user_context = "Track progress"

    result = await simulator.generate(chat_history, user_context)

    # Should not falsely detect as done
    assert result.metadata.get("is_done") is not True
    # Check that sanitization happened (indirectly - LLM didn't see literal <DONE>)
    sent_prompt = simulator.history[0][1].content  # the user_msg sent to LLM
    assert "<DONE>" not in sent_prompt


def test_user_simulator_prompts_differ_by_end_capability():
    """Test that ending support changes the user simulator prompt."""
    from thinkingbox.common.user_simulated_answer import (
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_WITH_END,
    )

    assert SYSTEM_PROMPT != SYSTEM_PROMPT_WITH_END
    assert "<DONE>" in SYSTEM_PROMPT_WITH_END
    assert "ENDING THE CONVERSATION" in SYSTEM_PROMPT_WITH_END
    assert "ENDING THE CONVERSATION" not in SYSTEM_PROMPT
