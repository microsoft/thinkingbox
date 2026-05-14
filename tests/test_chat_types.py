# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from thinkingbox.common.chat_types import (
    ParallelToolCall,
    Text,
    ToolCall,
    ToolResponse,
    get_conversation_transcript,
)


def _create_conversation_pattern(num_turns: int) -> list:
    """Create a test conversation"""
    messages = [Text(role="system", content="You are a helpful assistant")]

    for i in range(num_turns):
        messages.append(Text(role="user", content=f"User message {i+1}"))
        messages.append(
            Text(
                role="assistant",
                content=f"Thinking about message {i+1}",
                metadata={"tag": "think"},
            )
        )
        messages.append(
            ParallelToolCall(
                tool_calls=[
                    ToolCall(
                        name=f"tool_{i+1}",
                        arguments={"arg": f"value_{i+1}"},
                        metadata={"error": None},
                    )
                ]
            )
        )
        messages.append(ToolResponse(name=f"tool_{i+1}", content=f"Tool result {i+1}"))
        messages.append(
            Text(
                role="assistant",
                content=f"More thinking {i+1}",
                metadata={"tag": "think"},
            )
        )
        messages.append(
            Text(
                role="assistant",
                content=f"Assistant response {i+1}",
                metadata={"tag": "text"},
            )
        )

    return messages


def test_get_conversation_transcript_limit_1():
    """Test with limit=1, should only include the last assistant message."""
    messages = _create_conversation_pattern(3)

    transcript = get_conversation_transcript(messages, limit=1)

    # Should only include the last assistant message (no user message after it)
    expected = "Assistant: Assistant response 3"
    assert transcript == expected


def test_get_conversation_transcript_5_limit_3():
    """Test with limit=3 when there are more than 3 assistant messages."""
    messages = _create_conversation_pattern(5)

    transcript = get_conversation_transcript(messages, limit=3)

    # Should include last 3 assistant messages (counting backwards from end)
    # Stops after the 3rd assistant message from the end
    expected = (
        "Assistant: Assistant response 3\n\n"
        "User: User message 4\n\n"
        "Assistant: Assistant response 4\n\n"
        "User: User message 5\n\n"
        "Assistant: Assistant response 5"
    )
    assert transcript == expected


def test_get_conversation_transcript_2_limit_3():
    """Test with limit=3 when there are less than 3 assistant messages."""
    messages = _create_conversation_pattern(2)

    transcript = get_conversation_transcript(messages, limit=3)

    # Should include all messages since there are only 2 assistant messages
    expected = (
        "User: User message 1\n\n"
        "Assistant: Assistant response 1\n\n"
        "User: User message 2\n\n"
        "Assistant: Assistant response 2"
    )
    assert transcript == expected


def test_get_conversation_transcript_with_transform():
    """Test using content_transform_fn to transform message content."""
    messages = _create_conversation_pattern(2)

    # Transform function to uppercase the content
    def uppercase_transform(content: str) -> str:
        return content.upper()

    transcript = get_conversation_transcript(
        messages, limit=0, content_transform_fn=uppercase_transform
    )

    # All content should be uppercase
    expected = (
        "User: USER MESSAGE 1\n\n"
        "Assistant: ASSISTANT RESPONSE 1\n\n"
        "User: USER MESSAGE 2\n\n"
        "Assistant: ASSISTANT RESPONSE 2"
    )
    assert transcript == expected


def test_get_conversation_transcript_full_conversation():
    """Test with limit=0, should include all messages."""
    messages = _create_conversation_pattern(3)

    transcript = get_conversation_transcript(messages, limit=0)

    # Should include all user and assistant messages (excluding system and think)
    expected = (
        "User: User message 1\n\n"
        "Assistant: Assistant response 1\n\n"
        "User: User message 2\n\n"
        "Assistant: Assistant response 2\n\n"
        "User: User message 3\n\n"
        "Assistant: Assistant response 3"
    )
    assert transcript == expected


def test_get_conversation_transcript_excludes_thinking_messages():
    """Test that thinking messages are excluded from transcript."""
    messages = _create_conversation_pattern(1)

    transcript = get_conversation_transcript(messages, limit=0)

    # Should not include thinking messages or system messages
    assert "Thinking" not in transcript
    assert "You are a helpful assistant" not in transcript
    assert "User message 1" in transcript
    assert "Assistant response 1" in transcript


def test_get_conversation_transcript_limit_negative_raises_error():
    """Test that negative limit raises ValueError."""
    messages = _create_conversation_pattern(1)

    try:
        get_conversation_transcript(messages, limit=-1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "limit must be >= 0" in str(e)


def test_get_conversation_transcript_empty_messages():
    """Test with empty messages list."""
    messages: list = []

    transcript = get_conversation_transcript(messages, limit=0)

    assert transcript == ""


def test_get_conversation_transcript_content_transform_with_limit():
    """Test using content_transform_fn with limit parameter."""
    messages = _create_conversation_pattern(3)

    # Transform function to add prefix
    def add_prefix(content: str) -> str:
        return f"[TRANSFORMED] {content}"

    transcript = get_conversation_transcript(
        messages, limit=2, content_transform_fn=add_prefix
    )

    # Should include last 2 assistant messages with transformation
    # Counting backwards: stops after 2nd assistant from end
    expected = (
        "Assistant: [TRANSFORMED] Assistant response 2\n\n"
        "User: [TRANSFORMED] User message 3\n\n"
        "Assistant: [TRANSFORMED] Assistant response 3"
    )
    assert transcript == expected


def test_dummy_message_filtered_from_transcript():
    """Dummy messages should be excluded from get_conversation_transcript."""

    messages = [
        Text(role="user", content="Hello"),
        Text(
            role="assistant",
            content="thinking...",
            metadata={"tag": "think"},
        ),
        Text(
            role="assistant",
            content="",
            metadata={"is_dummy": True},
        ),
    ]
    transcript = get_conversation_transcript(messages)
    assert "User: Hello" in transcript
    # Dummy should not appear
    assert "Assistant: " not in transcript
    # Only user message
    assert transcript.strip() == "User: Hello"
