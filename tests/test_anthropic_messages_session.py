# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from functools import partial

import httpx
import pytest
import yaml

from thinkingbox.common.anthropic_messages_session import AnthropicMessagesSession
from thinkingbox.common.chat_types import (
    ParallelToolCall,
    Text,
    ToolCall,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import AnthropicMessagesSessionConfig

CONFIG = """
type: anthropic
credential:
    api_key: "anthropic_api_key"
deployment: claude-test
endpoint_url: "https://some.anthropic.site/api/messages"
temperature: 1.0
max_completion_tokens: 4096
timeout: 600.0
"""


def get_test_anthropic_config():
    config = yaml.safe_load(CONFIG)
    return AnthropicMessagesSessionConfig.model_validate(config)


def get_tools():
    return [
        ToolDef(
            name="get_stock_price",
            description="Get the current stock price for a given ticker symbol.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                    }
                },
                "required": ["ticker"],
            },
        )
    ]


def _mock_client_factory(response_payload):
    class _MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def post(self, url, *args, **kwargs):
            return httpx.Response(
                status_code=200,
                request=httpx.Request("POST", url),
                content=json.dumps(response_payload),
            )

    return _MockAsyncClient()


@pytest.mark.asyncio
async def test_anthropic_call(monkeypatch):
    config = get_test_anthropic_config()
    session = AnthropicMessagesSession.from_config(config)
    conversation = [Text(role="user", content="Hello, how are you?")]

    mock_response = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! I'm fine."}],
    }
    monkeypatch.setattr(
        session,
        "get_client",
        partial(_mock_client_factory, mock_response),
    )

    response = await session.get_completion(
        conversation=conversation,
        update_conversation=False,
    )

    assert response is not None
    assert len(response) == 1
    assert response[0].role == "assistant"
    assert isinstance(response[0].content, str)
    assert len(response[0].content) > 0


@pytest.mark.asyncio
async def test_anthropic_call_with_tools(monkeypatch):
    config = get_test_anthropic_config()
    session = AnthropicMessagesSession.from_config(config)
    session.add_tools(get_tools())
    conversation = [Text(role="user", content="What's the SPY at today?")]

    mock_response = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check."},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "get_stock_price",
                "input": {"ticker": "SPY"},
            },
        ],
    }
    monkeypatch.setattr(
        session,
        "get_client",
        partial(_mock_client_factory, mock_response),
    )

    response = await session.get_completion(
        conversation=conversation,
        update_conversation=False,
    )

    assert response is not None
    assert len(response) == 2
    # Text alongside tool calls
    assert isinstance(response[0], Text)
    assert response[0].tag == "text"
    assert response[0].content == "Let me check."
    # Tool call
    tool_call = response[1]
    assert isinstance(tool_call, ParallelToolCall)
    assert len(tool_call.tool_calls) == 1
    assert tool_call.tool_calls[0].name == "get_stock_price"
    assert tool_call.tool_calls[0].arguments == {"ticker": "SPY"}


@pytest.mark.asyncio
async def test_anthropic_call_with_thinking(monkeypatch):
    config = get_test_anthropic_config()
    session = AnthropicMessagesSession.from_config(config)
    conversation = [Text(role="user", content="What is 2+2?")]

    mock_response = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants to add 2+2, which is 4.",
                "signature": "sig_abc123",
            },
            {"type": "text", "text": "2+2 equals 4."},
        ],
    }
    monkeypatch.setattr(
        session,
        "get_client",
        partial(_mock_client_factory, mock_response),
    )

    response = await session.get_completion(
        conversation=conversation,
        update_conversation=False,
    )

    assert len(response) == 2
    # Thinking block
    assert isinstance(response[0], Text)
    assert response[0].tag == "think"
    assert response[0].content == "The user wants to add 2+2, which is 4."
    assert response[0].metadata["thinking_signature"] == "sig_abc123"
    assert not response[0].is_visible
    # Text block
    assert isinstance(response[1], Text)
    assert response[1].tag == "text"
    assert response[1].content == "2+2 equals 4."
    assert response[1].is_visible


@pytest.mark.asyncio
async def test_anthropic_call_thinking_with_tools(monkeypatch):
    """Thinking + text + tool_use"""
    config = get_test_anthropic_config()
    session = AnthropicMessagesSession.from_config(config)
    session.add_tools(get_tools())
    conversation = [Text(role="user", content="What's SPY at?")]

    mock_response = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "I should look up the stock price.",
                "signature": "sig_xyz",
            },
            {"type": "text", "text": "Let me check the stock price."},
            {
                "type": "tool_use",
                "id": "call_2",
                "name": "get_stock_price",
                "input": {"ticker": "SPY"},
            },
        ],
    }
    monkeypatch.setattr(
        session,
        "get_client",
        partial(_mock_client_factory, mock_response),
    )

    response = await session.get_completion(
        conversation=conversation,
        update_conversation=False,
    )

    assert len(response) == 3
    # Thinking
    assert isinstance(response[0], Text)
    assert response[0].tag == "think"
    # Text alongside tool calls
    assert isinstance(response[1], Text)
    assert response[1].tag == "text"
    assert response[1].content == "Let me check the stock price."
    # Tool call
    assert isinstance(response[2], ParallelToolCall)
    assert response[2].tool_calls[0].name == "get_stock_price"


def test_thinking_round_trip_encode():
    """Thinking blocks should round-trip through encode/decode preserving signature."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    think_msg = Text(
        role="assistant",
        content="some reasoning",
        metadata={"tag": "think", "thinking_signature": "sig_123"},
    )
    encoded = session._encode_message(think_msg)
    assert len(encoded) == 1
    assert encoded[0]["role"] == "assistant"
    block = encoded[0]["content"][0]
    assert block["type"] == "thinking"
    assert block["thinking"] == "some reasoning"
    assert block["signature"] == "sig_123"


def test_redacted_thinking_round_trip():
    """Redacted thinking blocks should round-trip preserving the data blob."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    # Decode
    msg = {
        "role": "assistant",
        "content": [
            {"type": "redacted_thinking", "data": "encrypted_blob_data"},
            {"type": "text", "text": "Here is my answer."},
        ],
    }
    decoded = session._decode_message(msg)
    assert len(decoded) == 2
    assert decoded[0].tag == "think"
    assert decoded[0].metadata["redacted_thinking"] == "encrypted_blob_data"

    # Re-encode the redacted thinking
    encoded = session._encode_message(decoded[0])
    assert len(encoded) == 1
    block = encoded[0]["content"][0]
    assert block["type"] == "redacted_thinking"
    assert block["data"] == "encrypted_blob_data"


def test_thinking_payload_config():
    """thinking and output_config should be forwarded to request payload."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
    )
    session.conversation = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    payload, _ = session._encode_payload()
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "medium"}


def test_disable_parallel_tool_use_payload():
    """parallel_tool_calls=False should set tool_choice with disable_parallel_tool_use."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    session.add_tools(get_tools())
    session.conversation = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    session.parallel_tool_calls = False
    payload, _ = session._encode_payload()
    assert payload["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_parallel_tool_use_payload():
    """parallel_tool_calls=True should not set tool_choice."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    session.add_tools(get_tools())
    session.conversation = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    session.parallel_tool_calls = True
    payload, _ = session._encode_payload()
    assert "tool_choice" not in payload


def test_no_tool_choice_without_tools():
    """Without tools, tool_choice should not be in payload."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    session.conversation = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    session.parallel_tool_calls = False
    payload, _ = session._encode_payload()
    assert "tool_choice" not in payload


def test_parallel_tool_responses_merged_into_single_user_message():
    """Multiple ToolResponses must be merged into one user message for the Anthropic API."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    session.add_tools(get_tools())

    # Simulate: user message, assistant with parallel tool calls, then two tool responses
    user_msg = Text(role="user", content="What's SPY and AAPL?")
    parallel_tc = ParallelToolCall(
        tool_calls=[
            ToolCall(name="get_stock_price", arguments={"ticker": "SPY"}, id="call_1"),
            ToolCall(name="get_stock_price", arguments={"ticker": "AAPL"}, id="call_2"),
        ]
    )
    tr1 = ToolResponse(name="get_stock_price", content='{"price": 500}', id="call_1")
    tr2 = ToolResponse(name="get_stock_price", content='{"price": 200}', id="call_2")

    session.add_messages([user_msg, parallel_tc, tr1, tr2])
    session.parallel_tool_calls = True
    payload, _ = session._encode_payload()

    messages = payload["messages"]
    # Should be: user, assistant, user (with both tool_results merged)
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    # The merged user message should contain both tool results
    assert len(messages[2]["content"]) == 2
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == "call_1"
    assert messages[2]["content"][1]["type"] == "tool_result"
    assert messages[2]["content"][1]["tool_use_id"] == "call_2"


def test_thinking_and_tool_calls_merged_into_single_assistant_message():
    """Thinking + text + tool calls must be one assistant message."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    # Encode: think, text, and tool calls (as separate Message objects)
    think_msg = Text(
        role="assistant",
        content="reasoning",
        metadata={"tag": "think", "thinking_signature": "sig"},
    )
    text_msg = Text(
        role="assistant",
        content="Let me check.",
        metadata={"tag": "text"},
    )
    tool_msg = ParallelToolCall(
        tool_calls=[
            ToolCall(name="get_stock_price", arguments={"ticker": "SPY"}, id="call_1"),
        ]
    )
    user_msg = Text(role="user", content="What's SPY?")
    session.add_messages([user_msg, think_msg, text_msg, tool_msg])
    payload, _ = session._encode_payload()

    messages = payload["messages"]
    assert len(messages) == 2  # user, assistant (merged)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    # All three blocks in one assistant message
    content = messages[1]["content"]
    assert len(content) == 3
    assert content[0]["type"] == "thinking"
    assert content[1]["type"] == "text"
    assert content[2]["type"] == "tool_use"


def test_no_thinking_payload_without_config():
    """When thinking, output_config are not set, they should not appear in payload."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    session.conversation = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    payload, _ = session._encode_payload()
    assert "thinking" not in payload
    assert "output_config" not in payload


def test_sanitize_tool_schema_dollar_keys():
    """Tool schemas with $-prefixed argument names should be sanitized."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    tool = ToolDef(
        name="odata_query",
        description="Run an OData query",
        input_schema={
            "type": "object",
            "properties": {
                "$filter": {"type": "string", "description": "OData filter"},
                "$top": {"type": "integer", "description": "Limit results"},
                "table": {"type": "string"},
            },
            "required": ["$filter", "table"],
        },
    )
    session.add_tools([tool])

    # Check the encoded tool has sanitized keys
    encoded = session.tools[0]
    props = encoded["input_schema"]["properties"]
    assert "_x24filter" in props
    assert "_x24top" in props
    assert "table" in props
    assert "$filter" not in props

    # required list should also be sanitized
    assert "_x24filter" in encoded["input_schema"]["required"]
    assert "table" in encoded["input_schema"]["required"]
    assert "$filter" not in encoded["input_schema"]["required"]

    # Mapping should be stored
    assert session._tool_arg_mappings["odata_query"] == {
        "_x24filter": "$filter",
        "_x24top": "$top",
    }


def test_sanitize_tool_schema_collision_raises():
    """Colliding sanitized keys should raise ValueError."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    # "a b" sanitizes to "a_x20b", which collides with the literal key "a_x20b"
    tool = ToolDef(
        name="collider",
        description="tool with colliding keys",
        input_schema={
            "type": "object",
            "properties": {
                "a b": {"type": "string"},
                "a_x20b": {"type": "string"},
            },
        },
    )
    with pytest.raises(ValueError, match="collision"):
        session.add_tools([tool])


@pytest.mark.asyncio
async def test_sanitized_args_reverse_mapped_on_decode(monkeypatch):
    """Tool call arguments with sanitized keys should be reverse-mapped."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    tool = ToolDef(
        name="odata_query",
        description="Run an OData query",
        input_schema={
            "type": "object",
            "properties": {
                "$filter": {"type": "string"},
                "table": {"type": "string"},
            },
            "required": ["$filter", "table"],
        },
    )
    session.add_tools([tool])

    # Simulate API response using sanitized key names (as the model would)
    mock_response = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "odata_query",
                "input": {"_x24filter": "status eq 'active'", "table": "orders"},
            },
        ],
    }
    monkeypatch.setattr(
        session,
        "get_client",
        partial(_mock_client_factory, mock_response),
    )

    conversation = [Text(role="user", content="Query orders")]
    response = await session.get_completion(
        conversation=conversation, update_conversation=False
    )

    tc = response[0]
    assert isinstance(tc, ParallelToolCall)
    # Arguments should have original $-prefixed keys
    assert tc.tool_calls[0].arguments == {
        "$filter": "status eq 'active'",
        "table": "orders",
    }


def test_add_tools_replaces_existing():
    """Calling add_tools() a second time should replace existing tools without error."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    session.add_tools(get_tools())
    assert len(session.tools) == 1
    assert session.tools[0]["name"] == "get_stock_price"
    original_desc = session.tools[0]["description"]

    # Replace with updated description
    updated_tools = [
        ToolDef(
            name="get_stock_price",
            description="Updated description.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        )
    ]
    session.add_tools(updated_tools)

    # Should still be one tool, not two
    assert len(session.tools) == 1
    assert session.tools[0]["name"] == "get_stock_price"
    assert session.tools[0]["description"] == "Updated description."
    assert session.tools[0]["description"] != original_desc


def test_no_mapping_for_clean_tools():
    """Tools with valid keys should produce no mapping."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    session.add_tools(get_tools())
    assert "get_stock_price" not in session._tool_arg_mappings


def test_response_schema_in_payload():
    """response_schema should build output_config.format in the payload."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    conversation = [Text(role="user", content="Hello")]
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    payload, _ = session._encode_payload(
        conversation=conversation,
        response_schema=schema,
    )
    assert payload["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }


def test_response_schema_merges_with_existing_output_config():
    """response_schema should merge into existing output_config."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
        output_config={"effort": "medium"},
    )
    conversation = [Text(role="user", content="Hello")]
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    payload, _ = session._encode_payload(
        conversation=conversation,
        response_schema=schema,
    )
    assert payload["output_config"]["effort"] == "medium"
    assert payload["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }


@pytest.mark.asyncio
async def test_response_schema_requires_explicit_conversation():
    """response_schema without explicit conversation should raise ValueError."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    with pytest.raises(ValueError, match="response_schema"):
        await session.get_completion(
            response_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )


def test_response_schema_requires_additional_properties_false():
    """response_schema without additionalProperties: false should raise ValueError."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    conversation = [Text(role="user", content="Hello")]

    with pytest.raises(ValueError, match="additionalProperties"):
        session._encode_payload(
            conversation=conversation,
            response_schema={"type": "object", "properties": {}},
        )


def test_anthropic_config_top_k_negative():
    config = {
        "type": "anthropic",
        "deployment": "claude-test",
        "endpoint_url": "https://some.anthropic.site/api/messages",
        "top_k": -1,
    }
    try:
        AnthropicMessagesSessionConfig.model_validate(config)
        assert False, "Expected ValueError for top_k < 0"
    except ValueError as e:
        assert "top_k must be >= 0" in str(e)


def test_anthropic_config_top_p_out_of_range():
    config = {
        "type": "anthropic",
        "deployment": "claude-test",
        "endpoint_url": "https://some.anthropic.site/api/messages",
        "top_p": 1.5,
    }
    try:
        AnthropicMessagesSessionConfig.model_validate(config)
        assert False, "Expected ValueError for top_p out of range"
    except ValueError as e:
        assert "top_p must be between 0.0 and 1.0" in str(e)


def test_anthropic_config_temperature_and_top_p_set():
    config = {
        "type": "anthropic",
        "deployment": "claude-test",
        "endpoint_url": "https://some.anthropic.site/api/messages",
        "temperature": 0.5,
        "top_p": 0.5,
    }
    try:
        AnthropicMessagesSessionConfig.model_validate(config)
        assert False, "Expected ValueError for both temperature and top_p set"
    except ValueError as e:
        assert "temperature and top_p cannot both be set at the same time" in str(e)


def test_thinking_only_response_appends_dummy():
    """A response with only thinking blocks should get a dummy text message appended."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    msg = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "I need to think about this.",
                "signature": "sig_abc",
            },
        ],
    }
    decoded = session._decode_message(msg)
    assert len(decoded) == 2
    # First is the thinking block
    assert isinstance(decoded[0], Text)
    assert decoded[0].tag == "think"
    # Second is the dummy
    assert isinstance(decoded[1], Text)
    assert decoded[1].tag == "text"
    assert decoded[1].content == ""
    assert decoded[1].metadata.get("is_dummy") is True


def test_dummy_message_filtered_from_payload():
    """Dummy messages should be dropped when encoding to API payload."""
    session = AnthropicMessagesSession(
        deployment="test",
        endpoint_url="https://test",
    )
    think_msg = Text(
        role="assistant",
        content="reasoning",
        metadata={"tag": "think", "thinking_signature": "sig"},
    )
    dummy_msg = Text(
        role="assistant",
        content="",
        metadata={"is_dummy": True},
    )
    user_msg = Text(role="user", content="Thanks")

    session.add_messages([user_msg, think_msg, dummy_msg, user_msg])
    payload, _ = session._encode_payload()

    messages = payload["messages"]
    # Should be: user, assistant (thinking only), user — dummy is gone
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "thinking"
    assert messages[2]["role"] == "user"


@pytest.mark.asyncio
async def test_last_usage_set_after_completion(monkeypatch):
    """last_usage should be populated from the response usage field."""
    config = get_test_anthropic_config()
    session = AnthropicMessagesSession.from_config(config)
    conversation = [Text(role="user", content="Hello")]

    mock_response = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there."}],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 10,
        },
    }
    monkeypatch.setattr(
        session,
        "get_client",
        partial(_mock_client_factory, mock_response),
    )

    await session.get_completion(conversation=conversation, update_conversation=False)

    assert session.last_usage is not None
    assert session.last_usage.input_tokens == 100
    assert session.last_usage.output_tokens == 50
    assert session.last_usage.input_tokens_details.cached_tokens == 10
    assert session.last_usage.total_tokens == 150
