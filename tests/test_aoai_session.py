# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thinkingbox.common.aoai_session import AOAISession
from thinkingbox.common.config_types import AOAISessionConfig, ApiKeyCredentialConfig
from thinkingbox.common.usage_types import (
    InputTokensDetails,
    OutputTokensDetails,
    Usage,
)


def get_test_aoai_config():
    return AOAISessionConfig(
        deployment="test_deployment",
        endpoint_url="https://test.endpoint.com",
        timeout=60.0,
        client_certificate=None,
    )


@pytest.mark.parametrize("reasoning_effort", ["xhigh", "provider-defined"])
def test_config_accepts_provider_defined_reasoning_effort(reasoning_effort):
    config = AOAISessionConfig(
        deployment="test_deployment",
        reasoning_effort=reasoning_effort,
    )

    assert config.reasoning_effort == reasoning_effort


SIMPLE_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "Test response"}}]
}


def get_mock_async_client(response_json):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value=response_json)
    mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.mark.asyncio
@pytest.mark.parametrize("use_certificate", [True, False])
@patch("thinkingbox.common.http_client.Path.is_file", return_value=True)
@patch("thinkingbox.common.http_client.ssl.create_default_context")
@patch("thinkingbox.common.llm_session_base.httpx.AsyncClient")
async def test_client_certificate(
    mock_async_client, mock_ssl_context, _mock_path_is_file, use_certificate
):
    # AsyncClient mock needs to return a mock response for the _get_completion call
    mock_async_client.return_value = get_mock_async_client(SIMPLE_RESPONSE)

    config = get_test_aoai_config()

    if use_certificate:
        # Mock SSL context and certificate loading
        mock_ssl_instance = MagicMock()
        mock_ssl_context.return_value = mock_ssl_instance
        config.client_certificate = "/path/to/client.pem"

    session = AOAISession.from_config(config)

    assert session.ssl_context is None if not use_certificate else not None

    # Call the private _get_completion method
    await session._get_completion()

    # Assert SSL context was not created
    (
        mock_ssl_context.assert_not_called()
        if not use_certificate
        else mock_ssl_context.assert_called_once()
    )

    # Assert AsyncClient was initialized correctly
    mock_async_client.assert_called_once()
    _, kwargs = mock_async_client.call_args
    if use_certificate:
        assert kwargs.get("verify") is session.ssl_context
    else:
        assert kwargs.get("verify") is True


@patch("thinkingbox.common.http_client.Path.is_file")
@patch("thinkingbox.common.http_client.ssl.create_default_context")
def test_get_completion_certificate_file_does_not_exist(
    _mock_ssl_context, _mock_path_is_file
):
    _mock_path_is_file.return_value = False
    config = get_test_aoai_config()
    config.client_certificate = "/invalid/path/to/client.pem"
    with pytest.raises(
        ValueError,
        match=".*/path/to/client.pem.*",
    ):
        AOAISession.from_config(config)


@pytest.mark.asyncio
@patch("thinkingbox.common.llm_session_base.httpx.AsyncClient")
async def test_client_api_key(mock_async_client):
    # AsyncClient mock needs to return a mock response for the _get_completion call
    mock_async_client.return_value = get_mock_async_client(SIMPLE_RESPONSE)

    config = get_test_aoai_config()
    config.credential = ApiKeyCredentialConfig(api_key="MySecret")
    session = AOAISession.from_config(config)
    # Call the private _get_completion method
    await session._get_completion()

    # Assert AsyncClient was initialized correctly
    mock_async_client.assert_called_once()
    _, kwargs = mock_async_client.call_args
    assert kwargs.get("verify") is True


@pytest.mark.asyncio
@patch("thinkingbox.common.llm_session_base.httpx.AsyncClient")
async def test_response_schema_in_payload(mock_async_client):
    """response_schema should build the full response_format in the request payload."""
    mock_async_client.return_value = get_mock_async_client(SIMPLE_RESPONSE)

    session = AOAISession.from_config(get_test_aoai_config())

    from thinkingbox.common.chat_types import Text

    conversation = [Text(role="user", content="Hello")]
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    await session.get_completion(
        conversation=conversation,
        response_schema=schema,
    )

    # Check the payload sent to the API
    post_mock = mock_async_client.return_value.__aenter__.return_value.post
    call_kwargs = post_mock.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response_schema",
            "strict": True,
            "schema": schema,
        },
    }


@pytest.mark.asyncio
@patch("thinkingbox.common.llm_session_base.httpx.AsyncClient")
async def test_xhigh_reasoning_effort_in_payload(mock_async_client):
    mock_async_client.return_value = get_mock_async_client(SIMPLE_RESPONSE)
    config = get_test_aoai_config().model_copy(
        update={"is_reasoning": True, "reasoning_effort": "xhigh"}
    )
    session = AOAISession.from_config(config)

    await session._get_completion()

    post_mock = mock_async_client.return_value.__aenter__.return_value.post
    call_kwargs = post_mock.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
    assert payload["reasoning_effort"] == "xhigh"


@pytest.mark.asyncio
async def test_response_schema_requires_explicit_conversation():
    """response_schema without explicit conversation should raise ValueError."""
    session = AOAISession.from_config(get_test_aoai_config())

    with pytest.raises(ValueError, match="response_schema"):
        await session.get_completion(
            response_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )


@pytest.mark.asyncio
async def test_response_schema_requires_additional_properties_false():
    """response_schema without additionalProperties: false should raise ValueError."""
    session = AOAISession.from_config(get_test_aoai_config())

    from thinkingbox.common.chat_types import Text

    conversation = [Text(role="user", content="Hello")]

    with pytest.raises(ValueError, match="additionalProperties"):
        await session.get_completion(
            conversation=conversation,
            response_schema={"type": "object", "properties": {}},
        )


RESPONSE_WITH_USAGE = {
    "choices": [{"message": {"content": "Test response"}}],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20},
        "completion_tokens_details": {"reasoning_tokens": 10},
    },
}


@pytest.mark.asyncio
@patch("thinkingbox.common.llm_session_base.httpx.AsyncClient")
async def test_get_completion_reports_usage(mock_async_client):
    mock_async_client.return_value = get_mock_async_client(RESPONSE_WITH_USAGE)

    session = AOAISession.from_config(get_test_aoai_config())
    await session._get_completion()

    assert session.last_usage == Usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_tokens_details=InputTokensDetails(cached_tokens=20),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=10),
    )
