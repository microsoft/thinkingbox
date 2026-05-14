# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from functools import partial

import httpx
import pytest

from thinkingbox.common.aoai_responses_session import AOAIResponsesSession
from thinkingbox.common.chat_types import Text


def _mock_client_factory(response_payload):
    class _MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

        async def post(self, url, *args, **kwargs):
            self.last_post_kwargs = kwargs
            self.last_post_url = url
            return httpx.Response(
                status_code=200,
                request=httpx.Request("POST", url),
                content=json.dumps(response_payload),
            )

    return _MockAsyncClient()


SIMPLE_RESPONSE = {
    "id": "resp_1",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello"}],
        }
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    },
}


@pytest.mark.asyncio
async def test_response_schema_in_payload(monkeypatch):
    """response_schema should build the full text.format in the request payload."""
    session = AOAIResponsesSession(
        deployment="test",
        endpoint_url="https://test",
    )

    mock_client = _mock_client_factory(SIMPLE_RESPONSE)
    monkeypatch.setattr(session, "get_client", partial(lambda c: c, mock_client))

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

    payload = mock_client.last_post_kwargs["json"]
    assert payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "response_schema",
            "strict": True,
            "schema": schema,
        }
    }


@pytest.mark.asyncio
async def test_response_schema_requires_explicit_conversation():
    """response_schema without explicit conversation should raise ValueError."""
    session = AOAIResponsesSession(
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


@pytest.mark.asyncio
async def test_response_schema_requires_additional_properties_false(monkeypatch):
    """response_schema without additionalProperties: false should raise ValueError."""
    session = AOAIResponsesSession(
        deployment="test",
        endpoint_url="https://test",
    )

    conversation = [Text(role="user", content="Hello")]

    with pytest.raises(ValueError, match="additionalProperties"):
        await session.get_completion(
            conversation=conversation,
            response_schema={"type": "object", "properties": {}},
        )
