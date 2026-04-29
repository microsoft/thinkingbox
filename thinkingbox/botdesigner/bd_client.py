# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import contextlib
import copy
import json
import os
from typing import Any, AsyncIterator, NamedTuple

import httpx
from pydantic import BaseModel, Field

from thinkingbox.botdesigner.utils import YAMLRenderer
from thinkingbox.common.http_client import (
    BackoffAsyncClient,
    SSEResponseHeaders,
    create_custom_ssl_context,
    get_httpx_client,
)
from thinkingbox.common.utils import (
    CredentialBase,
    raise_for_status_with_error,
)

DEBUG = os.environ.get("THINKINGBOX_DEBUG_BD_CLIENT") == "1"


class StepResult(BaseModel):
    activities: list[dict] = Field(default_factory=list)
    action: str = "continue"
    num_steps: int = 0

    def should_continue(self) -> bool:
        return self.action == "continue"

    def extend(self, other: StepResult):
        self.activities.extend(other.activities)
        self.action = other.action
        self.num_steps += other.num_steps


class BotDesignerRequest(BaseModel):
    bot_id: str
    headers: dict[str, str] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    feature_overrides: dict[str, Any] = Field(default_factory=dict)
    bot_definition_override: dict[str, Any] | None = None
    connector_endpoint_override: str | None = None

    def build(self) -> tuple[dict[str, Any], dict[str, str]]:
        headers = self.headers.copy()
        feature_overrides = self.feature_overrides.copy()
        payload: dict[str, Any] = self.payload.copy()
        if self.bot_definition_override is not None:
            data = copy.deepcopy(self.bot_definition_override)
            payload["botDefinitionOverrideYaml"] = YAMLRenderer().render(data)
            feature_overrides["BotProvider.EnableBotDefinitionOverride"] = True
        if self.connector_endpoint_override is not None:
            feature_overrides[
                "ConnectorCallClient.ConnectorEndpointOverrideEnabled"
            ] = True
            feature_overrides[
                "RuntimeConnectorContextProvider.RuntimeOverrideEndpoint"
            ] = self.connector_endpoint_override
        if feature_overrides:
            headers["x-ms-feature-overrides"] = json.dumps(feature_overrides)
        return payload, headers


class ConversationContext(BaseModel):
    conversation_id: str
    result: StepResult
    req: BotDesignerRequest
    raw_response: dict[str, Any]


class BotDesignerResponse(NamedTuple):
    headers: dict
    data: dict


class BotDesignerClient:
    def __init__(
        self,
        endpoint: str,
        environment_id: str,
        base_bot_id: str,
        feature_overrides: dict[str, Any] | None = None,
        timeout: float = 120.0,
        use_dns_cache: bool = False,
        credential: CredentialBase | None = None,
        client_certificate: str | None = None,
        trust_ca_path: str | None = None,
        headers: dict[str, str] | None = None,
        max_retries_server_error: int = 5,
        retryable_server_errors: tuple[int | str, ...] = (502, 503),
        use_sse_protocol: bool = False,
    ):
        self._base_url = endpoint.rstrip("/")
        self._environment_id = environment_id
        self._base_request = BotDesignerRequest(
            bot_id=base_bot_id,
            headers={},
            feature_overrides=feature_overrides or {},
        )
        self._use_sse_protocol = use_sse_protocol
        self.use_dns_cache = use_dns_cache
        self.credential = credential
        self.max_retries_server_error = max_retries_server_error
        self.retryable_server_errors = retryable_server_errors
        if headers:
            self._base_request.headers.update(headers)
        self.timeout_config = httpx.Timeout(
            timeout=timeout,
            connect=30.0,
            read=timeout,
            write=30.0,
        )
        self.ssl_context = None
        if client_certificate is not None or trust_ca_path is not None:
            self.ssl_context = create_custom_ssl_context(
                client_certificate=client_certificate,
                trust_ca_path=trust_ca_path,
            )

    @contextlib.asynccontextmanager
    async def _get_http_client(self, headers: dict[str, str] | None = None):
        verify = self.ssl_context if self.ssl_context else True
        client_kwargs = {}
        if headers:
            client_kwargs["headers"] = headers
        async with get_httpx_client(
            use_dns_cache=self.use_dns_cache,
            timeout=self.timeout_config,
            verify=verify,
            **client_kwargs,
        ) as client:
            yield BackoffAsyncClient(
                client,
                credential=self.credential,
                max_retries_server_error=self.max_retries_server_error,
                retryable_server_errors=self.retryable_server_errors,
                # on timeout, the operation might have happened on the server.
                # Do not re-try.
                max_retries_timeout=0,
            )

    def _maybe_log_request(
        self, url: str, payload: dict[str, Any], headers: dict[str, Any] | None = None
    ):
        # TODO remove this and add feature to log all HTTP requests from BackoffAsyncClient
        # to a JSONL file instead
        if not DEBUG:
            return
        print(f">>> POST {url}")
        tmp = payload.copy()
        if isinstance(tmp.get("botDefinitionOverrideYaml"), str):
            tmp["botDefinitionOverrideYaml"] = (
                tmp["botDefinitionOverrideYaml"][:10] + "..."
            )
        print("data:", json.dumps(tmp))
        print("headers:", json.dumps(headers), flush=True)

    def _maybe_log_response(self, resp: Any):
        if not DEBUG:
            return
        print(f"<<< {resp!r}", flush=True)

    async def _post(
        self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> BotDesignerResponse:
        self._maybe_log_request(url=url, payload=payload, headers=headers)
        if self._use_sse_protocol:
            sse_headers = {
                "accept": "text/event-stream,application/json;q=0.9,*/*;q=0.8",
            }
            return await self._sse_post(
                url=url, payload=payload, headers={**headers, **sse_headers}
            )
        async with self._get_http_client(headers=headers) as client:
            r = await client.post(url, json=payload)
        raise_for_status_with_error(r)
        out = BotDesignerResponse(headers=dict(r.headers.items()), data=r.json())
        self._maybe_log_response(out)
        return out

    def _sse_should_keep_activity(self, data: dict[str, Any]):
        activity_type = data.get("type")
        if activity_type in (
            "event",
            "message",
        ):
            return True
        # we only need typing activities for tool call events
        # avoid storing all typing chunks.
        # TODO why does BotDesigner provide tool calls as "typing" instead of events?
        if activity_type == "typing" and data.get("entities"):
            return True
        return False

    async def _sse_post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> BotDesignerResponse:
        self._maybe_log_request(url=url, payload=payload, headers=headers)
        out = BotDesignerResponse(
            headers={},
            data={
                "activities": [],
            },
        )
        async with self._get_http_client(headers=headers) as client:
            async for chunk in client.post_iter_sse_chunks(url, json=payload):
                self._maybe_log_response(chunk)
                if isinstance(chunk, SSEResponseHeaders):
                    out.headers.update(chunk.data)
                    continue
                if chunk.data.strip() == "end":
                    continue
                data = json.loads(chunk.data)
                if not isinstance(data, dict):
                    continue
                if chunk.event_type == "activity":
                    if self._sse_should_keep_activity(data):
                        out.data["activities"].append(data)
                if "action" in data:
                    out.data["action"] = data["action"]
        return out

    async def conversation_start(
        self,
        emit_start_conversation_event: bool = False,
        locale: str = "en-US",
        test_mode: str = "Text",
        bot_definition_override: dict[str, Any] | None = None,
        connector_endpoint_override: str | None = None,
    ) -> ConversationContext:
        connector_endpoint_override = (
            connector_endpoint_override.rstrip("/") + "/"
            if connector_endpoint_override
            else None
        )
        req = self._base_request.model_copy(deep=True)

        req.payload["emitStartConversationEvent"] = emit_start_conversation_event
        req.payload["locale"] = locale
        req.payload["testMode"] = test_mode
        req.bot_definition_override = bot_definition_override
        req.connector_endpoint_override = connector_endpoint_override

        payload, headers = req.build()
        url = f"{self._base_url}/environments/{self._environment_id}/bots/{req.bot_id}/test/conversations"
        resp = await self._post(url, payload=payload, headers=headers)
        conversation_id = resp.headers["x-ms-conversationid"]

        new_req = self._base_request.model_copy(deep=True)
        new_req.bot_definition_override = bot_definition_override
        new_req.connector_endpoint_override = connector_endpoint_override
        new_req.headers["x-ms-conversationid"] = conversation_id
        return ConversationContext(
            conversation_id=conversation_id,
            result=StepResult(
                activities=resp.data.get("activities", []),
                action=resp.data.get("action", "waiting").lower(),
                num_steps=1,
            ),
            req=new_req,
            raw_response=resp.data,
        )

    async def conversation_continue_iter(
        self, conversation: ConversationContext, limit: int = -1
    ) -> AsyncIterator[StepResult]:
        # TODO is this even needed with streaming?
        req = conversation.req.model_copy(deep=True)
        url = f"{self._base_url}/environments/{self._environment_id}/bots/{conversation.req.bot_id}/test/conversations/{conversation.conversation_id}/continue"
        payload, headers = req.build()
        i: int = 0
        while True:
            resp = await self._post(url, payload=payload, headers=headers)
            activities = resp.data.get("activities", [])
            result = StepResult(
                activities=activities,
                action=resp.data.get("action", "waiting").lower(),
                num_steps=1,
            )
            yield result
            if not result.should_continue():
                break
            i += 1
            if limit != -1 and i >= limit:
                break

    async def conversation_continue(
        self, conversation: ConversationContext, limit: int = -1
    ) -> StepResult:
        merged_result: StepResult = StepResult()
        async for result in self.conversation_continue_iter(
            conversation=conversation, limit=limit
        ):
            merged_result.extend(result)
        return merged_result

    async def conversation_send_message(
        self, conversation: ConversationContext, activity: dict
    ) -> StepResult:
        req = conversation.req.model_copy(deep=True)
        req.payload["activity"] = activity
        url = f"{self._base_url}/environments/{self._environment_id}/bots/{conversation.req.bot_id}/test/conversations/{conversation.conversation_id}"
        payload, headers = req.build()
        resp = await self._post(url, payload=payload, headers=headers)
        action = resp.data.get("action", "waiting").lower()
        return StepResult(
            activities=resp.data.get("activities", []),
            action=action,
            num_steps=1,
        )
