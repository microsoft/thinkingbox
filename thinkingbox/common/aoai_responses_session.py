# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import json
from enum import Enum
from typing import Any, Literal

import httpx

from thinkingbox.common.chat_types import (
    Message,
    ParallelToolCall,
    Text,
    ToolCall,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import AOAIResponsesSessionConfig
from thinkingbox.common.credential_factory import create_credential
from thinkingbox.common.llm_session_base import HTTPLLMSessionBase
from thinkingbox.common.usage_types import Usage
from thinkingbox.common.utils import (
    CredentialBase,
    raise_for_status_with_error,
    syncify,
)


class ReasoningSource(Enum):
    NONE = "none"
    SUMMARY = "summary"
    CONTENT = "content"


def _encode_tool(tool: ToolDef) -> dict:
    """encode ToolDef -> AOAI Responses tools"""
    parameters_schema = copy.deepcopy(tool.input_schema)
    parameters_schema["additionalProperties"] = False
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters_schema,
    }


def _convert_function_call(tc: dict) -> ToolCall:
    error = None
    try:
        arguments = json.loads(tc["arguments"])
    except (ValueError, TypeError) as e:
        arguments = {}
        error = f"Error: {e!s}"
    if not isinstance(arguments, dict):
        error = "Error: tool arguments must be an object"
    return ToolCall(
        name=tc["name"],
        arguments=arguments,
        id=tc["call_id"],
        metadata={"error": error},
    )


def _clean_up_output_message_inplace(msg: dict):
    """Fix output message so that it can be passed back as input"""
    if msg["type"] == "message" and isinstance(msg.get("content"), list):
        for part in msg["content"]:
            # vLLM returns logprobs=null when not requesting logprobs
            # however its request schema validates that logprobs must be a list
            # remove logprobs if null, so that the response message can be passed
            # back as part of the conversation prefix for the next request
            if "logprobs" in part and part["logprobs"] is None:
                del part["logprobs"]


def get_message_reasoning_text(msg: dict, source: ReasoningSource):
    """Extract reasoning text from responses API message."""
    if source is ReasoningSource.SUMMARY:
        summary = msg.get("summary")
        if not summary:
            return "(hidden)"
        out = []
        for s in summary:
            if s["type"] == "summary_text":
                out.append(s["text"])
        return "\n".join(out)
    elif source is ReasoningSource.CONTENT:
        content = msg.get("content")
        if not content:
            return ""
        out = []
        for s in content:
            if s["type"] == "reasoning_text":
                out.append(s["text"])
        return "\n".join(out)
    assert source is ReasoningSource.NONE
    return ""


def get_message_content_text(msg: dict):
    content = msg.get("content")
    if not content:
        return ""
    out = []
    for s in content:
        if s["type"] == "output_text":
            out.append(s["text"])
    return "\n".join(out)


class AOAIResponsesSession(HTTPLLMSessionBase):
    def __init__(
        self,
        deployment: str,
        credential: CredentialBase | None = None,
        client_certificate: str | None = None,
        trust_ca_path: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        use_dns_cache: bool = False,
        max_retries_server_error: int = 5,
        retryable_server_errors: tuple[int | str, ...] = (502, 503, 504),
        account_name: str | None = None,
        endpoint_url: str | None = None,
        api_version: str = "2024-10-21",
        seed: int | None = None,
        temperature: float = 1.0,
        max_completion_tokens: int = 4096,
        is_reasoning: bool = False,
        reasoning_source: str | ReasoningSource = ReasoningSource.SUMMARY,
        reasoning_effort: str | None = None,
        use_stateful_protocol: bool = False,
        parallel_tool_calls: bool = False,
    ):
        super().__init__(
            credential=credential,
            client_certificate=client_certificate,
            trust_ca_path=trust_ca_path,
            headers=headers,
            timeout=timeout,
            use_dns_cache=use_dns_cache,
            max_retries_server_error=max_retries_server_error,
            retryable_server_errors=retryable_server_errors,
        )
        self.aoai_mode: bool = True
        if endpoint_url is not None:
            self.aoai_mode = False

        if self.aoai_mode:
            if account_name is None:
                raise ValueError(
                    "AOAI mode (endpoint_url=null): account_name is required"
                )
            self.responses_url = f"https://{account_name}.openai.azure.com/openai/responses?api-version={api_version}"
        else:
            self.responses_url = endpoint_url

        self.deployment = deployment
        self.seed = seed
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.is_reasoning = is_reasoning
        self.reasoning_source = ReasoningSource(reasoning_source)
        self.reasoning_effort = reasoning_effort
        self.use_stateful_protocol = use_stateful_protocol
        self.parallel_tool_calls = parallel_tool_calls
        self.tools = []
        self._tool_names = set()
        self.conversation = []

    @property
    def tool_names(self) -> set[str]:
        """Collection of available tool names"""
        return self._tool_names

    def add_tools(self, tools: list[ToolDef]) -> None:
        tools_map = {x.name: x for x in tools}
        seen = set()
        for i in range(len(self.tools)):
            name = self.tools[i]["name"]
            if name in tools_map:
                self.tools[i] = _encode_tool(tools_map[name])
                self._tool_names.add(name)
                seen.add(name)
        for name in tools_map:
            if name not in seen:
                self.tools.append(_encode_tool(tools_map[name]))
                self._tool_names.add(name)

    def reset_tools(self) -> None:
        self.tools.clear()
        self._tool_names.clear()

    def add_messages(self, messages: list[Message]) -> None:
        for msg in messages:
            for msg_enc in self._encode_message(msg):
                self.conversation.append(
                    {
                        "response_id": None,
                        "message": msg_enc,
                    }
                )

    def reset_messages(self) -> None:
        self.conversation.clear()

    def get_internal_conversation(self) -> Any:
        return self.conversation

    async def get_completion(
        self,
        conversation: None | list[dict] = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[Message]:
        if response_schema is not None and conversation is None:
            raise ValueError(
                "Specifying response_schema requires passing an explicit conversation"
            )
        messages = await self._get_completion_messages(
            conversation=conversation,
            update_conversation=update_conversation,
            response_schema=response_schema,
        )
        return self._decode_message_list(messages)

    def get_completion_sync(
        self,
        conversation: None | list[dict] = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[Message]:
        coro = self.get_completion(
            conversation=conversation,
            update_conversation=update_conversation,
            response_schema=response_schema,
        )
        return syncify(coro)

    def _get_conversation_messages(
        self, stateful: bool
    ) -> tuple[str | None, list[dict]]:
        if stateful:
            for i in range(len(self.conversation) - 1, -1, -1):
                response_id = self.conversation[i].get("response_id")
                if response_id:
                    return response_id, [
                        msg["message"] for msg in self.conversation[i + 1 :]
                    ]
        return None, [msg["message"] for msg in self.conversation]

    async def _responses(
        self,
        messages: list[dict],
        reasoning_effort_hint: Literal["low", "medium", "high", None],
        **kwargs,
    ):
        payload = {
            "model": self.deployment,
            "input": messages,
            **kwargs,
        }
        if self.is_reasoning:
            reasoning_params = {}
            if reasoning_effort_hint is not None:
                reasoning_params["effort"] = reasoning_effort_hint
            if self.reasoning_source is ReasoningSource.SUMMARY:
                reasoning_params["summary"] = "detailed"
            if reasoning_params:
                payload["reasoning"] = reasoning_params

        payload["store"] = True
        async with self.get_client() as client:
            response = await client.post(
                self.responses_url,
                json=payload,
            )
            raise_for_status_with_error(response)
            return response.json()

    async def _get_completion_messages(
        self,
        conversation: None | list[dict] = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[dict]:
        kwargs = {}
        messages = []
        if conversation is None:
            # when stateful is enabled, assuming conversation is append only,
            # pass the last previous_response_id instead of actual messages, to
            # continue the stored conversation
            previous_response_id, messages = self._get_conversation_messages(
                stateful=self.use_stateful_protocol,
            )
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id
        else:
            encoded_conversation = []
            for msg in conversation:
                encoded_conversation.extend(self._encode_message(msg))
            messages = encoded_conversation
            update_conversation = False

        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["parallel_tool_calls"] = self.parallel_tool_calls

        if response_schema is not None:
            if response_schema.get("additionalProperties") is not False:
                raise ValueError(
                    "response_schema must have 'additionalProperties' set to false"
                )
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response_schema",
                    "strict": True,
                    "schema": response_schema,
                }
            }

        out = await self._responses(
            messages=messages,
            reasoning_effort_hint=self.reasoning_effort,
            temperature=self.temperature,
            max_output_tokens=self.max_completion_tokens,
            **kwargs,
        )

        output_messages = out["output"]
        response_id = out["id"]

        self.last_usage = Usage(**out["usage"])

        if update_conversation:
            assert conversation is None
            for msg in output_messages:
                _clean_up_output_message_inplace(msg)
                self.conversation.append(
                    {
                        "response_id": response_id,
                        "message": msg,
                    }
                )

        return output_messages

    def _encode_message(self, msg: Message) -> list[dict[str, Any]]:
        # encode Message -> list of AOAI messages
        if isinstance(msg, Text):
            role = msg.role
            if msg.tag == "think":
                # discard reasoning messages from context
                # TODO or better fail?
                return []
            if self.is_reasoning and role == "system":
                role = "developer"
            return [{"role": role, "content": msg.content}]
        elif isinstance(msg, ParallelToolCall):
            out = []
            for tc in msg.tool_calls:
                tc_dict = {
                    "arguments": json.dumps(tc.arguments),
                    "call_id": tc.id,
                    "name": tc.name,
                    "type": "function_call",
                }
                out.append(tc_dict)
            return out
        elif isinstance(msg, ToolCall):
            return self._encode_message(ParallelToolCall(tool_calls=[msg]))
        elif isinstance(msg, ToolResponse):
            return [
                {
                    "call_id": msg.id,
                    "output": msg.content,
                    "type": "function_call_output",
                }
            ]
        raise TypeError(type(msg))

    def _decode_message_list(self, messages: list[dict]) -> list[Message]:
        decoded_messages: list[Message] = []

        for msg in messages:
            msg_type = msg.get("type")
            msg_role = msg.get("role")
            if msg_type == "reasoning":
                out = Text(
                    role="assistant",
                    content=get_message_reasoning_text(
                        msg, source=self.reasoning_source
                    ),
                    metadata={
                        "tag": "think",
                    },
                )
                decoded_messages.append(out)
            elif msg_type == "function_call":
                if decoded_messages and isinstance(
                    decoded_messages[-1], ParallelToolCall
                ):
                    out = decoded_messages[-1]
                else:
                    out = ParallelToolCall(tool_calls=[])
                    decoded_messages.append(out)
                out.tool_calls.append(_convert_function_call(msg))
            elif msg_type == "message" and msg_role in (
                "system",
                "developer",
                "user",
                "assistant",
            ):
                out = Text(
                    role=msg_role,
                    content=get_message_content_text(msg),
                )
                if self.is_reasoning:
                    out.metadata["tag"] = "text"
                decoded_messages.append(out)
            else:
                raise NotImplementedError(
                    f"cannot convert type={msg_type} role={msg_role}"
                )
        return decoded_messages

    @staticmethod
    def from_config(config: AOAIResponsesSessionConfig):
        credential = None
        if config.credential is not None:
            credential = create_credential(config.credential)
        return AOAIResponsesSession(
            deployment=config.deployment,
            credential=credential,
            client_certificate=config.client_certificate,
            trust_ca_path=config.trust_ca_path,
            headers=config.headers,
            timeout=config.timeout,
            use_dns_cache=config.use_dns_cache,
            max_retries_server_error=config.max_retries_server_error,
            retryable_server_errors=config.retryable_server_errors,
            account_name=config.account_name,
            endpoint_url=config.endpoint_url,
            seed=config.seed,
            temperature=config.temperature,
            max_completion_tokens=config.max_completion_tokens,
            is_reasoning=config.is_reasoning,
            reasoning_source=ReasoningSource(config.reasoning_source),
            reasoning_effort=config.reasoning_effort,
            use_stateful_protocol=config.use_stateful_protocol,
            api_version=config.api_version,
            parallel_tool_calls=config.parallel_tool_calls,
        )
