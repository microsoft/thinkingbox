# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import json
from typing import Any, Sequence

import httpx

from thinkingbox.common.chat_types import (
    Message,
    ParallelToolCall,
    Text,
    ToolCall,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import AOAISessionConfig
from thinkingbox.common.credential_factory import create_credential
from thinkingbox.common.llm_session_base import HTTPLLMSessionBase
from thinkingbox.common.usage_types import (
    InputTokensDetails,
    OutputTokensDetails,
    Usage,
)
from thinkingbox.common.utils import (
    CredentialBase,
    raise_for_status_with_error,
    syncify,
)


def _encode_tool(tool: ToolDef) -> dict:
    # encode ToolDef -> AOAI tools
    parameters_schema = copy.deepcopy(tool.input_schema)
    parameters_schema["additionalProperties"] = False
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema,
        },
    }


def _convert_tool_call(tc) -> ToolCall:
    error = None
    try:
        arguments = json.loads(tc["function"]["arguments"])
    except (ValueError, TypeError) as e:
        arguments = {}
        error = f"Error: {e!s}"
    if not isinstance(arguments, dict):
        error = "Error: tool arguments must be an object"
    return ToolCall(
        name=tc["function"]["name"],
        arguments=arguments,
        id=tc["id"],
        metadata={"error": error},
    )


def httpx_giveup_fn(e):
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code != 429
    if isinstance(e, httpx.TimeoutException):
        return False
    return True


NO_DISABLE_PARAMS = (
    "messages",
    "tools",
)


class AOAISession(HTTPLLMSessionBase):
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
        reasoning_effort: str | None = None,
        disabled_params: Sequence[str] = (),
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
            self.chat_completions_url = f"https://{account_name}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        else:
            self.chat_completions_url = endpoint_url
        self.deployment = deployment
        self.seed = seed
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.is_reasoning = is_reasoning
        self.reasoning_effort = reasoning_effort
        self.parallel_tool_calls = parallel_tool_calls
        self.disabled_params = list(disabled_params)
        for key in self.disabled_params:
            if key in NO_DISABLE_PARAMS:
                raise ValueError(f"Invalid element in disabled_params: {key!r}")
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
            name = self.tools[i]["function"]["name"]
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
            self.conversation.extend(self._encode_message(msg))

    def reset_messages(self) -> None:
        self.conversation.clear()

    def get_internal_conversation(self) -> Any:
        return self.conversation

    async def get_completion(
        self,
        conversation: list[Message] | None = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[Message]:
        if response_schema is not None and conversation is None:
            raise ValueError(
                "Specifying response_schema requires passing an explicit conversation"
            )
        message = await self._get_completion(
            conversation=conversation,
            update_conversation=update_conversation,
            response_schema=response_schema,
        )
        return self._decode_message(message)

    def get_completion_sync(
        self,
        conversation: list[Message] | None = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[Message]:
        coro = self.get_completion(
            conversation=conversation,
            update_conversation=update_conversation,
            response_schema=response_schema,
        )
        return syncify(coro)

    async def _get_completion(
        self,
        conversation: list[Message] | None = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> dict:
        payload = {}

        if not self.aoai_mode:
            payload["model"] = self.deployment

        if conversation is None:
            payload["messages"] = self.conversation
        else:
            encoded_conversation = []
            for msg in conversation:
                encoded_conversation.extend(self._encode_message(msg))
            payload["messages"] = encoded_conversation
            update_conversation = False

        payload["temperature"] = self.temperature
        payload["max_completion_tokens"] = self.max_completion_tokens
        payload["seed"] = self.seed

        if self.tools:
            payload["tools"] = self.tools
            payload["parallel_tool_calls"] = self.parallel_tool_calls
        if self.is_reasoning and self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if response_schema is not None:
            if response_schema.get("additionalProperties") is not False:
                raise ValueError(
                    "response_schema must have 'additionalProperties' set to false"
                )
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        for key in self.disabled_params:
            if key in payload:
                del payload[key]

        async with self.get_client() as client:
            response = await client.post(
                self.chat_completions_url,
                json=payload,
            )
            raise_for_status_with_error(response)
            out = response.json()

        msg = out["choices"][0]["message"]

        usage = out.get("usage") or {}
        prompt_tokens_details = usage.get("prompt_tokens_details") or {}
        completion_tokens_details = usage.get("completion_tokens_details") or {}
        self.last_usage = Usage(
            input_tokens=usage.get("prompt_tokens") or 0,
            output_tokens=usage.get("completion_tokens") or 0,
            total_tokens=usage.get("total_tokens"),
            input_tokens_details=InputTokensDetails(
                cached_tokens=prompt_tokens_details.get("cached_tokens") or 0,
            ),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=completion_tokens_details.get("reasoning_tokens") or 0,
            ),
        )

        if update_conversation:
            assert conversation is None
            self.conversation.append(msg)

        return msg

    def _encode_message(self, msg: Message) -> list[dict[str, Any]]:
        # encode Message -> list of AOAI messages
        if isinstance(msg, Text):
            role = msg.role
            return [{"role": role, "content": msg.content}]
        elif isinstance(msg, ParallelToolCall):
            tool_calls = []
            for tc in msg.tool_calls:
                tc_dict = {
                    "function": {
                        "arguments": json.dumps(tc.arguments),
                        "name": tc.name,
                    },
                    "id": tc.id,
                    "type": "function",
                }
                tool_calls.append(tc_dict)
            return [{"role": "assistant", "tool_calls": tool_calls}]
        elif isinstance(msg, ToolCall):
            return self._encode_message(ParallelToolCall(tool_calls=[msg]))
        elif isinstance(msg, ToolResponse):
            return [
                {
                    "role": "tool",
                    "tool_call_id": msg.id,
                    "content": msg.content,
                }
            ]
        raise TypeError(type(msg))

    def _decode_message(self, msg: dict) -> list[Message]:
        role = msg.get("role")
        if role in ("system", "developer", "user"):
            out = Text(
                role=role,
                content=msg.get("content", ""),
            )
            return [out]
        elif role == "assistant":
            out_list = []
            if self.is_reasoning:
                # check for deepseek-style reasoning and produce
                # a think message if it's there
                reasoning_content = msg.get("reasoning_content")
                if reasoning_content:
                    out = Text(
                        role=role, content=reasoning_content, metadata={"tag": "think"}
                    )
                    out_list.append(out)
            content = msg.get("content")
            if content:
                out = Text(
                    role=role,
                    content=content,
                )
                if self.is_reasoning:
                    # if it's a reasoning model, reasoning should be in reasoning_content
                    # and this must be the non-reasoning final response
                    out.metadata["tag"] = "text"
                out_list.append(out)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                out = ParallelToolCall(
                    tool_calls=[_convert_tool_call(tc) for tc in tool_calls],
                )
                out_list.append(out)
            return out_list
        raise NotImplementedError(f"cannot convert role={role}")

    @staticmethod
    def from_config(config: AOAISessionConfig):
        credential = None
        if config.credential is not None:
            credential = create_credential(config.credential)
        return AOAISession(
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
            reasoning_effort=config.reasoning_effort,
            api_version=config.api_version,
            disabled_params=config.disabled_params,
            parallel_tool_calls=config.parallel_tool_calls,
        )
