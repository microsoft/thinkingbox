# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import re
from typing import Any

from thinkingbox.common.chat_types import (
    Message,
    ParallelToolCall,
    Text,
    ToolCall,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import AnthropicMessagesSessionConfig
from thinkingbox.common.credential_factory import create_credential
from thinkingbox.common.llm_session_base import HTTPLLMSessionBase
from thinkingbox.common.usage_types import InputTokensDetails, Usage
from thinkingbox.common.utils import (
    CredentialBase,
    raise_for_status_with_error,
    syncify,
)

_VALID_PROPERTY_KEY_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


class AnthropicMessagesSession(HTTPLLMSessionBase):
    def __init__(
        self,
        deployment: str,
        credential: CredentialBase | None = None,
        client_certificate: str | None = None,
        trust_ca_path: str | None = None,
        account_name: str | None = None,
        endpoint_url: str | None = None,
        temperature: float = None,
        max_completion_tokens: int = 4096,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        thinking: dict | None = None,
        output_config: dict | None = None,
        timeout=60.0,
        headers: dict[str, str] | None = None,
        use_dns_cache: bool = False,
        max_retries_server_error: int = 5,
        retryable_server_errors: tuple[int | str, ...] = (502, 503, 504),
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
        self.deployment = deployment
        self.account_name = account_name
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.stop_sequences = stop_sequences
        self.thinking = copy.deepcopy(thinking)
        self.output_config = copy.deepcopy(output_config)
        self.parallel_tool_calls = parallel_tool_calls
        self.tools = []
        self._tool_names = set()
        self._tool_arg_mappings: dict[str, dict[str, str]] = {}
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
                self.tools[i] = self._encode_tool(tools_map[name])
                self._tool_names.add(name)
                seen.add(name)
        for name in tools_map:
            if name not in seen:
                self.tools.append(self._encode_tool(tools_map[name]))
                self._tool_names.add(name)

    def reset_tools(self) -> None:
        self.tools.clear()
        self._tool_names.clear()
        self._tool_arg_mappings.clear()

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
        return await self._get_completion(
            conversation=conversation,
            update_conversation=update_conversation,
            response_schema=response_schema,
        )

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
    ) -> list[Message]:
        payload, force_update_conversation_false = self._encode_payload(
            conversation,
            response_schema=response_schema,
        )
        if force_update_conversation_false:
            update_conversation = False
        async with self.get_client() as client:
            response = await client.post(
                self.endpoint_url,
                json=payload,
            )
            raise_for_status_with_error(response)
            response_body = response.json()

        msg = response_body
        decoded_messages = self._decode_message(msg)

        """
        Usage: does not contain reasoning token count

        "usage": {
            "input_tokens": 761,
            "output_tokens": 167,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            ...
        }
        """

        usage = msg.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("cache_read_input_tokens", 0)
        self.last_usage = Usage(
            input_tokens=input_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        if update_conversation:
            assert conversation is None
            encoded_messages = [
                x for m in decoded_messages for x in self._encode_message(m)
            ]
            self.conversation.extend(encoded_messages)

        return decoded_messages

    def _encode_payload(
        self,
        conversation: list[Message] | None = None,
        response_schema: dict | None = None,
    ) -> tuple[dict, bool]:
        payload = {}
        update_conversation = False
        payload["model"] = self.deployment
        payload["cache_control"] = {"type": "ephemeral"}
        if conversation is None:
            payload["messages"] = self.conversation
        else:
            encoded_conversation = []
            for msg in conversation:
                encoded_msg = self._encode_message(msg)
                encoded_conversation.extend(encoded_msg)
            payload["messages"] = encoded_conversation
            update_conversation = True

        if self.temperature is not None:
            payload["temperature"] = self.temperature

        payload["max_tokens"] = self.max_completion_tokens

        if self.top_k is not None:
            payload["top_k"] = self.top_k

        if self.top_p is not None:
            payload["top_p"] = self.top_p

        if self.stop_sequences is not None and len(self.stop_sequences) > 0:
            payload["stop_sequences"] = self.stop_sequences

        if self.tools:
            payload["tools"] = self.tools
            if not self.parallel_tool_calls:
                payload["tool_choice"] = {
                    "type": "auto",
                    "disable_parallel_tool_use": True,
                }

        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.output_config is not None:
            payload["output_config"] = self.output_config
        if response_schema is not None:
            if response_schema.get("additionalProperties") is not False:
                raise ValueError(
                    "response_schema must have 'additionalProperties' set to false"
                )
            output_config = copy.deepcopy(payload.get("output_config", {}))
            output_config["format"] = {
                "type": "json_schema",
                "schema": response_schema,
            }
            payload["output_config"] = output_config

        # Merge sequential messages with the same role into one message.
        # This is required because:
        # - Text(tag=think) and ParallelToolCall encode as separate
        #   assistant messages that must be combined
        # - Multiple ToolResponse messages encode as separate user messages
        #   that must be combined (Anthropic requires alternating roles and
        #   all tool_results in a single user message)
        new_messages = []
        for msg in payload["messages"]:
            if new_messages and msg["role"] == new_messages[-1]["role"]:
                new_messages[-1]["content"].extend(copy.deepcopy(msg["content"]))
            else:
                new_messages.append(copy.deepcopy(msg))

        # Extract system messages from the beginning and set the system field.
        # System messages after non-system messages are not allowed.
        system_texts = []
        while new_messages and new_messages[0]["role"] == "system":
            for block in new_messages.pop(0)["content"]:
                system_texts.append(block["text"])
        # Check for system messages after non-system messages
        for msg in new_messages:
            if msg["role"] == "system":
                raise ValueError(
                    "System prompt/s can only be set at the beginning of the conversation"
                )
        if system_texts:
            payload["system"] = [
                {
                    "type": "text",
                    "text": "\n".join(system_texts),
                },
            ]

        payload["messages"] = new_messages

        return payload, update_conversation

    def _encode_message(self, msg: Message) -> list[dict]:
        """
        Encode Message object to Anthropic API format.

        Anthropic format:
        {
            "role": "user|assistant",
            "content": [
                {
                    "type": "text",
                    "text": "This is a test message",
                }
            ],
        }
        """
        if isinstance(msg, Text):
            # Skip dummy messages inserted for thinking-only responses
            if msg.metadata.get("is_dummy"):
                return []
            if msg.role == "system":
                # system messages are converted to a fictional "system" role.
                # these messages are concatenated and passed to the endpoint in the "system" field
                # system messages can only appear at the beginning
                return [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": msg.content,
                            }
                        ],
                    }
                ]
            elif msg.tag == "think":
                redacted_data = msg.metadata.get("redacted_thinking")
                if redacted_data is not None:
                    # Re-encode redacted thinking block
                    return [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "redacted_thinking",
                                    "data": redacted_data,
                                }
                            ],
                        }
                    ]
                # Re-encode thinking block with preserved signature
                thinking_block = {
                    "type": "thinking",
                    "thinking": msg.content,
                }
                signature = msg.metadata.get("thinking_signature")
                if signature:
                    thinking_block["signature"] = signature
                return [
                    {
                        "role": "assistant",
                        "content": [thinking_block],
                    }
                ]
            else:
                return [
                    {
                        "role": msg.role,
                        "content": [
                            {
                                "type": "text",
                                "text": msg.content,
                            }
                        ],
                    }
                ]

        elif isinstance(msg, ParallelToolCall):
            content = []
            for tc in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )

            return [
                {
                    "role": "assistant",
                    "content": content,
                }
            ]

        elif isinstance(msg, ToolCall):
            # Convert single tool call to parallel format
            return self._encode_message(ParallelToolCall(tool_calls=[msg]))

        elif isinstance(msg, ToolResponse):
            # Anthropic format for tool responses
            # Tool responses are expected to be under user role
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.id,
                            "content": [
                                {
                                    "type": "text",
                                    "text": msg.content,
                                }
                            ],
                        }
                    ],
                }
            ]

        raise TypeError(f"Unsupported message type: {type(msg)}")

    def _decode_message(self, msg: dict) -> list[Message]:
        """
        Decode Anthropic API response format back to Message objects.

        Anthropic response can contain multiple content blocks:
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "call_123", "name": "tool_name", "input": {...}}
            ]
        }
        """
        role = msg.get("role")
        content = msg.get("content", [])

        if not isinstance(content, list):
            # Handle case where content is a string (shouldn't happen in normal Anthropic responses)
            if role in ("user", "assistant"):
                return [Text(role=role, content=str(content))]
            else:
                raise ValueError(f"Unexpected role: {role}")

        messages = []
        tool_calls = []

        # Process each content block
        for content_block in content:
            content_type = content_block.get("type")

            if content_type == "thinking":
                thinking_content = content_block.get("thinking", "")
                if thinking_content:
                    text_msg = Text(
                        role=role,
                        content=thinking_content,
                        metadata={
                            "tag": "think",
                            "thinking_signature": content_block.get("signature"),
                        },
                    )
                    messages.append(text_msg)

            elif content_type == "redacted_thinking":
                # Preserve redacted thinking blocks for round-tripping
                text_msg = Text(
                    role=role,
                    content="",
                    metadata={
                        "tag": "think",
                        "redacted_thinking": content_block.get("data"),
                    },
                )
                messages.append(text_msg)

            elif content_type == "text":
                text_content = content_block.get("text", "")
                if text_content:
                    text_msg = Text(role=role, content=text_content)
                    messages.append(text_msg)

            elif content_type == "tool_use":
                content_block = self._reverse_sanitize_tool_use_content(content_block)
                tool_call = ToolCall(
                    name=content_block.get("name", ""),
                    arguments=content_block.get("input", {}),
                    id=content_block.get("id", ""),
                )
                tool_calls.append(tool_call)

            else:
                raise ValueError(f"Unknown content type: {content_type}")

        if tool_calls:
            messages.append(ParallelToolCall(tool_calls=tool_calls))

        # If the response only contains thinking blocks (no visible text, no
        # tool calls), insert a dummy empty text message.
        # This means that we should return control to the agent/user loop to make a decision
        # on whether to stop or involve user, but the relevant text message is the last non-thinking
        # non-tool message (the last message the user can see)
        if (
            not tool_calls
            and messages
            and all(isinstance(m, Text) and m.tag == "think" for m in messages)
        ):
            messages.append(Text(role=role, content="", metadata={"is_dummy": True}))

        return messages

    def _encode_tool(self, tool: ToolDef) -> dict:
        # encode ToolDef -> Anthropic messages tools
        parameters_schema = copy.deepcopy(tool.input_schema)
        mapping: dict[str, str] = {}
        parameters_schema = self._sanitize_tool_schema(parameters_schema, mapping)
        if mapping:
            self._tool_arg_mappings[tool.name] = mapping
        return {
            "type": "custom",
            "name": tool.name,
            "description": tool.description,
            "input_schema": parameters_schema,
        }

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Replace characters not matching [a-zA-Z0-9_.-] with _xHH encoding."""
        if _VALID_PROPERTY_KEY_RE.match(key):
            return key
        return re.sub(r"[^a-zA-Z0-9_.-]", lambda m: f"_x{ord(m.group()):02x}", key)[:64]

    @classmethod
    def _sanitize_tool_schema(cls, obj, mapping: dict[str, str]):
        """Sanitize property names in an object-typed JSON schema.

        If the schema has "type": "object" and "properties", sanitizes the
        property keys and corresponding "required" entries.
        Populates mapping with {sanitized: original} for any key that changed.

        Note: this is only necessary on the first level, the API is more flexible
        for nested objects.
        """
        if not isinstance(obj, dict):
            return obj
        if obj.get("type") != "object" or "properties" not in obj:
            return obj
        result = dict(obj)
        new_properties = {}
        for key, value in obj["properties"].items():
            sanitized_key = cls._sanitize_key(key)
            if sanitized_key in new_properties:
                raise ValueError(
                    f"Tool schema key collision: {key!r} and "
                    f"{mapping.get(sanitized_key, sanitized_key)!r} both "
                    f"sanitize to {sanitized_key!r}"
                )
            if sanitized_key != key:
                mapping[sanitized_key] = key
            new_properties[sanitized_key] = value
        result["properties"] = new_properties
        if "required" in obj and isinstance(obj["required"], list):
            result["required"] = [
                cls._sanitize_key(s) if isinstance(s, str) else s
                for s in obj["required"]
            ]
        return result

    def _reverse_sanitize_tool_use_content(self, content_block: dict) -> dict[str, Any]:
        """Reverse-map sanitized tool call keys back to originals."""
        tool_name = content_block.get("name", "")
        mapping = self._tool_arg_mappings.get(tool_name, {})
        if not mapping:
            return content_block
        arguments = content_block.get("input", {})
        new_content = content_block.copy()
        if isinstance(arguments, dict):
            new_content["input"] = {mapping.get(k, k): v for k, v in arguments.items()}
        return new_content

    @staticmethod
    def from_config(config: AnthropicMessagesSessionConfig):
        credential = None
        if config.credential is not None:
            credential = create_credential(config.credential)
        return AnthropicMessagesSession(
            deployment=config.deployment,
            credential=credential,
            client_certificate=config.client_certificate,
            trust_ca_path=config.trust_ca_path,
            endpoint_url=config.endpoint_url,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            stop_sequences=config.stop_sequences,
            thinking=config.thinking,
            output_config=config.output_config,
            max_completion_tokens=config.max_completion_tokens,
            timeout=config.timeout,
            headers=config.headers,
            use_dns_cache=config.use_dns_cache,
            max_retries_server_error=config.max_retries_server_error,
            retryable_server_errors=config.retryable_server_errors,
            parallel_tool_calls=config.parallel_tool_calls,
        )
