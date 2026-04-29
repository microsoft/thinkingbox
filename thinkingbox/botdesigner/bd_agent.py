# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from thinkingbox.botdesigner.activities import (
    ActivityParser,
    message_to_activity,
)
from thinkingbox.botdesigner.bd_client import (
    BotDesignerClient,
    ConversationContext,
    StepResult,
)
from thinkingbox.botdesigner.bot_override import BotOverrideTemplate
from thinkingbox.botdesigner.utils import YAMLRenderer
from thinkingbox.common.agent_session_base import AgentSessionBase
from thinkingbox.common.chat_types import (
    Message,
    TestContext,
    Text,
    ToolCall,
    ToolCallResponse,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import (
    AgentConfig,
    BotDesignerRecognizerKind,
    BotDesignerToolTranslationMode,
)
from thinkingbox.common.mcp_proxy_client import MCPProxyContext

logger = logging.getLogger(__name__)


class BotDesignerActivityError(Exception):
    pass


def _extract_tool_calls_from_messages(messages: list) -> list[ToolCallResponse]:
    """Extract ToolCallResponse pairs from a messages list.

    The messages list contains ToolCall and ToolResponse objects.
    Pair them by matching IDs.
    """
    tool_calls = []
    pending: dict[str, ToolCall] = {}

    for msg in messages:
        if isinstance(msg, ToolCall):
            pending[msg.id] = msg
        elif isinstance(msg, ToolResponse):
            if msg.id in pending:
                tc = pending.pop(msg.id)
                tool_calls.append(ToolCallResponse(tool_call=tc, tool_response=msg))

    return tool_calls


def _invert_str_map(d: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, value in d.items():
        if value not in out:
            out[value] = [key]
        else:
            out[value].append(key)
    return out


class BotDesignerScenarioMetadata(BaseModel):
    bd_bot_template: dict[str, Any] | None = None
    bd_bot_variables: dict[str, Any] = Field(default_factory=dict)

    # The options below are not supported in BotDesigner yet, do not set them
    bd_connector_tools: list[str] = Field(default_factory=list)
    bd_prebuilt_tool_overrides: dict[str, str] = Field(default_factory=dict)


class BotDesignerAgentSession(AgentSessionBase):
    def __init__(
        self,
        config: AgentConfig,
        mcp_proxy: MCPProxyContext,
        mcp_tools: list[ToolDef],
        bot_instructions: str | None,
        scenario_metadata: dict[str, Any],
        botdesigner_client: BotDesignerClient,
        bot_template: dict[str, Any],
        bot_variables: dict[str, Any],
        locale: str = "en-US",
        connector_endpoint_override: str | None = None,
        tool_translation_mode: BotDesignerToolTranslationMode = "connector",
        recognizer_kind: BotDesignerRecognizerKind = "GenerativeAIRecognizer",
    ):
        super().__init__(
            config=config,
            mcp_proxy=mcp_proxy,
            mcp_tools=mcp_tools,
            bot_instructions=bot_instructions,
            scenario_metadata=scenario_metadata,
        )
        self.bd_client = botdesigner_client
        self.locale = locale

        self.bd_metadata = BotDesignerScenarioMetadata(**self.scenario_metadata)
        if self.bd_metadata.bd_bot_template is not None:
            bot_template = self.bd_metadata.bd_bot_template

        self.bot_template = BotOverrideTemplate(
            bot_template,
            variables={
                **bot_variables,
                **self.bd_metadata.bd_bot_variables,
            },
        )
        self.connector_endpoint_override = connector_endpoint_override
        self.tool_translation_mode = tool_translation_mode
        self.recognizer_kind = recognizer_kind
        self.bd_conversation: ConversationContext | None = None
        self.activity_parser = ActivityParser()

        # This is a placeholder for BD's system message, that is not provided by us,
        # and is not returned to us from the API
        msg_system = Text(
            role="system",
            content="(BotDesigner System)",
        )
        self.conversation.messages.append(msg_system)
        if self.bot_instructions:
            msg_bot = Text(
                role="system",
                content=self.bot_instructions,
            )
            self.conversation.messages.append(msg_bot)
        self.pending_messages: list[Message] = []
        self.raw_events: list[dict] = []

    def add_messages(self, messages: list[Message], add_to_llm: bool = True):
        if add_to_llm:
            self.pending_messages.extend(messages)
        self.conversation.messages.extend(messages)

    def _add_bd_messages(self, result: StepResult) -> list[Message]:
        for activity in result.activities:
            self.raw_events.append(
                {
                    "source": "activities",
                    "data": activity,
                }
            )
        parsed = self.activity_parser.parse_activities(result.activities)
        messages = parsed.messages.copy()
        errors: list[str] = []
        for msg in messages:
            is_user_msg = isinstance(msg, Text) and msg.role == "user"
            if not is_user_msg:
                self.add_messages([msg], add_to_llm=False)

        if parsed.plan_finished:
            if (
                len(messages) > 0
                and isinstance(messages[-1], Text)
                and messages[-1].role == "assistant"
            ):
                messages[-1].metadata["is_done"] = True
            else:
                done_msg = Text(
                    role="assistant", content="<DONE>", metadata={"is_done": True}
                )
                self.add_messages([done_msg], add_to_llm=False)
                messages.append(done_msg)
        errors.extend(parsed.errors)
        if errors:
            raise BotDesignerActivityError("\n".join(str(err) for err in errors))
        return messages

    async def _start_conversation(self) -> AsyncIterator[Message]:
        assert self.bd_conversation is None

        connector_tools: list[ToolDef] = []
        mcp_tool_names: list[str] = []
        bd_connector_tools = set(self.bd_metadata.bd_connector_tools)
        bd_prebuilt_tool_overrides: dict[str, str] = {}
        bd_prebuilt_tool_overrides_inv = _invert_str_map(
            self.bd_metadata.bd_prebuilt_tool_overrides,
        )

        # Note: BotDesigner does not support overriding MCP or prebuilt tools yet.
        # Here we just create an interface for TB tests to define these overrides.

        for tool in self.tools:
            # if translation_mode is "connector"
            #   - all tools are translated to connectors
            # if translation_mode is "mcp"
            #   - the MCP connector is inserted, expecting list_tools to
            #       return all tool defs by default
            #   - tools in bd_metadata.bd_connector_tools are still translated to
            #       connectors, and list_tools should not return them
            # if translation_mode is "none"
            #   - tools are not translated, expecting that they are not needed
            #       or they are hardcoded into the bot template.
            #       prebuilt tool overrides are still forwarded.

            # if tool is mapped to override a prebuilt tool, do not
            # include a connector for it
            if tool.name in bd_prebuilt_tool_overrides_inv:
                for prebuilt_tool_name in bd_prebuilt_tool_overrides_inv[tool.name]:
                    bd_prebuilt_tool_overrides[prebuilt_tool_name] = tool.name
            else:
                if self.tool_translation_mode == "none":
                    continue
                bd_tool_type: BotDesignerToolTranslationMode = (
                    "connector"
                    if (
                        self.tool_translation_mode == "connector"
                        or tool.name in bd_connector_tools
                    )
                    else "mcp"
                )
                if bd_tool_type == "connector":
                    connector_tools.append(tool)
                else:
                    mcp_tool_names.append(tool.name)

        # store MCP tools filter in the session proxy.
        # If mcp_tool_names is empty, store an empty filter
        # this will result in an empty response from list_tools
        # note: this only affects the MCP endpoint for BotDesigner
        # (/connetors/{session_id}/mcp_b354464ac7e246a7aa29fe3304a9a3ba)
        await self.mcp.client.session_info(
            update={
                "bd_mcp_tool_names": mcp_tool_names,
                "bd_prebuilt_tool_overrides": bd_prebuilt_tool_overrides,
            },
        )

        bot_override = self.bot_template.render(
            bot_instructions=self.bot_instructions,
            connector_tools=connector_tools,
            add_mcp_connector=(len(mcp_tool_names) > 0),
            recognizer_kind=self.recognizer_kind,
        )
        # The ending "/" is important or BotDesigner will ignore the entire last part of the path
        connector_endpoint_override = (
            self.connector_endpoint_override.rstrip("/") + f"/{self.mcp.session_id}/"
            if self.connector_endpoint_override is not None
            else None
        )
        self.raw_events.append(
            {
                "source": "bd_agent",
                "data": {
                    "bot_override": YAMLRenderer().render(bot_override),
                    "connector_endpoint_override": connector_endpoint_override,
                },
            }
        )

        with self.timers.measure("time_agent"):
            bd_conversation = await self.bd_client.conversation_start(
                emit_start_conversation_event=True,
                locale=self.locale,
                test_mode="Text",
                bot_definition_override=bot_override,
                connector_endpoint_override=connector_endpoint_override,
            )

        self.raw_events.append(
            {
                "source": "bd_agent",
                "data": {
                    "conversation_id": bd_conversation.conversation_id,
                },
            }
        )

        self.bd_conversation = bd_conversation
        self.activity_parser.set_conversation_id(self.bd_conversation.conversation_id)
        new_messages = self._add_bd_messages(bd_conversation.result)
        for msg in new_messages:
            yield msg

    async def decode_turn_iter(
        self, user_message: Text | None
    ) -> AsyncIterator[Message]:

        if self.bd_conversation is None:
            async for msg in self._start_conversation():
                yield msg

        assert self.bd_conversation is not None

        if user_message is not None:
            assert user_message.role == "user"
            self.add_messages([user_message.model_copy(deep=True)])

        result: StepResult | None = None

        while self.pending_messages:
            # TODO can we send multiple messages or should we limit to 1?
            # In practice it should always be 1
            activity = message_to_activity(self.pending_messages.pop(0))
            result = await self.bd_client.conversation_send_message(
                self.bd_conversation, activity=activity
            )
            new_messages = self._add_bd_messages(result)
            for msg in new_messages:
                yield msg

        if (result is not None) and (not result.should_continue()):
            return

        with self.timers.measure("time_agent"):
            async for result in self.bd_client.conversation_continue_iter(
                self.bd_conversation
            ):
                new_messages = self._add_bd_messages(result)
                for msg in new_messages:
                    yield msg

    def can_end_conversation(self) -> bool:
        return self.recognizer_kind != "CLIAgentRecognizer"

    async def make_test_context(self) -> TestContext:
        ctx = await super().make_test_context()

        # MCS doesn't give us the tool calls reliably, we retrieve them
        # from the session proxy
        ctx.tool_calls.clear()
        proxy_info = ctx.effects.pop("__reserved__proxy_info", None)
        if proxy_info is not None:
            for tcr_dict in proxy_info["tool_calls"]:
                tcr = ToolCallResponse(
                    tool_call=ToolCall(
                        name=tcr_dict["tool_name"],
                        arguments=tcr_dict["arguments"],
                    ),
                    tool_response=ToolResponse(
                        name=tcr_dict["tool_name"],
                        content=tcr_dict["response"],
                    ),
                )
                ctx.tool_calls.append(tcr)
        else:
            # session proxy is not configured to return tool calls / responses
            # best effort attempt to extract tool calls from conversation,
            # but these will have different names and modified or incomplete
            # params and results
            # TODO should we just fail instead?
            ctx.tool_calls.extend(
                _extract_tool_calls_from_messages(self.conversation.messages),
            )
        return ctx

    def get_raw_messages(self) -> list | None:
        return self.raw_events.copy()

    @staticmethod
    def from_config(
        config: AgentConfig,
        mcp_proxy: MCPProxyContext,
        mcp_tools: list[ToolDef],
        bot_instructions: str | None,
        scenario_metadata: dict[str, Any],
        botdesigner_client: BotDesignerClient,
        bot_template: dict[str, Any],
        bot_variables: dict[str, Any],
        locale: str = "en-US",
        connector_endpoint_override: str | None = None,
        tool_translation_mode: BotDesignerToolTranslationMode = "connector",
        recognizer_kind: BotDesignerRecognizerKind = "GenerativeAIRecognizer",
    ) -> BotDesignerAgentSession:
        return BotDesignerAgentSession(
            config=config,
            mcp_proxy=mcp_proxy,
            mcp_tools=mcp_tools,
            bot_instructions=bot_instructions,
            scenario_metadata=scenario_metadata,
            botdesigner_client=botdesigner_client,
            bot_template=bot_template,
            bot_variables=bot_variables,
            locale=locale,
            connector_endpoint_override=connector_endpoint_override,
            tool_translation_mode=tool_translation_mode,
            recognizer_kind=recognizer_kind,
        )
