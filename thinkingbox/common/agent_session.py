# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from thinkingbox.common.agent_session_base import AgentSessionBase
from thinkingbox.common.chat_types import (
    Conversation,
    Message,
    MessageConfig,
    ParallelToolCall,
    Text,
    ToolCall,
    ToolCallResponse,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import AgentConfig
from thinkingbox.common.llm_session_base import LLMSessionBase
from thinkingbox.common.mcp_proxy_client import MCPProxyContext

MESSAGE_CONFIG = MessageConfig(
    tag_done="<DONE>",
)


def make_empty_tool_response(tc: ToolCall) -> ToolResponse:
    return ToolResponse(
        name=tc.name,
        content="",
        id=tc.id,
    )


def try_pop_direct_response(
    tc: ToolCall,
    response: str,
    direct_response: str,
    error_message: str,
) -> ToolResponse:
    try:
        obj: dict = json.loads(response)
        direct_response = direct_response.format(**obj)

        if not isinstance(direct_response, str):
            raise ValueError(f"expected {direct_response}: str")

        return ToolResponse(
            name=tc.name,
            content=response,
            id=tc.id,
            metadata={"direct_response": direct_response},
        )
    except (TypeError, ValueError, KeyError):
        return ToolResponse(
            name=tc.name,
            content=error_message,
            id=tc.id,
            metadata={"direct_response": None},
        )


class AgentSession(AgentSessionBase):
    def __init__(
        self,
        config: AgentConfig,
        mcp_proxy: MCPProxyContext,
        mcp_tools: list[ToolDef],
        bot_instructions: str | None,
        scenario_metadata: dict[str, Any],
        agent_model: LLMSessionBase,
        system_instructions: str,
        parallel_tool_execution: bool = False,
    ):
        super().__init__(
            config=config,
            mcp_proxy=mcp_proxy,
            mcp_tools=mcp_tools,
            bot_instructions=bot_instructions,
            scenario_metadata=scenario_metadata,
        )
        self.llm = agent_model
        self.parallel_tool_execution = parallel_tool_execution

        self.message_config = MESSAGE_CONFIG

        # get "direct response" and "end turn" tools, add direct response description
        self.tools_direct_response = {}
        self.end_turn_tools = set()
        tool_seen = set()
        for tool in self.tools:
            if tool.name in tool_seen:
                raise ValueError(f"Duplicate tool name {tool.name!r}")
            tool_seen.add(tool.name)
            if tool.direct_response is not None:
                self.tools_direct_response[tool.name] = tool.direct_response
            if tool.is_end_turn:
                self.end_turn_tools.add(tool.name)

        # create prefix conversation
        msg_system = Text(
            role="system",
            content=system_instructions,
        )
        if self.bot_instructions is not None and self.bot_instructions != "":
            msg_bot = Text(
                role="system",
                content=self.bot_instructions,
            )
            self.prefix_conversation = Conversation(messages=[msg_system, msg_bot])
        else:
            self.prefix_conversation = Conversation(messages=[msg_system])

        # add tools to session
        self.llm.add_tools(self.tools)

        self._reset_conversation()

    def _reset_conversation(self):
        self.llm.reset_messages()
        self.llm.add_messages(self.prefix_conversation.model_copy(deep=True).messages)
        self.conversation = self.prefix_conversation.model_copy(deep=True)
        self.conversation.metadata["tool_calls"] = []

    def add_messages(self, messages: list[Message], add_to_llm: bool = True):
        if add_to_llm:
            self.llm.add_messages(messages)
        self.conversation.messages.extend(messages)

    async def decode_turn_iter(
        self, user_message: Text | None
    ) -> AsyncIterator[Message]:
        """
        Decode until the next interaction with user, or one of the end tools is called
        """

        if user_message is not None:
            assert user_message.role == "user"
            self.add_messages([user_message.model_copy(deep=True)])

        """
        The interaction starts with a user request. The interaction could end either because
            more information is necessary to satisfy the request, or because the request has
            been satisfied or denied (impossibility or refusal)

        Some tools may respond directly, this is a client configuration, forwarding text from
            a specific key in a JSON response object. This direct response is emitted as an
            assistant message after the tool is executed, followed by an actual completion
            from the agent.

        When there is nothing to do, the agent emits <DONE> in a text response to the user. This
            indicates that further interaction with the user is not needed to satisfy the user
            request. Otherwise, a response is expected from the user and the conversation should
            continue with another call to this function, until <DONE>.

        * Example conversation

        user: please do X
        assistant: call X(...)
        tool: [X] response
        assistant: done, how else can I help? <DONE>

        * Example conversation with direct response

        user: please tell me X and Y
        assistant: call tell_X(...)
        tool: [tell_X] response
        assistant: X (direct response, injected)
        assistant: call get_Y(...)
        tool: [get_Y] response
        assistant: Also Y. how else can I help? <DONE>

        * Example conversation with user response

        user: send X by email
        assistant: recipient email address?
        user: example@example.com, please also tell me the weather in Seattle
        assistant: call send_email(...)
        tool: [send_email] response
        assistant: call get_weather(...)
        tool: [get_weather] response
        assistant: email sent and weather in Seattle is ... <DONE>
        """

        # direct responses to forward after tool execution
        # these are messages that will be yielded after the tool calls
        tr_fw_messages = []

        while True:
            # get new message
            with self.timers.measure("time_agent"):
                messages = await self.llm.get_completion()

            last_usage = self.llm.last_usage
            # if the LLM session has prompt usage data, add it to the conversation metadata
            if last_usage:
                self.conversation.metadata["usage"].append(last_usage)

            end_turn_tool_found = False

            for msg in messages:
                if isinstance(msg, Text):
                    # check if the assistant message is done
                    if msg.role == "assistant" and msg.tag == "text":
                        # check if the message is done
                        if self.message_config.is_done(msg.content):
                            msg.metadata["is_done"] = True

                elif isinstance(msg, ParallelToolCall):
                    # look for end turn tools, tag them, set a flag to stop after yielding
                    for tc in msg.tool_calls:
                        if tc.name in self.end_turn_tools:
                            end_turn_tool_found = True
                            msg.metadata["is_end_turn_tool"] = True
                            tc.metadata["is_end_turn_tool"] = True

                            # we still need to record a dummy response so that it can be tested
                            tcr = ToolCallResponse(
                                tool_call=tc,
                                tool_response=make_empty_tool_response(tc),
                            )
                            self.conversation.metadata["tool_calls"].append(tcr)

            if not messages:
                # no new messages, stop to prevent infinite loop
                break

            # Add each message to the conversation and yield it one at a
            # time so that if the caller breaks mid-iteration (e.g. due to
            # max_agent_sim_turns), un-yielded messages are not persisted.
            for msg in messages:
                self.add_messages([msg], add_to_llm=False)
                yield msg

            # stop conditions
            last_msg = self.conversation.messages[-1]
            if (
                isinstance(last_msg, Text)
                and last_msg.role == "assistant"
                and last_msg.tag == "text"
            ):
                # interact with the user
                break

            if end_turn_tool_found:
                break

            # look for tools and call them
            for msg in messages:
                if isinstance(msg, ParallelToolCall):

                    # call a list of tools in a message
                    tool_responses: list[ToolResponse] = []
                    with self.timers.measure("time_tool"):
                        if self.parallel_tool_execution:
                            # may cause concurrent execution on the server if supported
                            coros = []
                            for tc in msg.tool_calls:
                                coros.append(self._handle_tool(tc))
                            tool_responses.extend(await asyncio.gather(*coros))
                        else:
                            # call sequentially
                            for tc in msg.tool_calls:
                                result = await self._handle_tool(tc)
                                tool_responses.append(result)

                    # add tool responses to conversation
                    for tc, tr in zip(msg.tool_calls, tool_responses, strict=True):
                        # extract direct responses
                        direct_response: str | None = tr.metadata.pop(
                            "direct_response", None
                        )
                        tr_fw_messages = []

                        if direct_response is not None:
                            msg_fw_content = Text(
                                role="assistant",
                                content=direct_response,
                                metadata={"tag": "direct"},
                            )
                            tr_fw_messages.append(msg_fw_content)

                        # add call/response to metadata
                        tcr = ToolCallResponse(
                            tool_call=tc,
                            tool_response=tr,
                        )
                        self.conversation.metadata["tool_calls"].append(tcr)

                        self.add_messages([tr])
                        yield tr

                # forward tool responses
                for msg_fw in tr_fw_messages:
                    self.add_messages([msg_fw])
                    yield msg_fw
                tr_fw_messages.clear()

    def can_end_conversation(self) -> bool:
        return True

    async def _handle_tool(self, tc: ToolCall) -> ToolResponse:
        tc_error = tc.metadata.get("error")
        if tc_error:
            return ToolResponse(
                name=tc.name,
                content=tc_error,
                id=tc.id,
            )
        assert isinstance(tc.arguments, dict)
        if tc.name not in self.llm.tool_names:
            return ToolResponse(
                name=tc.name,
                content=f"Error: function '{tc.name}' does not exist",
                id=tc.id,
            )
        result = await self.mcp.call_tool(tc.name, **tc.arguments)

        direct_response = self.tools_direct_response.get(tc.name)

        if direct_response is not None:
            # Must be valid JSON, otherwise it's an error
            return try_pop_direct_response(
                tc,
                result,
                direct_response,
                error_message="Error in function execution",
            )

        return ToolResponse(
            name=tc.name,
            content=result,
            direct_response=None,
            id=tc.id,
        )

    def get_raw_messages(self) -> list | None:
        raw_messages = self.llm.get_internal_conversation()
        if isinstance(raw_messages, list):
            return raw_messages
        return None

    @staticmethod
    def from_config(
        config: AgentConfig,
        mcp_proxy: MCPProxyContext,
        mcp_tools: list[ToolDef],
        bot_instructions: str | None,
        scenario_metadata: dict[str, Any],
        agent_model: LLMSessionBase,
    ) -> AgentSession:
        all_tools = config.builtin_tools + mcp_tools
        return AgentSession(
            config=config,
            agent_model=agent_model,
            mcp_proxy=mcp_proxy,
            mcp_tools=all_tools,
            system_instructions=config.system_instructions,
            bot_instructions=bot_instructions,
            scenario_metadata=scenario_metadata,
            parallel_tool_execution=False,  # TODO only if all tools are safe...
        )
