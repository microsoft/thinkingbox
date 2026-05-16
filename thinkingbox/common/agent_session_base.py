# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from thinkingbox.common.chat_types import (
    Conversation,
    FinishReason,
    Message,
    ParallelToolCall,
    TestContext,
    Text,
    ToolDef,
)
from thinkingbox.common.config_types import AgentConfig
from thinkingbox.common.mcp_proxy_client import MCPProxyContext
from thinkingbox.common.utils import Timers


def get_last_assistant_text_message_content(messages: list[Message]) -> str:
    for msg in messages[::-1]:
        if (
            isinstance(msg, Text)
            and msg.role == "assistant"
            and (msg.tag == "text")
            and not msg.metadata.get("is_dummy", False)
        ):
            return msg.content
    return ""


def should_end_conversation(messages: list[Message]) -> FinishReason | None:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        # first non-dummy message -> stop
        if (
            isinstance(msg, Text)
            and msg.role == "assistant"
            and msg.is_visible
            and not msg.metadata.get("is_dummy", False)
        ):
            is_done = msg.metadata.get("is_done", False)
            if is_done:
                # agent decided that the task has been completed
                return "done"
            else:
                # there is an assistant message which requires answering
                return None

        if isinstance(msg, ParallelToolCall):
            is_end_turn = msg.metadata.get("is_end_turn_tool", False)
            if is_end_turn:
                # a tool marked as "end turn" was called last
                return "end_turn_tool"

    return None


class AgentSessionBase(ABC):
    def __init__(
        self,
        config: AgentConfig,
        mcp_proxy: MCPProxyContext,
        mcp_tools: list[ToolDef],
        bot_instructions: str | None,
        scenario_metadata: dict[str, Any],
    ):
        self.timers = Timers()
        self.mcp = mcp_proxy
        self.tools = [tool.model_copy(deep=True) for tool in mcp_tools]
        self.config = config
        self.bot_instructions = bot_instructions
        self.scenario_metadata = scenario_metadata
        # Subclasses must maintain self.conversation with:
        #   - .messages: list of Message objects (used by should_end_conversation, make_test_context)
        #   - .metadata["tool_calls"]: list of ToolCallResponse (used by make_test_context)
        self.conversation = Conversation()

    @abstractmethod
    def add_messages(self, messages: list[Message], add_to_llm: bool = True):
        """Append messages to the conversation.
        If add_to_llm is True, also feed the new messages to the orchestrator / agent.
        Otherwise just append them to self.conversation.messages.
        """
        ...

    @abstractmethod
    async def decode_turn_iter(
        self, user_message: Text | None
    ) -> AsyncIterator[Message]:
        """Run the agent loop: send a user message (or None to continue), yield messages
        until the agent stops to wait for user input or an end-turn condition is reached.
        """
        ...

    @abstractmethod
    def can_end_conversation(self) -> bool:
        """Return whether the session is able to determine when the conversation should end
        (e.g. all user intents have been satisfied -> ready for grading).
        If agent can end the conversation, it must implement should_end_conversation() accordingly.
        Otherwise, the user/agent loop will require simulated user to end the conversation.
        """
        ...

    def should_end_conversation(self) -> FinishReason | None:
        """Check the tail of self.conversation.messages and return a FinishReason if the
        conversation should end ("done" or "end_turn_tool"), or None if it should continue.

        The default implementation looks at the messages metadata to determine whether
        the conversation should end due to agent signaling DONE (message.metadata["is_done"])
        or due to an end turn tool being called (message.metadata["is_end_turn_tool"]).
        If different logic is needed, the subclass should override this function and implement
        alternative logic.
        If agent cannot determine when the conversation should end, this should always return None.
        """
        return should_end_conversation(self.conversation.messages)

    async def get_effects(self) -> dict[str, Any]:
        """Retrieve side-effects recorded by the MCP server during the session."""
        with self.timers.measure("time_tool"):
            return await self.mcp.get_effects()

    async def make_test_context(self) -> TestContext:
        """Build a TestContext snapshot from the current conversation and world state,
        used for grading.

        If tool_calls exists in self.conversation.metadata, this function will retrieve
        the list of ToolCallResponse from there.
        Otherwise, the subclass should override this function and ensure
        TestContext.tool_calls is populated.
        """
        effects = await self.get_effects()
        response_text = ""
        tool_direct_responses = []

        if self.conversation.messages:
            # get last message content if it was a text message
            response_text = get_last_assistant_text_message_content(
                self.conversation.messages
            )

            # get direct tool responses
            for msg in self.conversation.messages:
                if isinstance(msg, Text) and (msg.tag == "direct"):
                    tool_direct_responses.append(msg.content)

        tool_calls = self.conversation.metadata.get("tool_calls", [])
        return TestContext(
            response=response_text,
            tool_calls=tool_calls,
            tool_direct_responses=tool_direct_responses,
            effects=effects,
            messages=self.conversation.messages,
            session_id=self.mcp.session_id,
            init_result=self.mcp.init_result,
        )

    def pop_timings(self) -> dict[str, float]:
        """Return accumulated timing measurements and reset the timers."""
        return self.timers.pop()

    def get_raw_messages(self) -> list | None:
        """Return a list of raw conversation messages or events with their
        internal representation.
        """
        return None
