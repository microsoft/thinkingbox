#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import traceback
from collections.abc import Callable

from thinkingbox.common.agent_session_base import AgentSessionBase
from thinkingbox.common.chat_types import (
    Conversation,
    DecodeResult,
    FinishReason,
    Message,
    MessageT,
    ParallelToolCall,
    Text,
    ToolCall,
)
from thinkingbox.common.config_types import (
    HydratedTestCase,
    format_string_or_none,
    merge_bot_instructions,
    update_tools_with_client_config,
)
from thinkingbox.common.llm_session_base import LLMSessionBase
from thinkingbox.common.mcp_proxy_client import MCPProxyContext
from thinkingbox.common.user_simulated_answer import UserSimulator
from thinkingbox.common.utils import ErrorInfo, Timers

logger = logging.getLogger(__name__)


async def replay_history_tool_calls(
    history: list[MessageT],
    mcp_proxy: MCPProxyContext,
    builtin_tool_names: set[str] | None = None,
) -> list[str]:
    """Re-execute ToolCall entries from history against the MCP server sequentially.

    Skips tool calls with metadata["error"] (originally failed) and builtin tool
    calls (handled by the framework, not the MCP server). Replay errors are logged
    as warnings and returned — they do not propagate.
    """
    builtin_tool_names = builtin_tool_names or set()
    warnings: list[str] = []
    for msg in history:
        tool_calls_to_replay: list[ToolCall] = []
        if isinstance(msg, ToolCall):
            tool_calls_to_replay = [msg]
        elif isinstance(msg, ParallelToolCall):
            tool_calls_to_replay = list(msg.tool_calls)

        for tc in tool_calls_to_replay:
            if tc.metadata.get("error"):
                continue
            if tc.name in builtin_tool_names:
                continue
            try:
                await mcp_proxy.call_tool(tc.name, **tc.arguments)
            except Exception as e:
                warning = (
                    f"History replay: tool call {tc.name}({tc.arguments!r}) failed: {e}"
                )
                logger.warning(warning)
                warnings.append(warning)
    return warnings


async def _simulate_user(
    user_model: LLMSessionBase,
    user_context: str,
    chat_history: list[Message],
    out_user_llm_history: list[list[Message]] | None = None,
    can_end_conversation: bool = False,
):
    user_sim = UserSimulator(
        user_model,
        can_end_conversation=can_end_conversation,
    )
    msg_user = await user_sim.generate(
        chat_history=chat_history,
        user_context=user_context,
    )
    assert msg_user.role == "user"

    if out_user_llm_history is not None:
        out_user_llm_history.extend(user_sim.history.copy())

    msg_user = msg_user.model_copy(deep=True)
    msg_user.metadata["is_user_llm"] = True
    return msg_user


async def run_agent_user_loop(
    tc: HydratedTestCase,
    agent_session_factory: Callable[..., AgentSessionBase],
    mcp_proxy: MCPProxyContext,
    user_model: LLMSessionBase | None = None,
    store_user_sim_history: bool = False,
    store_tool_definitions: bool = False,
    store_test_context: bool = False,
    user_can_end_conversation: bool = False,
    store_raw_messages: bool = False,
) -> DecodeResult:
    user_llm_enabled: bool = (user_model is not None) and bool(tc.user_context)
    user_llm_history: list[list[Message]] | None = (
        [] if store_user_sim_history else None
    )
    max_user_sim_turns = tc.max_user_sim_turns
    max_agent_sim_turns = tc.max_agent_sim_turns
    if user_can_end_conversation:
        # for parity with agent ending the conversation, allow one more
        max_user_sim_turns = max_user_sim_turns + 1

    timers = Timers()

    with timers.measure("time_tool"):
        mcp_tools = await mcp_proxy.list_tools()

    update_tools_with_client_config(
        mcp_tools,
        tc.scenario.tools,
    )

    query = tc.query
    user_context = tc.user_context
    test_specific_bot_instructions = tc.bot_instructions

    if tc.format_query:
        test_specific_bot_instructions = format_string_or_none(
            test_specific_bot_instructions,
            mcp_proxy.init_result,
        )
        query = format_string_or_none(query, mcp_proxy.init_result)
        user_context = format_string_or_none(user_context, mcp_proxy.init_result)

    bot_instructions = merge_bot_instructions(
        scenario_instructions=tc.scenario.bot_instructions,
        testcase_instructions=test_specific_bot_instructions,
    )

    # create agent session
    agent = None
    try:
        agent = agent_session_factory(
            mcp_proxy=mcp_proxy,
            mcp_tools=mcp_tools,
            config=tc.agent,
            bot_instructions=bot_instructions,
            scenario_metadata=tc.scenario.metadata,
        )

        if not user_can_end_conversation and not agent.can_end_conversation():
            raise ValueError(
                "Neither user nor agent can end the conversation."
                " Use an agent that can end the conversation, or allow"
                " user to end the conversation."
            )

        # Inject prior history and replay tool calls to restore MCP server state
        replay_warnings: list[str] = []
        if tc.history:
            agent.add_messages(tc.history)
            builtin_names = {t.name for t in tc.agent.builtin_tools}
            replay_warnings = await replay_history_tool_calls(
                tc.history, mcp_proxy, builtin_tool_names=builtin_names
            )

        turn: int = 0
        agent_turn_count: int = 0
        finish_reason: FinishReason = "user_limit"
        while True:
            # Check agent turn limit first — takes priority over user-LLM
            # checks so the finish reason correctly reflects why we stopped.
            if agent_turn_count >= max_agent_sim_turns:
                finish_reason = "agent_limit"
                break

            if turn == 0:
                # first turn: user message from test case query
                msg_user = Text(role="user", type="text", content=query)
            else:
                if not user_llm_enabled:
                    # no User-LLM, stop here
                    finish_reason = "no_user_llm"
                    break
                if turn > max_user_sim_turns:
                    # exceeded max number of User-LLM responses, stop
                    finish_reason = "user_limit"
                    break
                try:
                    with timers.measure("time_user"):
                        msg_user = await _simulate_user(
                            user_model,
                            user_context=user_context,
                            chat_history=agent.conversation.messages,
                            out_user_llm_history=user_llm_history,
                            can_end_conversation=user_can_end_conversation,
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Error during user LLM simulation for test case {tc.uid}: {e}"
                    ) from e

            if user_can_end_conversation:
                if msg_user.metadata.get("is_done", False):
                    agent.add_messages([msg_user])
                    finish_reason = "user_done"
                    break

            # Agent turn
            try:
                async for msg in agent.decode_turn_iter(msg_user):
                    # not saving messages because they are stored in the AgentSession
                    timers.update(agent.pop_timings())
                    # Break as soon as the limit is reached, regardless of
                    # message type.  This prevents unbounded tool-call loops
                    # when the agent never emits a non-think text response.
                    if agent_turn_count >= max_agent_sim_turns:
                        break
                    # Count each agent action toward the turn limit.
                    if isinstance(msg, ParallelToolCall) or (
                        isinstance(msg, Text)
                        and msg.role == "assistant"
                        and msg.is_visible
                    ):
                        agent_turn_count += 1
            except Exception as e:
                raise RuntimeError(
                    f"Error during decoding turn {turn} for test case {tc.uid}: {e}"
                ) from e

            if not agent.conversation.messages:
                # no completion from agent, this is unexpected
                # we still assume this to be a model error, and the decoding result to be
                # valid, since the client did not raise an exception
                finish_reason = "agent_error"
                break

            if agent.can_end_conversation():
                # detect whether the last message has the DONE marker or
                # an end turn tool call
                finish_reason_or_none = agent.should_end_conversation()
                if finish_reason_or_none is not None:
                    finish_reason = finish_reason_or_none
                    break

            turn += 1

        result_metadata = tc.metadata.copy()
        if replay_warnings:
            result_metadata["replay_warnings"] = replay_warnings
        result = DecodeResult(
            uid=tc.uid,
            messages=[],
            test_result=None,
            test_context=None,
            test_tags=tc.tags,
            tools=None,
            raw_messages=None,
            user_llm_history=user_llm_history,
            usage=None,
            metadata=result_metadata,
            finish_reason=finish_reason,
        )
        _try_store_messages(agent, result)
        if store_raw_messages:
            _try_store_raw_messages(agent, result)
        if store_tool_definitions:
            result.tools = agent.tools
        if store_test_context:
            result.test_context = await agent.make_test_context()

        timers.update(agent.pop_timings())
        timers.merge_into(result.metadata)
        return result
    except Exception as e:
        error_metadata = ErrorInfo(
            type=type(e).__name__,
            message=str(e),
            tb=traceback.format_exc(),
        )
        metadata = tc.metadata.copy()
        metadata["error"] = error_metadata.model_dump()
        # Always include conversation up to error if agent is available
        result = DecodeResult(
            uid=tc.uid,
            messages=[],
            test_result=None,
            test_context=None,
            test_tags=tc.tags,
            tools=None,
            raw_messages=None,
            user_llm_history=user_llm_history,
            usage=[],
            metadata=metadata,
            is_system_error=True,
        )
        _try_store_messages(agent, result)
        if store_raw_messages:
            _try_store_raw_messages(agent, result)
        return result


def _try_store_messages(agent: AgentSessionBase | None, result: DecodeResult):
    result.usage = []
    if not agent:
        return
    assert isinstance(
        agent.conversation, Conversation
    ), "This is initialized in AgentSessionBase and cannot be null"
    result.messages.extend(agent.conversation.messages)
    usage = agent.conversation.metadata.get("usage")
    if usage:
        result.usage.extend(usage)


def _try_store_raw_messages(agent: AgentSessionBase | None, result: DecodeResult):
    if not agent:
        return
    try:
        # save it if it's a list of dict, otherwise ignore it if
        # pydantic validation fails
        # TODO let raw_messages be Any type instead?
        result.raw_messages = agent.get_raw_messages()
    except ValueError:
        pass
