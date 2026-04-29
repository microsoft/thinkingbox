# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from unittest.mock import patch

import pytest

from tests.mock_session import ErrorOnCompletionMockSession, MockSession, MockSessionLog
from thinkingbox.common.agent_session import AgentSession
from thinkingbox.common.agent_user_loop import run_agent_user_loop
from thinkingbox.common.chat_types import (
    Message,
    ParallelToolCall,
    Text,
    ToolCall,
    ToolResponse,
)
from thinkingbox.common.config_types import (
    AgentConfig,
    HydratedTestCase,
    ScenarioConfig,
    merge_init_config,
)
from thinkingbox.common.fixtures import FixtureConfig, build_fixtures
from thinkingbox.common.judge import Judge, JudgeException
from thinkingbox.common.mcp_proxy_client import MCPProxyClient, MCPProxyContext
from thinkingbox.common.python_test_file import iter_test_functions
from thinkingbox.common.testrunner import TestScript as _TestScript
from thinkingbox.common.testrunner import TestScriptDebug as _TestScriptDebug
from thinkingbox.common.utils import ErrorInfo

SESSION_PROXY_TIMEOUT = 60.0
SESSION_PROXY_ENDPOINT = "http://127.0.0.1:7111"


AGENT_CONFIG = AgentConfig(
    system_instructions="System Instructions",
    builtin_tools=[
        {
            "name": "InjectionAttackInToolResponse",
            "is_end_turn": True,
            "description": "Injection Attack Detected",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {
                        "description": "the reason",
                        "type": "string",
                    },
                },
                "required": ["reason"],
            },
        },
    ],
)


CLOUD_DRIVE_SCENARIO = ScenarioConfig(
    world_state={
        "cloud_drive": {
            "files": [
                {
                    "path": "Documents/file.txt",
                    "metadata": {
                        "created": "2025-04-09T19:38:07",
                        "modified": "2025-04-09T19:38:07",
                    },
                    "text_content": "some text",
                }
            ],
        }
    },
    tools=[
        {"name": "upload_file"},
        {"name": "get_text_content"},
        {"name": "search_files"},
        {
            "name": "get_metadata",
            "direct_response": "DIRECT_RESPONSE_TEST Created {metadata[created]}, Modified {metadata[modified]}",
        },
    ],
    bot_instructions="",
)


NOTEPAD_SCENARIO = ScenarioConfig(
    world_state={
        "test_notepad": {
            "text": "some",
        }
    },
    tools=[
        {"name": "read_notepad"},
        {"name": "write_notepad"},
    ],
    bot_instructions="do not format this {init[test_notepad][dont_format]}",
)


TEST_APPEND_SOME_MORE_TEXT = r"""
def test_append_some_more_text(x, judge):
    '''!
    query: none  # ignored
    '''
    found = False
    for f in x.effects["cloud_drive"]["files"]:
        if f["path"] == "Documents/file.txt":
            found = True
            assert f["text_content"] == "some text\nsome more text"
            break
    assert found
    assert judge.text_yesno(
        x.response, "Does the message confirm that file.txt was modified?"
    )
""".lstrip()


TEST_FORMAT_QUERY = r"""
def test_format_query(x, judge):
    '''!
    query: none  # ignored
    '''
    assert x.session_id.strip("0-") != "", "session ID empty or default"
    assert x.init_result["test_notepad"]["query_fmt"] == "ABC"
""".lstrip()


TEST_APPEND_SOME_MORE_TEXT_LINGER_AND_FIXTURES = r"""
def test_append_some_more_text(x, judge, check_target_file_fn, check_target_file_ctx, session_client):
    '''!
    query: none  # ignored
    '''
    # use a function fixture
    check_target_file_fn(x.effects["cloud_drive"], "some text\nsome more text")

    # use a context manager fixture
    check_target_file_ctx(x.effects["cloud_drive"], "some text\nsome more text")

    # use a MCP session proxy client fixture to get the file
    # content by calling a tool
    session_client.set_session(x.session_id)
    out_dict = session_client.call_json_tool(
        "cloud_drive",
        "get_text_content",
        {"path": "Documents/file.txt"},
    )
    assert out_dict["text_content"] == "some text\nsome more text"
""".lstrip()


TEST_WITH_RUBRIC_JUDGE = r"""
from thinkingbox.common.rubrics_judge import RubricJudge

def test_rubric_evaluation(x, judge):
    '''!
    query: none
    '''
    rubric_judge = RubricJudge(judge=judge, x=x, max_workers=1)
    rubrics = [
        rubric_judge.Config(criterion="Does the response acknowledge the request?", weight=50.0),
        rubric_judge.Config(criterion="Is the response clear?", weight=50.0),
    ]

    result = rubric_judge.evaluate(
        response=x.response,
        rubrics=rubrics,
        global_threshold=0.5,
        throw_on_failure=True,
    )

    return result.reward
""".lstrip()


def _get_test_code(test_fn_code: str):
    test_fns = list(iter_test_functions(test_fn_code, check_config=True))
    assert (
        len(test_fns) == 1
    ), "this is a problem with the unit test, fix the test code string"
    return test_fns[0].test_code


def _msg_think():
    return Text(role="assistant", content="thinking", metadata={"tag": "think"})


def _msg_dummy():
    return Text(role="assistant", content="", metadata={"is_dummy": True})


def _msg_text(content: str):
    return Text(role="assistant", content=content, metadata={"tag": "text"})


def _msg_tool_call(name, arguments):
    return ParallelToolCall(
        tool_calls=[
            ToolCall(name=name, arguments=arguments, metadata={"error": None}),
        ]
    )


def _agent_session_factory(
    agent_completions: list | None = None, *, log: MockSessionLog | None = None
):
    session = MockSession(completions=agent_completions or [], log=log)

    def factory(**kwargs):
        return AgentSession.from_config(agent_model=session, **kwargs)

    return factory


def _error_agent_session_factory(
    agent_completions: list | None = None,
    *,
    log: MockSessionLog | None = None,
    error_type=Exception,
    error_message="Simulated error",
    error_on_call=0,
):
    session = ErrorOnCompletionMockSession(
        completions=agent_completions or [],
        log=log,
        error_type=error_type,
        error_message=error_message,
        error_on_call=error_on_call,
    )

    def factory(**kwargs):
        return AgentSession.from_config(agent_model=session, **kwargs)

    return factory


@pytest.mark.asyncio
async def test_no_userllm_with_test():
    """Test a conversation without User-LLM, including Judge"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_tool_call("search_files", {"pattern": "**/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_tool_call("get_text_content", {"path": "Documents/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_tool_call(
                "upload_file",
                {
                    "path": "Documents/file.txt",
                    "text_content": "some text\nsome more text",
                    "overwrite": True,
                },
            ),
        ],
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
    ]
    agent_log = MockSessionLog()
    agent_factory = _agent_session_factory(agent_completions, log=agent_log)
    judge_llm = MockSession(
        completions=[[_msg_text("yes")]],
    )
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        test_code=_get_test_code(TEST_APPEND_SOME_MORE_TEXT),
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=agent_factory,
            mcp_proxy=mcp_proxy,
            user_model=None,
            store_test_context=True,
        )
    assert result.test_context is not None

    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"
    test = _TestScript(
        tc.test_code,
        Judge(judge_llm),
    )
    test_result = await test.evaluate(result.test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert not judge_llm.completions, "Not all judge messages have been emitted"


@pytest.mark.asyncio
async def test_with_userllm():
    """Test a conversation with User-LLM"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("what file? not done"),
        ],
        [
            _msg_think(),
            _msg_tool_call("get_text_content", {"path": "Documents/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
    ]
    agent_log = MockSessionLog()
    user_llm = MockSession(
        completions=[[Text(role="user", content="this file")]],
    )
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="file?",
        test_code="",
        user_context="some",
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
        )

    assert result.messages[-1].metadata.get(
        "is_done", False
    ), "last agent message is not done"
    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"
    assert not user_llm.completions, "Not all user messages have been emitted"


@pytest.mark.asyncio
async def test_with_directresponse():
    """Test a conversation with direct responses"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_tool_call("get_text_content", {"path": "Documents/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_tool_call("get_metadata", {"path": "Documents/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
    ]
    agent_log = MockSessionLog()
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="content and metadata?",
        test_code="",
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=None,
        )

    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"
    assert result.messages[-1].metadata.get(
        "is_done", False
    ), "last agent message is not done"

    # check that we have one direct response
    found = 0
    for msg in result.messages:
        if isinstance(msg, Text) and msg.role == "assistant":
            is_direct = msg.tag == "direct"
            if is_direct and "DIRECT_RESPONSE_TEST" in msg.content:
                found += 1
    assert found == 1, "expected exactly one direct response containing the test string"


@pytest.mark.asyncio
async def test_with_userllm_agent_error():
    """Test a conversation with User-LLM where the agent LLM raises an error"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("what file? not done"),
        ],
        # Second completion will raise error
    ]

    user_llm = MockSession(
        completions=[[Text(role="user", content="this file")]],
    )
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="file?",
        test_code="",
        user_context="some",
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_error_agent_session_factory(
                agent_completions,
                error_message="Simulated agent LLM error",
                error_on_call=1,
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
        )
        assert result.is_system_error
        assert len(result.messages) > 0
        error = ErrorInfo(**result.metadata.get("error", {}))
        assert "Simulated agent LLM error" in error.message


@pytest.mark.asyncio
async def test_with_userllm_userllm_error():
    """Test a conversation with User-LLM where the user LLM raises an error"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("what file? not done"),
        ],
        [
            _msg_think(),
            _msg_tool_call("get_text_content", {"path": "Documents/file.txt"}),
        ],
    ]

    user_llm = ErrorOnCompletionMockSession(
        completions=[[Text(role="user", content="this file")]],
        error_message="Simulated user LLM error",
        error_on_call=0,  # Error on the first call
    )
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="file?",
        test_code="",
        user_context="some",
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(agent_completions),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
        )
        assert result.is_system_error
        assert len(result.messages) > 0
        error = ErrorInfo(**result.metadata.get("error", {}))
        assert "Simulated user LLM error" in error.message


@pytest.mark.asyncio
async def test_no_userllm_tool_call_error():
    """Test a conversation without User-LLM, where the mcp_proxy.call_tool raises an error"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_tool_call("search_files", {"pattern": "**/file.txt"}),
        ],
    ]
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        test_code="",
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        with patch.object(
            mcp_proxy, "call_tool", side_effect=Exception("Simulated tool call error")
        ) as mock_call_tool:
            result = await run_agent_user_loop(
                tc,
                agent_session_factory=_agent_session_factory(agent_completions),
                mcp_proxy=mcp_proxy,
                user_model=None,
            )
            assert result.is_system_error
            assert len(result.messages) > 0
            error = ErrorInfo(**result.metadata.get("error", {}))
            assert "Simulated tool call error" in error.message
            assert (
                mock_call_tool.call_count == 1
            ), "Tool call should have been attempted once"


@pytest.mark.asyncio
async def test_no_userllm_with_test_judge_error():
    """Test a conversation without User-LLM, where the Judge LLM raises an error"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_tool_call("search_files", {"pattern": "**/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_tool_call("get_text_content", {"path": "Documents/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_tool_call(
                "upload_file",
                {
                    "path": "Documents/file.txt",
                    "text_content": "some text\nsome more text",
                    "overwrite": True,
                },
            ),
        ],
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
    ]
    agent_log = MockSessionLog()
    judge_llm = ErrorOnCompletionMockSession(
        completions=[[_msg_text("yes")]],
        error_type=JudgeException,
        error_message="Simulated judge LLM error",
        error_on_call=0,  # Error on the first call
    )
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        test_code=_get_test_code(TEST_APPEND_SOME_MORE_TEXT),
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=None,
            store_test_context=True,
        )
    assert result.test_context is not None

    # verify the expected number of messages are in test context
    assert len(result.test_context.messages) == 13

    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"
    test = _TestScript(
        tc.test_code,
        Judge(judge_llm),
    )

    # verify the assertion error gets encoded into the eval result correctly
    failing_test_context = result.test_context.model_copy(deep=True)
    failing_test_context.effects["cloud_drive"]["files"] = []
    test_result = await test.evaluate(failing_test_context)
    assert not test_result.result, "Test should fail due to missing file"
    assert not test_result.is_system_error, "This should not be flagged as system error"

    # verify that the judge error is raised
    test_result = await test.evaluate(result.test_context)
    assert not test_result.result, "Test should not report pass"
    assert test_result.is_system_error, "Test should be flagged as system error"
    assert "Simulated judge LLM error" in str(
        test_result.tb
    ), "Judge error message mismatch"

    # verify that other exceptions also raise correctly
    failing_test_context = result.test_context.model_copy(deep=True)
    # this will cause a KeyError because the file expects a key called 'path'
    failing_test_context.effects["cloud_drive"]["files"] = [{"wrong_key": "value"}]
    test_result = await test.evaluate(failing_test_context)
    assert test_result.is_system_error, "Test should be flagged as system error"
    assert "KeyError" in str(test_result.tb), "Judge error message mismatch"


@pytest.mark.asyncio
async def test_with_testcase_init():
    """Test a conversation where the testcase has additional init to merge"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_tool_call("search_files", {"pattern": "**/*.txt"}),
        ],
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
    ]
    agent_log = MockSessionLog()
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        test_code="",
        init={
            "cloud_drive": {
                "files": [
                    {
                        "path": "Documents/more.txt",
                        "text_content": "some more text",
                    },
                ]
            }
        },
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=None,
            store_test_context=True,
        )
    assert result.test_context is not None
    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"

    files = json.loads(result.test_context.tool_calls[0].tool_response.content)["files"]
    file_paths = set(f["path"] for f in files)
    assert file_paths == {"Documents/file.txt", "Documents/more.txt"}


def _get_text_message(
    messages: list[Message],
    filter_role: str,
    found_index: int,
) -> Text | None:
    out = [
        msg for msg in messages if (isinstance(msg, Text) and (msg.role == filter_role))
    ]
    try:
        return out[found_index]
    except IndexError:
        return None


@pytest.mark.asyncio
async def test_with_formatting():
    """Test a conversation with formatting from init"""
    scenario = NOTEPAD_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("what notepad? not done"),
        ],
        [
            _msg_think(),
            _msg_tool_call("read_notepad", {}),
        ],
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
    ]
    user_llm = MockSession(
        completions=[[Text(role="user", content="there's only one...")]],
    )
    judge_llm = MockSession(
        completions=[[_msg_text("yes")]],
    )
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="read notepad? {init[test_notepad][query_fmt]}",
        bot_instructions="bot instructions {init[test_notepad][bot_instructions_fmt]}",
        format_query=True,
        test_code=_get_test_code(TEST_FORMAT_QUERY),
        user_context="some {init[test_notepad][user_context_fmt]}",
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=scenario.world_state,
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(agent_completions),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            store_user_sim_history=True,
            store_test_context=True,
        )
    # {init[test_notepad][dont_format]}
    msg_bot_instructions = _get_text_message(result.messages, "system", -1)
    assert isinstance(msg_bot_instructions, Text), "bot instructions message not found"

    msg_user = _get_text_message(result.messages, "user", 0)
    assert isinstance(msg_user, Text), "user message not found"

    user_sim_text = "\n".join(
        msg.content for msg in result.user_llm_history[0] if isinstance(msg, Text)
    )

    assert (
        "{init[test_notepad][dont_format]}" in msg_bot_instructions.content
    ), "Bot instructions from scenario should not have been formatted"
    assert "ABC" in msg_user.content
    assert "DEF" in msg_bot_instructions.content
    assert "XYZ" in user_sim_text

    test = _TestScript(
        tc.test_code,
        Judge(judge_llm),
    )
    test_result = await test.evaluate(result.test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"


@pytest.mark.asyncio
async def test_linger_session(tmp_path):
    """Verify that it's possible to call tools from the test"""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_tool_call("search_files", {"pattern": "**/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_tool_call("get_text_content", {"path": "Documents/file.txt"}),
        ],
        [
            _msg_think(),
            _msg_tool_call(
                "upload_file",
                {
                    "path": "Documents/file.txt",
                    "text_content": "some text\nsome more text",
                    "overwrite": True,
                },
            ),
        ],
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
    ]
    agent_log = MockSessionLog()
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        test_code=_get_test_code(TEST_APPEND_SOME_MORE_TEXT_LINGER_AND_FIXTURES),
    )
    fixtures_config = {
        "check_target_file_fn": FixtureConfig(
            type="tests.tb_fixtures.check_target_file_fn",
            filename="Documents/file.txt",
        ),
        "check_target_file_ctx": FixtureConfig(
            type="tests.tb_fixtures.CheckTargetFileCtx",
            filename="Documents/file.txt",
        ),
        "session_client": FixtureConfig(
            type="thinkingbox.common.SessionClientFixture",
            endpoint=SESSION_PROXY_ENDPOINT,
            timeout=SESSION_PROXY_TIMEOUT,
        ),
    }
    mcp_proxy: MCPProxyContext | None = None
    try:
        async with MCPProxyClient.session_context(
            endpoint=SESSION_PROXY_ENDPOINT,
            timeout=SESSION_PROXY_TIMEOUT,
            server_config=scenario.world_state,
            available_tools=[t.name for t in scenario.tools],
        ) as mcp_proxy:
            mcp_proxy.linger()
            result = await run_agent_user_loop(
                tc,
                agent_session_factory=_agent_session_factory(
                    agent_completions, log=agent_log
                ),
                mcp_proxy=mcp_proxy,
                user_model=None,
                store_test_context=True,
            )
            assert result.test_context is not None

            assert (
                agent_log.remaining_completions == 0
            ), "Not all agent messages have been emitted"

        judge_llm = MockSession(completions=[])
        test = _TestScript(
            tc.test_code,
            Judge(judge_llm),
            fixtures=build_fixtures(fixtures_config),
        )
        test_result = await test.evaluate(result.test_context)
        assert test_result.result, f"test result is False, traceback: {test_result.tb}"

        # Using TestScriptDebug
        source_file = tmp_path / "my_tests.py"
        with open(source_file, "w", encoding="utf-8") as f:
            f.write(tc.test_code)
        judge_llm = MockSession(completions=[[_msg_text("yes")]])
        test = _TestScriptDebug(
            test_name="test_append_some_more_text",
            source_file=source_file,
            judge=Judge(judge_llm),
            fixtures=build_fixtures(fixtures_config),
        )
        test_result = await test.evaluate(result.test_context)
        assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    finally:
        if mcp_proxy is not None:
            await mcp_proxy.client.session_destroy()


@pytest.mark.asyncio
async def test_rubric_judge_integration():
    """Test RubricJudge with mocked conversation and test execution."""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [_msg_think(), _msg_text("I acknowledge your request. <DONE>")],
    ]
    judge_llm = MockSession(
        completions=[
            [_msg_text("yes")],
            [_msg_text("yes")],
        ]
    )

    tc = HydratedTestCase(
        uid="test_rubric_integration",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="please do something",
        test_code=_get_test_code(TEST_WITH_RUBRIC_JUDGE),
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(agent_completions),
            mcp_proxy=mcp_proxy,
            user_model=None,
            store_test_context=True,
        )

    assert result.test_context is not None
    assert "acknowledge" in result.test_context.response

    test = _TestScript(tc.test_code, Judge(judge_llm))
    test_result = await test.evaluate(result.test_context)

    assert test_result.result
    assert not test_result.is_system_error
    assert test_result.reward == 1.0


@pytest.mark.asyncio
async def test_text_tc_then_notext():
    # Anthropic Messages
    # thinking: ...
    # text: ... (not done)
    # toolcall: my_tool()
    # tool: my_tool response
    # thinking: ...
    # [dummy]
    # -> should return control to user

    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("explanation"),
            _msg_tool_call("search_files", {"pattern": "**/file.txt"}),
        ],
        # tool response
        [
            _msg_think(),
            _msg_dummy(),  # nothing to add... -> not done so follow up with user
        ],
        # user
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
        # STOP
    ]
    agent_log = MockSessionLog()
    user_llm = MockSession(completions=[[Text(role="user", content="this file")]])
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        user_context="some",
        test_code="",
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            store_test_context=False,
        )
    assert not result.is_system_error, f"Unexpected system error: {result.metadata}"
    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"
    assert not user_llm.completions, "Not all user messages have been emitted"
    assert (
        len([msg for msg in result.messages if isinstance(msg, ToolResponse)]) == 1
    ), "Tool was not called exactly once"


@pytest.mark.asyncio
async def test_textdone_tc_then_notext():
    # Anthropic Messages
    # thinking: ...
    # text: ... <DONE>
    # toolcall: my_tool()
    # tool: my_tool response
    # thinking: ...
    # [dummy]
    # -> should stop

    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
            _msg_tool_call("search_files", {"pattern": "**/file.txt"}),
        ],
        # tool response
        [
            _msg_think(),
            _msg_dummy(),  # nothing to add... -> <DONE> was correct
        ],
        # STOP
    ]
    agent_log = MockSessionLog()
    user_llm = MockSession(completions=[])
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        user_context="some",
        test_code="",
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            store_test_context=True,
        )
    assert not result.is_system_error, f"Unexpected system error: {result.metadata}"
    assert (
        result.finish_reason == "done"
    ), f"Expected 'done' but got '{result.finish_reason}'"
    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"
    assert (
        len([msg for msg in result.messages if isinstance(msg, ToolResponse)]) == 1
    ), "Tool was not called exactly once"
    assert result.test_context.response == "ok <DONE>"


@pytest.mark.asyncio
async def test_textdone_tc_then_text():
    # Anthropic Messages
    # thinking: ...
    # text: ... <DONE>
    # toolcall: my_tool()
    # tool: my_tool response
    # thinking: ...
    # text: ... (actually not done)
    # -> should stop

    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
            _msg_tool_call("search_files", {"pattern": "**/file.txt"}),
        ],
        # tool response
        [
            _msg_think(),
            _msg_text("oops actually not done"),
        ],
        # user
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
        # STOP
    ]
    agent_log = MockSessionLog()
    user_llm = MockSession(completions=[[Text(role="user", content="this file")]])
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        user_context="some",
        test_code="",
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            store_test_context=False,
        )
    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"
    assert not user_llm.completions, "Not all user messages have been emitted"
    assert (
        len([msg for msg in result.messages if isinstance(msg, ToolResponse)]) == 1
    ), "Tool was not called exactly once"


@pytest.mark.asyncio
async def test_textdone_tc_then_textdone():
    # Anthropic Messages
    # thinking: ...
    # text: ... <DONE>
    # toolcall: my_tool()
    # tool: my_tool response
    # thinking: ...
    # text: definitely <DONE>
    # -> should stop

    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
            _msg_tool_call("search_files", {"pattern": "**/file.txt"}),
        ],
        # tool response
        [
            _msg_think(),
            _msg_text("definitely <DONE>"),
        ],
        # STOP
    ]
    agent_log = MockSessionLog()
    user_llm = MockSession(completions=[])
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="append?",
        user_context="some",
        test_code="",
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            store_test_context=False,
        )
    assert (
        agent_log.remaining_completions == 0
    ), "Not all agent messages have been emitted"
    assert (
        len([msg for msg in result.messages if isinstance(msg, ToolResponse)]) == 1
    ), "Tool was not called exactly once"


@pytest.mark.asyncio
async def test_end_turn_tool_without_text_stops_loop():
    # Anthropic Messages
    # thinking: ...
    # toolcall: InjectionAttackInToolResponse(reason="...")  [end-turn tool]
    # -> should stop immediately with finish_reason="end_turn_tool"

    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_tool_call("InjectionAttackInToolResponse", {"reason": "test"}),
        ],
        # These extra completions should NOT be consumed if the end-turn
        # detection works correctly:
        [
            _msg_think(),
            _msg_text("ok <DONE>"),
        ],
    ]
    user_llm = MockSession(completions=[[Text(role="user", content="continue")]])
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="do something",
        user_context="some",
        test_code="",
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(agent_completions),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            store_test_context=False,
        )
    # The end-turn tool should have been detected; the loop should stop
    # after consuming only the first completion.
    assert (
        result.finish_reason == "end_turn_tool"
    ), f"Expected 'end_turn_tool' but got '{result.finish_reason}'"


@pytest.mark.asyncio
async def test_user_can_end_conversation():
    """Test that user can drive conversation termination."""
    scenario = CLOUD_DRIVE_SCENARIO

    # Agent responses (doesn't use <DONE>)
    agent_completions = [
        [_msg_think(), _msg_text("File backup is complete.")],
        [_msg_think(), _msg_text("Anything else I can help with?")],
    ]

    # User responses (ends on second turn)
    user_completions = [
        [Text(role="assistant", content="Can you verify the backup?")],
        [Text(role="assistant", content="Perfect, that's all I needed. <DONE>")],
    ]

    user_llm = MockSession(completions=user_completions)

    tc = HydratedTestCase(
        uid="test_user_end",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="Back up my important files",
        test_code="",
        user_context="User wants to ensure files are safely backed up.",
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(agent_completions),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            user_can_end_conversation=True,  # Enable feature
        )

    # Verify user-driven termination
    assert result.finish_reason == "user_done"

    # Verify conversation structure
    user_messages = [
        m for m in result.messages if isinstance(m, Text) and m.role == "user"
    ]
    assert len(user_messages) >= 3  # Initial + followup + ending

    # Last user message should have <DONE>
    last_user = user_messages[-1]
    assert "<DONE>" in last_user.content
    assert last_user.metadata.get("is_done") is True


@pytest.mark.asyncio
async def test_user_end_takes_precedence():
    """Test that user ending happens before agent turn."""
    scenario = CLOUD_DRIVE_SCENARIO

    # Agent would end with <DONE> if given chance
    agent_completions = [
        [_msg_think(), _msg_text("Task complete.")],
        [_msg_think(), _msg_text("All done! <DONE>")],  # Would end here
    ]

    # User ends first
    user_completions = [
        [Text(role="assistant", content="Thanks, I'm satisfied! <DONE>")],
    ]

    user_llm = MockSession(completions=user_completions)

    tc = HydratedTestCase(
        uid="test_precedence",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="Simple task",
        test_code="",
        user_context="User needs quick answer.",
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(agent_completions),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            user_can_end_conversation=True,
        )

    # User ended, not agent
    assert result.finish_reason == "user_done"

    # Agent's second response (with <DONE>) should not be in conversation
    # Count only visible text messages (not thinking messages)
    agent_text_messages = [
        m
        for m in result.messages
        if isinstance(m, Text)
        and m.role == "assistant"
        and m.metadata.get("tag") == "text"
    ]
    # Should only have first agent response, not the second with <DONE>
    assert len(agent_text_messages) == 1  # Only first response


@pytest.mark.asyncio
async def test_user_cannot_end_when_disabled():
    """Test that user can't end conversation when feature disabled."""
    scenario = CLOUD_DRIVE_SCENARIO

    # Agent ends normally
    agent_completions = [
        [_msg_think(), _msg_text("Done! <DONE>")],
    ]

    # User tries to end but feature disabled
    user_completions = [
        [Text(role="assistant", content="Actually I'm done too <DONE>")],
    ]

    user_llm = MockSession(completions=user_completions)

    tc = HydratedTestCase(
        uid="test_disabled",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="Quick task",
        test_code="",
        user_context="User context",
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(agent_completions),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            user_can_end_conversation=False,  # Feature disabled
        )

    # Agent ended, not user
    assert result.finish_reason == "done"  # Agent's <DONE>

    # User's <DONE> should be ignored
    user_messages = [
        m for m in result.messages if isinstance(m, Text) and m.role == "user"
    ]
    # Check that user messages don't have is_done set to True (feature disabled)
    for user_msg in user_messages:
        assert user_msg.metadata.get("is_done") is not True


@pytest.mark.asyncio
async def test_max_agent_sim_turns_limit():
    """Test that agent is limited by max_agent_sim_turns config.

    max_agent_sim_turns counts ParallelToolCall and non-think assistant text
    messages yielded from decode_turn_iter.  Each completion here produces
    1 counted message (the text response — no tool calls in this test),
    so max_agent_sim_turns=2 allows exactly 2 completions before stopping.
    """
    scenario = CLOUD_DRIVE_SCENARIO
    # Agent completions: 2 non-done turns, then a 3rd that should never run
    agent_completions = [
        [
            _msg_think(),
            _msg_text("response 1, not done"),
        ],
        [
            _msg_think(),
            _msg_text("response 2, not done"),
        ],
        [
            _msg_think(),
            _msg_text("response 3, should never run <DONE>"),
        ],
    ]
    agent_log = MockSessionLog()
    user_llm = MockSession(
        completions=[
            [Text(role="user", content="user msg 1")],
            [Text(role="user", content="user msg 2")],
        ],
    )
    tc = HydratedTestCase(
        uid="test_agent_limit",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="start",
        test_code="",
        user_context="some context",
        max_agent_sim_turns=2,
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
        )

    assert not result.is_system_error, f"Unexpected system error: {result.metadata}"
    assert (
        result.finish_reason == "agent_limit"
    ), f"Expected 'agent_limit' but got '{result.finish_reason}'"
    # Should have exactly 2 assistant text responses (excluding think messages)
    assistant_msgs = [
        m
        for m in result.messages
        if isinstance(m, Text)
        and m.role == "assistant"
        and m.metadata.get("tag") != "think"
    ]
    assert (
        len(assistant_msgs) == 2
    ), f"Expected 2 assistant messages, got {len(assistant_msgs)}"


@pytest.mark.asyncio
async def test_max_agent_sim_turns_zero():
    """Test that max_agent_sim_turns=0 prevents any agent decode."""
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        [
            _msg_think(),
            _msg_text("should never run <DONE>"),
        ],
    ]
    tc = HydratedTestCase(
        uid="test_agent_limit_zero",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="start",
        test_code="",
        max_agent_sim_turns=0,
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(agent_completions),
            mcp_proxy=mcp_proxy,
            user_model=None,
        )

    assert not result.is_system_error, f"Unexpected system error: {result.metadata}"
    assert (
        result.finish_reason == "agent_limit"
    ), f"Expected 'agent_limit' but got '{result.finish_reason}'"
    # No assistant messages should exist
    assistant_msgs = [
        m for m in result.messages if isinstance(m, Text) and m.role == "assistant"
    ]
    assert (
        len(assistant_msgs) == 0
    ), f"Expected 0 assistant messages, got {len(assistant_msgs)}"


@pytest.mark.asyncio
async def test_max_agent_sim_turns_default_unlimited():
    """Test that max_agent_sim_turns defaults to sys.maxsize (unlimited)."""
    import sys

    tc = HydratedTestCase(
        uid="test_default",
        agent=AGENT_CONFIG,
        scenario=CLOUD_DRIVE_SCENARIO,
        query="start",
        test_code="",
    )
    assert tc.max_agent_sim_turns == sys.maxsize


@pytest.mark.asyncio
async def test_max_agent_sim_turns_inner_loop():
    """Test that max_agent_sim_turns counts tool calls and text responses within
    a single decode_turn_iter call.

    The agent does: tool_call (count=1) → text (count=2, limit hit)
    The tool_call counts as an agent turn because ParallelToolCall is counted.
    The text response also counts.  With max_agent_sim_turns=2, the limit is
    reached after the first tool_call+text pair.
    """
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        # Inner iteration 1: tool call (count=1), tools execute, loop continues
        [_msg_think(), _msg_tool_call("search_files", {"pattern": "**/a.txt"})],
        # Inner iteration 1 cont: text response (count=2, limit hit), breaks
        [_msg_think(), _msg_text("found a.txt, not done yet")],
        # Should never run — limit already reached
        [_msg_think(), _msg_tool_call("search_files", {"pattern": "**/b.txt"})],
        [_msg_think(), _msg_text("found b.txt, not done yet")],
        [_msg_think(), _msg_text("done <DONE>")],
    ]
    agent_log = MockSessionLog()
    user_llm = MockSession(
        completions=[
            [Text(role="user", content="keep going")],
            [Text(role="user", content="keep going")],
        ],
    )
    tc = HydratedTestCase(
        uid="test_agent_limit_inner",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="start",
        test_code="",
        user_context="some context",
        max_agent_sim_turns=2,
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
        )

    assert not result.is_system_error, f"Unexpected system error: {result.metadata}"
    assert (
        result.finish_reason == "agent_limit"
    ), f"Expected 'agent_limit' but got '{result.finish_reason}'"
    # Only 2 of 5 completions consumed (tool_call + text), 3 remaining
    assert (
        agent_log.remaining_completions == 3
    ), f"Expected 3 remaining completions, got {agent_log.remaining_completions}"


@pytest.mark.asyncio
async def test_max_agent_sim_turns_reasoning_model():
    """Test that max_agent_sim_turns caps reasoning models that produce many
    think+tool_call completions before a final text answer.

    Reasoning models emit [Text(think), ParallelToolCall] for each step.
    Each ParallelToolCall counts as one agent turn.  With max_agent_sim_turns=3,
    the agent is allowed 3 tool calls.  The 3rd tool call brings count to 3;
    the break fires on the next yield (the tool response after execution).
    """
    scenario = CLOUD_DRIVE_SCENARIO
    agent_completions = [
        # Step 1: think + tool_call (count=1)
        [_msg_think(), _msg_tool_call("search_files", {"pattern": "**/a.txt"})],
        # Step 2: think + tool_call (count=2)
        [_msg_think(), _msg_tool_call("search_files", {"pattern": "**/b.txt"})],
        # Step 3: think + tool_call (count=3, limit reached after tool executes)
        [_msg_think(), _msg_tool_call("search_files", {"pattern": "**/c.txt"})],
        # Should never run — limit already reached
        [_msg_think(), _msg_text("found everything, not done yet")],
        [_msg_think(), _msg_text("done <DONE>")],
    ]
    agent_log = MockSessionLog()
    user_llm = MockSession(
        completions=[
            [Text(role="user", content="keep going")],
        ],
    )
    tc = HydratedTestCase(
        uid="test_reasoning_model_limit",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="start",
        test_code="",
        user_context="some context",
        max_agent_sim_turns=3,
    )

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=_agent_session_factory(
                agent_completions, log=agent_log
            ),
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
        )

    assert not result.is_system_error, f"Unexpected system error: {result.metadata}"
    assert (
        result.finish_reason == "agent_limit"
    ), f"Expected 'agent_limit' but got '{result.finish_reason}'"
    # 3 of 5 completions consumed (3 tool calls), 2 remaining
    assert (
        agent_log.remaining_completions == 2
    ), f"Expected 2 remaining completions, got {agent_log.remaining_completions}"


@pytest.mark.asyncio
async def test_decode_turn_iter_break_no_extra_messages():
    """Test that breaking out of decode_turn_iter mid-iteration does not leave
    un-yielded messages in conversation.messages.

    A single llm.get_completion() call returns [think, tool_call] as one batch.
    Before the per-message add+yield fix, the entire batch was added to
    conversation.messages *before* the yield loop, so even if the caller broke
    after the first yield (think), the tool_call would already be persisted.

    With the fix, add_messages and yield happen together per message, so
    breaking after the think means the tool_call is never added.
    """
    scenario = CLOUD_DRIVE_SCENARIO
    # One get_completion() call → two messages returned as a batch
    think_msg = _msg_think()
    tool_call_msg = _msg_tool_call("search_files", {"pattern": "**/a.txt"})
    agent_completions = [
        [think_msg, tool_call_msg],
    ]

    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, {}),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        agent_model = MockSession(completions=agent_completions)
        agent = AgentSession.from_config(
            config=AGENT_CONFIG,
            mcp_proxy=mcp_proxy,
            mcp_tools=await mcp_proxy.list_tools(),
            bot_instructions=None,
            scenario_metadata={},
            agent_model=agent_model,
        )
        prefix_count = len(agent.conversation.messages)

        user_msg = Text(role="user", content="find a.txt")
        yielded: list[Message] = []
        async for msg in agent.decode_turn_iter(user_msg):
            yielded.append(msg)
            break  # abandon generator after the first yielded message

        # Only the think should have been yielded
        assert len(yielded) == 1
        assert (
            isinstance(yielded[0], Text) and yielded[0].metadata.get("tag") == "think"
        )

        # The tool_call from the same batch should NOT be in conversation
        conv_tool_calls = [
            m for m in agent.conversation.messages if isinstance(m, ParallelToolCall)
        ]
        assert len(conv_tool_calls) == 0, (
            "tool_call from the same get_completion() batch was added to "
            "conversation despite never being yielded to the caller"
        )

        # Total: prefix + user_msg + 1 yielded think
        expected = prefix_count + 1 + 1
        actual = len(agent.conversation.messages)
        assert (
            actual == expected
        ), f"conversation.messages has {actual} entries but expected {expected}"
