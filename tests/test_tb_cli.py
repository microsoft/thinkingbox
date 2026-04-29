# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path

import pytest

from thinkingbox.cli import infer as tb_cli
from thinkingbox.common.chat_types import DecodeResult, Message, Text
from thinkingbox.common.config_types import (
    ConfigFile,
    CustomSessionConfig,
    ThinkingBoxOrchestratorConfig,
)


def get_agent():
    return """
system_instructions: You like apples
builtin_tools: []
"""


def get_scenario_data(bot_instructions: str | None = None):
    return """
world_state: {}
tools: []
bot_instructions: There are some scenario bot instructions
"""


def get_test_data():
    return '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_1(x: TestContext, judge: Judge):
    """!
    query: An apple a day...
    bot_instructions: "There are some test case bot instructions"
    """
    assert judge.text_yesno(x.response, "say yes")

def test_2(x: TestContext, judge: Judge):
    """!
    query: Two apples a day...
    """
    assert judge.text_yesno(x.response, "say no")
'''


def get_test_data_valid():
    """Test code that runs without errors (no assertions on agent behavior)."""
    return '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_valid(x: TestContext, judge: Judge):
    """!
    query: Hello world
    """
    # This test always passes — just validates code structure
    assert isinstance(x, TestContext)
'''


def get_test_data_two_valid():
    """Two always-passing tests used for previous-results testing."""
    return '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_a(x: TestContext, judge: Judge):
    """!
    query: Hello
    """
    assert isinstance(x, TestContext)

def test_b(x: TestContext, judge: Judge):
    """!
    query: World
    """
    assert isinstance(x, TestContext)
'''


def get_test_data_with_name_error():
    """Test code with a NameError (references undefined variable)."""
    return '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_broken(x: TestContext, judge: Judge):
    """!
    query: Hello world
    """
    result = undefined_variable  # noqa: F821 — intentional NameError
    assert result
'''


def get_config(
    agent_messages: list[list[Message]] = None,
    judge_messages: list[list[Message]] = None,
):
    return ConfigFile(
        mcp_proxy="http://127.0.0.1:7111",
        orchestrator=ThinkingBoxOrchestratorConfig(
            agent_model=CustomSessionConfig(
                factory="tests.mock_session.create_mock_session",
                completions=agent_messages or [],
            ),
        ),
        user_model=None,
        judge_model=CustomSessionConfig(
            factory="tests.mock_session.create_mock_session",
            completions=judge_messages or [],
        ),
    )


def _create_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _concatenate_conversation_text(messages: list[Message]):
    out = []
    for msg in messages:
        if isinstance(msg, Text):
            out.append(msg.content)
    return "\n".join(out)


@pytest.mark.asyncio
async def test_tb_cli_bot_instructions(tmp_path):
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(bot_instructions="Scenario instructions"),
    )
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        get_test_data(),
    )
    output_path = tmp_path / "output.jsonl"
    agent_messages = [
        [
            Text(role="assistant", content="thinking...", metadata={"tag": "think"}),
            Text(role="assistant", content="<DONE>", metadata={"tag": "text"}),
        ]
    ]
    judge_messages_yes = [
        [Text(role="assistant", content="Yes", metadata={"tag": "text"})]
    ]
    judge_messages_no = [
        [Text(role="assistant", content="No", metadata={"tag": "text"})]
    ]

    # test_1 should contain test case bot instructions in conversation text
    await tb_cli.async_main(
        decoder_args={
            "config": get_config(
                agent_messages=agent_messages.copy(),
                judge_messages=judge_messages_yes.copy(),
            ),
            "skip_test": False,
        },
        dataset=str(tmp_path),
        agent="myagent",
        input_name="mytest.py:test_1",
        input_file_or_folder=None,
        test_list_file=None,
        output=output_path,
    )
    with open(output_path, "r", encoding="utf-8") as f:
        output = DecodeResult(**json.load(f))
    assert (
        not output.is_system_error
    ), "There was an error during decoding, error information: " + str(
        output.metadata.get("error")
    )
    assert output.test_result.result, "Test should pass"
    all_text = _concatenate_conversation_text(output.messages)
    assert "There are some scenario bot instructions" in all_text
    assert "There are some test case bot instructions" in all_text

    # test_2 should NOT contain test case bot instructions in conversation text
    await tb_cli.async_main(
        decoder_args={
            "config": get_config(
                agent_messages=agent_messages.copy(),
                judge_messages=judge_messages_no.copy(),
            ),
            "skip_test": False,
        },
        dataset=str(tmp_path),
        agent="myagent",
        input_name="mytest.py:test_2",
        input_file_or_folder=None,
        test_list_file=None,
        output=output_path,
    )
    with open(output_path, "r", encoding="utf-8") as f:
        output = DecodeResult(**json.load(f))
    assert (
        not output.is_system_error
    ), "There was an error during decoding, error information: " + str(
        output.metadata.get("error")
    )
    assert not output.test_result.result, "Test should not pass"
    all_text = _concatenate_conversation_text(output.messages)
    assert "There are some scenario bot instructions" in all_text
    assert "There are some test case bot instructions" not in all_text


@pytest.mark.asyncio
async def test_skip_agent_valid_test(tmp_path):
    """--skip-agent with valid test code: is_system_error=false, test runs without code errors."""
    _create_file(tmp_path / "agent" / "myagent.yaml", get_agent())
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(),
    )
    _create_file(tmp_path / "test_case" / "mytest.py", get_test_data_valid())
    output_path = tmp_path / "output.jsonl"

    await tb_cli.async_main(
        decoder_args={
            "config": get_config(),
            "skip_test": False,
            "skip_agent": True,
        },
        dataset=str(tmp_path),
        agent="myagent",
        input_name="mytest.py:test_valid",
        input_file_or_folder=None,
        test_list_file=None,
        output=output_path,
    )
    with open(output_path, "r", encoding="utf-8") as f:
        output = DecodeResult(**json.load(f))

    assert not output.is_system_error, "Init should succeed: " + str(
        output.metadata.get("error")
    )
    assert output.finish_reason == "skipped"
    assert output.messages == []  # no agent messages
    assert output.test_result is not None, "Test should have run"
    assert (
        not output.test_result.is_system_error
    ), "Test code should have no structural errors: " + str(output.test_result.tb)


@pytest.mark.asyncio
async def test_skip_agent_code_error(tmp_path):
    """--skip-agent with broken test code: is_system_error=false but test_result.is_system_error=true."""
    _create_file(tmp_path / "agent" / "myagent.yaml", get_agent())
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(),
    )
    _create_file(tmp_path / "test_case" / "mytest.py", get_test_data_with_name_error())
    output_path = tmp_path / "output.jsonl"

    await tb_cli.async_main(
        decoder_args={
            "config": get_config(),
            "skip_test": False,
            "skip_agent": True,
        },
        dataset=str(tmp_path),
        agent="myagent",
        input_name="mytest.py:test_broken",
        input_file_or_folder=None,
        test_list_file=None,
        output=output_path,
    )
    with open(output_path, "r", encoding="utf-8") as f:
        output = DecodeResult(**json.load(f))

    assert not output.is_system_error, "Init should succeed: " + str(
        output.metadata.get("error")
    )
    assert output.finish_reason == "skipped"
    assert output.test_result is not None, "Test should have run"
    assert (
        output.test_result.is_system_error
    ), "Test code has NameError — should be a system error in test_result"


@pytest.mark.asyncio
async def test_skip_agent_no_test(tmp_path):
    """--skip-agent with --no-test: only validates init, no test execution."""
    _create_file(tmp_path / "agent" / "myagent.yaml", get_agent())
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(),
    )
    _create_file(tmp_path / "test_case" / "mytest.py", get_test_data_valid())
    output_path = tmp_path / "output.jsonl"

    await tb_cli.async_main(
        decoder_args={
            "config": get_config(),
            "skip_test": True,
            "skip_agent": True,
        },
        dataset=str(tmp_path),
        agent="myagent",
        input_name="mytest.py:test_valid",
        input_file_or_folder=None,
        test_list_file=None,
        output=output_path,
    )
    with open(output_path, "r", encoding="utf-8") as f:
        output = DecodeResult(**json.load(f))

    assert not output.is_system_error, "Init should succeed: " + str(
        output.metadata.get("error")
    )
    assert output.finish_reason == "skipped"
    assert output.test_result is None, "Test should not have run with --no-test"


@pytest.mark.asyncio
async def test_previous_results_file_reruns_system_errors(tmp_path):
    """--previous-results-file: system errors are re-run; successful results are reused."""
    _create_file(tmp_path / "agent" / "myagent.yaml", get_agent())
    _create_file(tmp_path / "scenario" / "myscenario.yaml", get_scenario_data())
    _create_file(tmp_path / "test_case" / "mytest.py", get_test_data_two_valid())
    output_path = tmp_path / "output.jsonl"
    previous_results_path = tmp_path / "previous.jsonl"

    # test_a: system error — should be re-run
    error_result = DecodeResult(
        uid="mytest.py:test_a",
        messages=[],
        test_result=None,
        test_context=None,
        test_tags=[],
        tools=None,
        raw_messages=None,
        user_llm_history=None,
        usage=[],
        metadata={
            "repetition": 0,
            "error": {"type": "RuntimeError", "message": "simulated failure", "tb": ""},
        },
        is_system_error=True,
    )
    # test_b: successful previous result — should be reused as-is
    cached_result = DecodeResult(
        uid="mytest.py:test_b",
        messages=[
            Text(role="assistant", content="cached response", metadata={"tag": "text"})
        ],
        test_result=None,
        test_context=None,
        test_tags=[],
        tools=None,
        raw_messages=None,
        user_llm_history=None,
        usage=[],
        metadata={"repetition": 0},
        is_system_error=False,
    )
    previous_results_path.write_text(
        error_result.model_dump_json() + "\n" + cached_result.model_dump_json() + "\n"
    )

    # Only one agent response needed — test_a is re-run, test_b is reused from previous
    agent_messages = [
        [Text(role="assistant", content="<DONE>", metadata={"tag": "text"})]
    ]

    await tb_cli.async_main(
        decoder_args={
            "config": get_config(agent_messages=agent_messages),
            "skip_test": True,
        },
        dataset=str(tmp_path),
        agent="myagent",
        input_file_or_folder=str(tmp_path / "test_case" / "mytest.py"),
        input_name=None,
        test_list_file=None,
        output=output_path,
        previous_results_file=str(previous_results_path),
    )

    with open(output_path, "r", encoding="utf-8") as f:
        results = [DecodeResult(**json.loads(line)) for line in f]

    result_a = next(r for r in results if r.uid == "mytest.py:test_a")
    result_b = next(r for r in results if r.uid == "mytest.py:test_b")

    assert (
        not result_a.is_system_error
    ), "System error should have been re-run, not reused"
    assert result_a.messages != [], "Re-run should have produced agent messages"

    assert not result_b.is_system_error, "Successful result should have been reused"
    assert (
        len(result_b.messages) == 1
        and result_b.messages[0].content == "cached response"
    ), "Reused result should match the previous result exactly"


@pytest.mark.asyncio
async def test_skip_agent_dump_testcontext(tmp_path):
    """--skip-agent with --dump testcontext should persist test_context in output."""
    _create_file(tmp_path / "agent" / "myagent.yaml", get_agent())
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(),
    )
    _create_file(tmp_path / "test_case" / "mytest.py", get_test_data_valid())
    output_path = tmp_path / "output.jsonl"

    await tb_cli.async_main(
        decoder_args={
            "config": get_config(),
            "skip_test": True,
            "skip_agent": True,
            "dump_testcontext": True,
        },
        dataset=str(tmp_path),
        agent="myagent",
        input_name="mytest.py:test_valid",
        input_file_or_folder=None,
        test_list_file=None,
        output=output_path,
    )
    with open(output_path, "r", encoding="utf-8") as f:
        output = DecodeResult(**json.load(f))

    assert not output.is_system_error, "Init should succeed: " + str(
        output.metadata.get("error")
    )
    assert output.finish_reason == "skipped"
    assert output.test_result is None, "Test should not run with --no-test"
    assert output.test_context is not None, "test_context should be dumped"
    assert output.test_context.session_id
    assert isinstance(output.test_context.init_result, dict)
