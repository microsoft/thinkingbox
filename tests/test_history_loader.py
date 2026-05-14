# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for history_loader.py: HistoryRef, MetaFile, HistoryLoader, parse_messages."""

import textwrap
from pathlib import Path

import pytest

from thinkingbox.common.chat_types import (
    ParallelToolCall,
    Text,
    ToolCall,
    ToolResponse,
)
from thinkingbox.common.config_types import TestCase
from thinkingbox.common.history_loader import (
    HistoryLoader,
    HistoryRef,
    MetaFile,
    _slice_messages,
    parse_messages,
)
from thinkingbox.common.hydrator import iter_cases_from_file_or_folder


def test_history_ref_parse_basic():
    """Parse a ref with integer start and empty end."""
    ref = HistoryRef.parse("mycase:0:")
    assert ref.key == "mycase"
    assert ref.start == 0
    assert ref.end == ""


def test_history_ref_parse_with_end():
    """Parse a ref with integer start and message_id end boundary."""
    ref = HistoryRef.parse("mycase:0:t2")
    assert ref.key == "mycase"
    assert ref.start == 0
    assert ref.end == "t2"


def test_history_ref_parse_message_id_start():
    """Parse a ref where start is a message_id string instead of an integer."""
    ref = HistoryRef.parse("mycase:t1:t2")
    assert ref.key == "mycase"
    assert ref.start == "t1"
    assert ref.end == "t2"


def test_history_ref_parse_key_with_colon():
    """rsplit from right preserves colons in the group key."""
    ref = HistoryRef.parse("Benefits & Leave:0:t2")
    assert ref.key == "Benefits & Leave"
    assert ref.start == 0
    assert ref.end == "t2"


def test_history_ref_parse_wrong_parts():
    """Raise ValueError when the string does not have exactly 3 colon-separated parts."""
    with pytest.raises(ValueError, match="3 colon-separated parts"):
        HistoryRef.parse("nostart")


def test_history_ref_parse_empty_key():
    """Raise ValueError when the key part is empty."""
    with pytest.raises(ValueError, match="key must not be empty"):
        HistoryRef.parse(":0:t2")


def test_history_ref_jsonl_roundtrip():
    """HistoryRef survives Pydantic model_dump/model_validate roundtrip."""
    ref = HistoryRef(key="mycase", start=0, end="t2")
    tc = TestCase(uid="t1", scenario="s", query="q", test_code="", history=ref)
    roundtripped = TestCase.model_validate(tc.model_dump())
    assert roundtripped.history == ref


MINIMAL_META = textwrap.dedent(
    """\
    $history:
      mygroup:
        - T: Text
          message_id: t1
          role: user
          content: Hello
        - T: Text
          role: assistant
          content: World
"""
)


def test_metafile_load_history(tmp_path):
    """History groups under $history: are stored as raw message lists."""
    p = tmp_path / "test.meta.yaml"
    p.write_text(MINIMAL_META)
    mf = MetaFile.load(p)
    assert "mygroup" in mf.history_raw
    assert len(mf.history_raw["mygroup"]) == 2


def test_metafile_load_skips_comment_keys(tmp_path):
    """History keys starting with '#' are treated as comments and ignored."""
    content = textwrap.dedent(
        """\
        $history:
          '# a comment': null
          realgroup:
            - T: Text
              role: user
              content: hi
    """
    )
    p = tmp_path / "test.meta.yaml"
    p.write_text(content)
    mf = MetaFile.load(p)
    assert "# a comment" not in mf.history_raw
    assert "realgroup" in mf.history_raw


def test_metafile_load_invalid_root(tmp_path):
    """Raise ValueError when the file root is not a YAML mapping."""
    p = tmp_path / "test.meta.yaml"
    p.write_text("- a list")
    with pytest.raises(ValueError, match="YAML mapping"):
        MetaFile.load(p)


def test_metafile_load_for_test_file_absent(tmp_path):
    """Return None when no .meta.yaml exists alongside the test file."""
    py_file = tmp_path / "mytest.py"
    py_file.write_text("")
    assert MetaFile.load_for_test_file(py_file) is None


RAW_MESSAGES = [
    {"T": "Text", "message_id": "t1", "role": "user", "content": "Hello"},
    {"T": "Text", "role": "assistant", "content": "Hi"},
    {"T": "Text", "message_id": "t2", "role": "user", "content": "Follow-up"},
    {"T": "Text", "role": "assistant", "content": "Done"},
]


def test_slice_integer_start_with_end_id():
    """Integer start with a message_id end returns messages up to that boundary."""
    result = _slice_messages(RAW_MESSAGES, 0, "t2")
    assert len(result) == 2
    assert result[0]["message_id"] == "t1"


def test_slice_message_id_start_to_end():
    """message_id start with empty end returns from that turn to the end."""
    result = _slice_messages(RAW_MESSAGES, "t2", "")
    assert len(result) == 2
    assert result[0]["message_id"] == "t2"


def test_slice_message_id_start_with_end_id():
    """message_id start is inclusive, message_id end is exclusive."""
    result = _slice_messages(RAW_MESSAGES, "t1", "t2")
    assert len(result) == 2
    assert result[0]["message_id"] == "t1"
    assert result[1]["role"] == "assistant"  # message between t1 and t2, no message_id


def test_slice_out_of_range_start():
    """Raise ValueError when the integer start index exceeds the list length."""
    with pytest.raises(ValueError, match="out of range"):
        _slice_messages(RAW_MESSAGES, 10, "")


def test_slice_end_before_start():
    """Raise ValueError when the end boundary resolves before the start boundary."""
    with pytest.raises(ValueError, match="not after start"):
        _slice_messages(RAW_MESSAGES, "t2", "t1")


def test_history_loader_resolve_raw(tmp_path):
    """resolve_raw returns the correct raw message slice for a valid HistoryRef."""
    p = tmp_path / "test.meta.yaml"
    p.write_text(MINIMAL_META)
    mf = MetaFile.load(p)
    loader = HistoryLoader(mf)
    ref = HistoryRef.parse("mygroup:0:")
    raw = loader.resolve_raw(ref)
    assert len(raw) == 2


def test_history_loader_no_meta_file():
    """Raise KeyError when resolve_raw is called with no MetaFile loaded."""
    loader = HistoryLoader(None)
    ref = HistoryRef.parse("mygroup:0:")
    with pytest.raises(KeyError, match="No .meta.yaml"):
        loader.resolve_raw(ref)


def test_history_loader_missing_key(tmp_path):
    """Raise KeyError when the requested history group key is absent from the file."""
    p = tmp_path / "test.meta.yaml"
    p.write_text(MINIMAL_META)
    mf = MetaFile.load(p)
    loader = HistoryLoader(mf)
    ref = HistoryRef.parse("nonexistent:0:")
    with pytest.raises(KeyError, match="not found"):
        loader.resolve_raw(ref)


RAW_TEXT = {"T": "Text", "role": "user", "content": "Hello"}
RAW_TOOL_CALL = {
    "T": "ToolCall",
    "name": "search",
    "arguments": {"query": "test"},
    "id": "call_123",
}
RAW_TOOL_RESPONSE = {
    "T": "ToolResponse",
    "name": "search",
    "content": '{"status": "ok"}',
    "id": "call_123",
}
RAW_PARALLEL = {
    "T": "ParallelToolCall",
    "tool_calls": [RAW_TOOL_CALL],
}


def test_parse_message_text():
    """Text message is correctly parsed with role and content."""
    msg = parse_messages([RAW_TEXT])[0]
    assert isinstance(msg, Text)
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_parse_message_tool_call_arguments():
    """ToolCall message is parsed with name, arguments, and id."""
    msg = parse_messages([RAW_TOOL_CALL])[0]
    assert isinstance(msg, ToolCall)
    assert msg.name == "search"
    assert msg.arguments == {"query": "test"}
    assert msg.id == "call_123"


def test_parse_message_tool_response():
    """ToolResponse message is parsed with content and id."""
    msg = parse_messages([RAW_TOOL_RESPONSE])[0]
    assert isinstance(msg, ToolResponse)
    assert msg.content == '{"status": "ok"}'


def test_parse_message_parallel_tool_call():
    """ParallelToolCall message is parsed with its inner tool calls."""
    msg = parse_messages([RAW_PARALLEL])[0]
    assert isinstance(msg, ParallelToolCall)
    assert len(msg.tool_calls) == 1


def test_parse_message_unknown_type():
    """An unrecognised T discriminator value raises a validation error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse_messages([{"T": "Unknown", "role": "user", "content": ""}])


def test_parse_message_no_message_id_in_metadata():
    """message_id at top level is an extra field and is not stored in the message metadata."""
    raw = {"T": "Text", "message_id": "t1", "role": "user", "content": "Hi"}
    msg = parse_messages([raw])[0]
    assert "message_id" not in msg.metadata


def test_iter_test_functions_metadata_rejected_by_check_config():
    """metadata: in a test docstring is not a supported CONFIG_FIELD and raises when check_config=True."""
    from thinkingbox.common.python_test_file import iter_test_functions

    code = textwrap.dedent(
        """\
        \"\"\"!
        scenario: test_scenario
        \"\"\"
        def my_case(x, judge):
            \"\"\"!
            query: hello
            metadata:
              query_id: QS_1
            \"\"\"
            pass
    """
    )
    with pytest.raises(ValueError, match="invalid key"):
        list(iter_test_functions(code, check_config=True))


def test_iter_test_functions_history_parsed_to_ref():
    """history: string is parsed into a HistoryRef object."""
    from thinkingbox.common.python_test_file import iter_test_functions

    code = textwrap.dedent(
        """\
        \"\"\"!
        scenario: test_scenario
        \"\"\"
        def my_case(x, judge):
            \"\"\"!
            query: follow-up
            history: "mygroup:0:"
            \"\"\"
            pass
    """
    )
    fns = list(iter_test_functions(code, check_config=True))
    ref = fns[0].config["history"]
    assert isinstance(ref, HistoryRef)
    assert ref.key == "mygroup"
    assert ref.start == 0
    assert ref.end == ""


AGENT_YAML = textwrap.dedent(
    """\
    system_instructions: Test agent
    builtin_tools: []
"""
)

SCENARIO_YAML = textwrap.dedent(
    """\
    world_state:
      test_server: {}
    tools:
    - name: test_tool
      description: a tool
      input_schema:
        type: object
        properties: {}
"""
)

META_YAML = textwrap.dedent(
    """\
    my_case:
      QueryID: QS_42
      Group: TestGroup

    $history:
      mygroup:
        - T: Text
          message_id: t1
          role: user
          content: Prior user message
        - T: Text
          role: assistant
          content: Prior assistant message
"""
)

TEST_PY_WITH_HISTORY = textwrap.dedent(
    """\
    from thinkingbox.common import Judge, TestContext

    \"\"\"!
    scenario: myscenario
    \"\"\"

    def my_case(x: TestContext, judge: Judge):
        \"\"\"!
        query: Follow-up question
        history: "mygroup:0:"
        \"\"\"
        pass
"""
)


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_hydrate_with_history_ref(tmp_path):
    """history: field is resolved into list[Message] in HydratedTestCase."""
    _write(tmp_path / "agent" / "myagent.yaml", AGENT_YAML)
    _write(tmp_path / "scenario" / "myscenario.yaml", SCENARIO_YAML)
    _write(tmp_path / "test_case" / "mytest.py", TEST_PY_WITH_HISTORY)
    _write(tmp_path / "test_case" / "mytest.meta.yaml", META_YAML)

    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 1
    tc = cases[0]
    assert tc.history is not None
    assert len(tc.history) == 2
    assert isinstance(tc.history[0], Text)
    assert tc.history[0].role == "user"
    assert tc.history[0].content == "Prior user message"
    assert isinstance(tc.history[1], Text)
    assert tc.history[1].role == "assistant"


def test_hydrate_no_meta_file(tmp_path):
    """Test cases without .meta.yaml work normally (no history, no external metadata)."""
    test_py = textwrap.dedent(
        """\
        from thinkingbox.common import Judge, TestContext

        \"\"\"!
        scenario: myscenario
        \"\"\"

        def my_case(x: TestContext, judge: Judge):
            \"\"\"!
            query: hello
            \"\"\"
            pass
    """
    )
    _write(tmp_path / "agent" / "myagent.yaml", AGENT_YAML)
    _write(tmp_path / "scenario" / "myscenario.yaml", SCENARIO_YAML)
    _write(tmp_path / "test_case" / "mytest.py", test_py)

    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 1
    assert cases[0].history is None
    assert cases[0].metadata.get("scenario") == "myscenario"


@pytest.mark.asyncio
async def test_replay_executes_tool_calls_in_order():
    """ToolCall and ParallelToolCall entries in history are each replayed against the proxy."""
    from unittest.mock import AsyncMock, call

    from thinkingbox.common.agent_user_loop import replay_history_tool_calls
    from thinkingbox.common.chat_types import ParallelToolCall, Text, ToolCall

    history = [
        Text(role="user", content="prior user message"),
        ToolCall(
            name="upload_file",
            arguments={"path": "a.txt", "text_content": "hello", "overwrite": False},
            id="c1",
        ),
        ParallelToolCall(
            tool_calls=[
                ToolCall(name="get_text_content", arguments={"path": "b.txt"}, id="c2"),
                ToolCall(name="delete_file", arguments={"path": "c.txt"}, id="c3"),
            ]
        ),
        Text(role="assistant", content="done"),
    ]
    proxy = AsyncMock()

    warnings = await replay_history_tool_calls(history, proxy)

    assert proxy.call_tool.call_count == 3
    proxy.call_tool.assert_any_call(
        "upload_file", path="a.txt", text_content="hello", overwrite=False
    )
    proxy.call_tool.assert_any_call("get_text_content", path="b.txt")
    proxy.call_tool.assert_any_call("delete_file", path="c.txt")
    assert warnings == []


@pytest.mark.asyncio
async def test_replay_skips_non_tool_call_messages():
    """Text and ToolResponse messages in history are not replayed."""
    from unittest.mock import AsyncMock

    from thinkingbox.common.agent_user_loop import replay_history_tool_calls
    from thinkingbox.common.chat_types import Text, ToolResponse

    history = [
        Text(role="user", content="question"),
        ToolResponse(name="search", content='{"status": "ok"}', id="c1"),
        Text(role="assistant", content="answer"),
    ]
    proxy = AsyncMock()

    warnings = await replay_history_tool_calls(history, proxy)

    proxy.call_tool.assert_not_called()
    assert warnings == []


@pytest.mark.asyncio
async def test_replay_logs_warning_on_error_and_continues():
    """A failed tool replay produces a warning and does not raise; remaining calls still execute."""
    from unittest.mock import AsyncMock

    from thinkingbox.common.agent_user_loop import replay_history_tool_calls
    from thinkingbox.common.chat_types import ToolCall

    history = [
        ToolCall(name="bad_tool", arguments={}, id="c1"),
        ToolCall(name="good_tool", arguments={"path": "x.txt"}, id="c2"),
    ]
    proxy = AsyncMock()
    proxy.call_tool.side_effect = [RuntimeError("server error"), None]

    warnings = await replay_history_tool_calls(history, proxy)

    assert proxy.call_tool.call_count == 2
    assert len(warnings) == 1
    assert "bad_tool" in warnings[0]
    assert "server error" in warnings[0]


@pytest.mark.asyncio
async def test_replay_skips_tool_calls_with_error_metadata():
    """Tool calls that originally failed (error in metadata) are not replayed."""
    from unittest.mock import AsyncMock

    from thinkingbox.common.agent_user_loop import replay_history_tool_calls
    from thinkingbox.common.chat_types import ToolCall

    history = [
        ToolCall(name="good_tool", arguments={"path": "a.txt"}, id="c1"),
        ToolCall(
            name="bad_tool",
            arguments={},
            id="c2",
            metadata={"error": "function 'bad_tool' does not exist"},
        ),
    ]
    proxy = AsyncMock()

    warnings = await replay_history_tool_calls(history, proxy)

    proxy.call_tool.assert_called_once_with("good_tool", path="a.txt")
    assert warnings == []


@pytest.mark.asyncio
async def test_replay_skips_builtin_tool_calls():
    """Builtin tool calls are not replayed — they are handled by the framework, not MCP."""
    from unittest.mock import AsyncMock

    from thinkingbox.common.agent_user_loop import replay_history_tool_calls
    from thinkingbox.common.chat_types import ToolCall

    history = [
        ToolCall(
            name="upload_file",
            arguments={"path": "a.txt", "text_content": "x", "overwrite": False},
            id="c1",
        ),
        ToolCall(
            name="InjectionAttackInToolResponse",
            arguments={"reason": "attack detected"},
            id="c2",
        ),
    ]
    proxy = AsyncMock()

    warnings = await replay_history_tool_calls(
        history, proxy, builtin_tool_names={"InjectionAttackInToolResponse"}
    )

    proxy.call_tool.assert_called_once_with(
        "upload_file", path="a.txt", text_content="x", overwrite=False
    )
    assert warnings == []
