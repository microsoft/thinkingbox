# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from thinkingbox.cli.agg_main import aggregate_results, aggregate_results_per_test
from thinkingbox.common.chat_types import (
    DecodeResult,
    ParallelToolCall,
)
from thinkingbox.common.chat_types import TestResult as _TestResult
from thinkingbox.common.chat_types import (
    Text,
    ToolCall,
    ToolResponse,
)


@pytest.fixture
def conversation():
    msg_system = Text(role="system", content="a message")
    msg_user = Text(role="user", content="another message")
    msg_toolcall = ParallelToolCall(
        tool_calls=[ToolCall(name="a_tool", arguments={})],
    )
    msg_toolresp = ToolResponse(
        name="a_tool",
        content="a tool response",
        id=msg_toolcall.tool_calls[0].id,
    )
    msg_assistant = Text(role="assistant", content="a text response")
    return [
        msg_system,
        msg_user,
        msg_toolcall,
        msg_toolresp,
        msg_assistant,
    ]


def test_aggregate_results_no_errors(conversation):
    result_no_error = DecodeResult(
        uid="mytest",
        messages=conversation,
        test_result=_TestResult(result=True, reward=1.0),
    )
    results = [result_no_error] * 3
    agg = aggregate_results_per_test([r.model_copy(deep=True) for r in results])
    assert list(agg.keys()) == ["mytest"]
    res = agg["mytest"]
    assert res.runs == 3
    assert res.test_returned_vals == [True] * 3
    assert not res.failed_decodings
    assert not res.failed_assertions
    assert res.assistant_turns == [1] * 3
    assert res.user_turns == [1] * 3
    assert res.tool_turns == [1] * 3
    assert res.turns == [5] * 3

    # Test aggregate metrics including 95% CI
    metrics = aggregate_results(agg)
    assert metrics.mean_pass == 1.0
    assert metrics.mean_pass_ci_low == pytest.approx(0.4644, abs=1e-4)
    assert metrics.mean_pass_ci_high == pytest.approx(0.9998, abs=1e-4)


def test_aggregate_decoding_error(conversation):
    result_no_error = DecodeResult(
        uid="mytest",
        messages=conversation,
        test_result=_TestResult(result=True, reward=1.0),
    )
    result_dec_error = DecodeResult(
        uid="mytest",
        messages=conversation,
        test_result=None,
        is_system_error=True,
        metadata={"error": {"type": "ValueError", "message": "dec"}},
    )
    results = [
        result_no_error,
        result_dec_error,
        result_dec_error,
    ]
    agg = aggregate_results_per_test([r.model_copy(deep=True) for r in results])
    assert list(agg.keys()) == ["mytest"]
    res = agg["mytest"]
    assert res.runs == 3
    assert res.test_returned_vals == [True]
    assert res.failed_decodings == ["dec", "dec"]

    # Test aggregate metrics CI with partial success (1 pass, 2 errors)
    metrics = aggregate_results(agg)
    assert metrics.mean_pass == pytest.approx(1 / 3)
    assert metrics.mean_pass_ci_low == pytest.approx(0.0387, abs=1e-4)
    assert metrics.mean_pass_ci_high == pytest.approx(0.8233, abs=1e-4)


def test_aggregate_test_errors(conversation):
    result_no_error = DecodeResult(
        uid="mytest",
        messages=conversation,
        test_result=_TestResult(result=True, reward=1.0),
    )
    result_test_fail = DecodeResult(
        uid="mytest",
        messages=conversation,
        test_result=_TestResult(
            result=False,
            reward=0.0,
            lineno=1,
            line_content="assert False",
            tb="AssertionError",
        ),
    )
    result_test_error = DecodeResult(
        uid="mytest",
        messages=conversation,
        test_result=_TestResult(
            result=False,
            reward=0.0,
            lineno=1,
            line_content="assert 0/0",
            tb="ZeroDivisionError: division by zero",
            is_system_error=True,
        ),
    )
    results = [
        result_no_error,
        result_test_fail,
        result_test_error,
    ]
    agg = aggregate_results_per_test([r.model_copy(deep=True) for r in results])
    assert list(agg.keys()) == ["mytest"]
    res = agg["mytest"]
    assert res.runs == 3
    assert res.test_returned_vals == [True, False]
    assert len(res.failed_decodings) == 1
    assert "assert 0/0" in res.failed_decodings[0]
    assert len(res.failed_assertions) == 1
    assert "assert False" in res.failed_assertions[0]
