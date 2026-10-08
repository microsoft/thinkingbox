import math

import pytest

from thinkingbox.cli.agg_main import (
    PerTestCaseResults,
    PerTestCaseTableRow,
    aggregate_benchmark_metrics,
    aggregate_dataset_metrics,
    aggregate_results,
    aggregate_results_per_test,
    build_instance_metrics,
    format_result,
)
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
from thinkingbox.cli.stats import (
    pass_at_k,
    pass_at_k_exact_se,
    pass_power_k,
    pass_power_k_exact_se,
    pass_rate_and_se,
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


def _per_test_result(runs: int, successes: int) -> PerTestCaseResults:
    return PerTestCaseResults(
        runs=runs,
        test_returned_vals=[True] * successes + [False] * (runs - successes),
    )


def test_statistical_results_use_at_most_four_significant_digits():
    assert format_result(0.123456) == "0.1235"
    assert format_result(12.3456) == "12.35"
    assert format_result(0.0000123456) == "1.235e-05"


def test_per_test_table_hides_operational_averages_by_default():
    columns = PerTestCaseTableRow.columns()
    default_columns = {name for name, column in columns.items() if not column.details}
    detail_columns = {name for name, column in columns.items() if column.details}

    assert detail_columns == {
        "assistant_turns",
        "user_turns",
        "tool_turns",
        "output_tokens",
        "reasoning_tokens",
        "char_lengths",
        "exec_time",
    }
    assert {"success_pct", "pass_at_n", "pass_power_n"} <= default_columns


def test_k_one_metrics_match_the_pass_rate_and_its_standard_error():
    pass_rate, pass_rate_ste = pass_rate_and_se(20, 7)

    assert pass_at_k(20, 7, 1) == pytest.approx(pass_rate)
    assert pass_at_k_exact_se(20, 7, 1) == pytest.approx(pass_rate_ste)
    assert pass_power_k(20, 7, 1) == pytest.approx(pass_rate)
    assert pass_power_k_exact_se(20, 7, 1) == pytest.approx(pass_rate_ste)


def test_hierarchical_metrics_aggregate_instances_then_datasets():
    results = {
        "alpha.py:one": _per_test_result(20, 10),
        "alpha.py:two": _per_test_result(20, 5),
        "beta.py:one": _per_test_result(20, 15),
    }

    instances, runs_per_test = build_instance_metrics(results)
    datasets = aggregate_dataset_metrics(instances)
    overall = aggregate_results(instances, datasets, runs_per_test)

    alpha, beta = datasets
    assert alpha.dataset == "alpha.py"
    assert alpha.pass_rate.estimate == pytest.approx(0.375)
    assert beta.dataset == "beta.py"
    assert beta.pass_rate.estimate == pytest.approx(0.75)

    alpha_within_variance = (0.5 * 0.5 / 20 + 0.25 * 0.75 / 20) / 4
    alpha_between_variance = ((0.5 - 0.25) ** 2 / 2) / 2
    assert alpha.pass_rate.ste == pytest.approx(
        math.sqrt(alpha_within_variance + alpha_between_variance)
    )

    instance_pass_rates = [0.5, 0.25, 0.75]
    instance_pass_rate_stes = [
        math.sqrt(0.5 * 0.5 / 20),
        math.sqrt(0.25 * 0.75 / 20),
        math.sqrt(0.75 * 0.25 / 20),
    ]
    expected_overall_ste = math.sqrt(
        sum(ste**2 for ste in instance_pass_rate_stes) / len(instance_pass_rates) ** 2
        + (sum((rate - 0.5) ** 2 for rate in instance_pass_rates) / 2)
        / len(instance_pass_rates)
    )
    assert overall.mean_pass == pytest.approx(0.5)
    assert overall.mean_pass_ste == pytest.approx(expected_overall_ste)
    assert [metric.k for metric in overall.pass_at_k] == [1, 5, 10, 20]
    assert [metric.k for metric in overall.pass_power_k] == [1, 5, 10, 20]

    alpha_instance = next(
        metric for metric in instances if metric.uid == "alpha.py:one"
    )
    assert alpha_instance.pass_at_n.estimate == pytest.approx(1.0)
    assert alpha_instance.pass_at_n.ste > 0.0
    assert alpha_instance.pass_power_n.estimate == pytest.approx(0.5**20)
    assert alpha_instance.pass_power_n.ste > 0.0


def test_benchmark_metrics_merge_suffix_matched_datasets_without_duplication():
    results = {
        "external_booking_v1_group1.py:base": _per_test_result(20, 10),
        "external_booking_v1_group1_rubrics_yesno.py:rubric": _per_test_result(20, 5),
        "other.py:only": _per_test_result(20, 15),
    }
    instances, runs_per_test = build_instance_metrics(results)
    datasets = aggregate_dataset_metrics(instances)
    benchmarks = aggregate_benchmark_metrics(instances, ("_rubrics_yesno",))
    overall = aggregate_results(instances, datasets, runs_per_test)

    assert len(benchmarks) == 1
    benchmark = benchmarks[0]
    assert benchmark.dataset == "external_booking_v1_group1.py"
    assert benchmark.source_datasets == [
        "external_booking_v1_group1.py",
        "external_booking_v1_group1_rubrics_yesno.py",
    ]
    assert benchmark.num_tests == 2
    assert benchmark.total_runs == 40
    assert benchmark.pass_rate.estimate == pytest.approx(0.375)
    assert overall.num_tests == 3
    assert overall.mean_pass == pytest.approx(0.5)


def test_benchmark_metrics_are_empty_without_aggregation_suffixes():
    instances, _ = build_instance_metrics(
        {"external_booking_v1_group1.py:base": _per_test_result(20, 10)}
    )

    assert aggregate_benchmark_metrics(instances, ()) == []


def test_instance_metrics_require_shared_sample_count():
    results = {
        "alpha.py:one": _per_test_result(10, 5),
        "alpha.py:two": _per_test_result(20, 5),
    }

    with pytest.raises(ValueError, match="same number of samples"):
        build_instance_metrics(results)


def test_instance_metrics_require_dataset_prefixed_uid():
    with pytest.raises(ValueError, match="dataset.py:instance_name"):
        build_instance_metrics({"missing-prefix": _per_test_result(20, 5)})
