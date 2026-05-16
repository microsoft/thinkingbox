# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sys
from typing import Annotated, Any, Callable, Iterable

import click
import numpy as np
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text as RichText

from thinkingbox.common.chat_types import (
    DecodeResult,
    Text,
    ToolResponse,
)
from thinkingbox.common.eval_utils import cred_int, prob_in_zone
from thinkingbox.common.utils import ErrorInfo, iter_validate_jsonl


class PerTestCaseResults(BaseModel):
    test_returned_vals: list[bool] = Field(default_factory=list)
    failed_decodings: list[str] = Field(default_factory=list)
    failed_assertions: list[str] = Field(default_factory=list)
    assistant_turns: list[int] = Field(default_factory=list)
    user_turns: list[int] = Field(default_factory=list)
    tool_turns: list[int] = Field(default_factory=list)
    turns: list[int] = Field(default_factory=list)
    content_char_length: list[int] = Field(default_factory=list)
    execution_time: list[float] = Field(default_factory=list)
    output_tokens: list[int] = Field(default_factory=list)
    reasoning_tokens: list[int] = Field(default_factory=list)
    runs: int = 0
    likelihood_in_goldilocks_zone: float = 0.0
    in_goldilocks_zone: bool | None = None


def safe_min(arr: list, default=None):
    return min(arr) if arr else default


def safe_max(arr: list, default=None):
    return max(arr) if arr else default


def safe_mean(arr: list, default=None):
    if not arr:
        return default
    return sum(arr) / len(arr)


class AvgMinMax(BaseModel):
    avg_val: float
    min_val: float
    max_val: float
    count: int

    @classmethod
    def from_list(cls, values: list) -> "AvgMinMax":
        return cls(
            avg_val=safe_mean(values, 0.0),
            min_val=safe_min(values, 0.0),
            max_val=safe_max(values, 0.0),
            count=len(values),
        )

    def to_str(self, avg_decimal_places: int = 0, min_max_decimal_places: int = 0):
        if self.count:
            fmt = "{avg_val:.%df} ({min_val:.%df}-{max_val:.%df})" % (
                avg_decimal_places,
                min_max_decimal_places,
                min_max_decimal_places,
            )
            return fmt.format(
                avg_val=self.avg_val, min_val=self.min_val, max_val=self.max_val
            )
        return "N/A"

    def __str__(self):
        return self.to_str()


class Column:
    def __init__(
        self, name: str, concise: bool = False, format: Callable | None = None, **kwargs
    ):
        self.name = name
        self.concise = concise
        self.format = format if (format is not None) else str
        self.kwargs = kwargs


class TableModel(BaseModel):
    @classmethod
    def columns(cls) -> dict[str, Column]:
        out: dict[str, Column] = {}
        for name, f in cls.model_fields.items():
            column = [x for x in f.metadata if isinstance(x, Column)]
            assert len(column) == 1, f"Missing column annotation for {name}"
            out[name] = column[0]
        return out


class PerTestCaseTableRow(TableModel):
    uid: Annotated[str, Column("Test Case ID", concise=True, style="cyan")]
    runs: Annotated[int, Column("Runs", concise=True, justify="right", style="green")]
    passed: Annotated[int, Column("Pass", concise=True, justify="right", style="green")]
    failed: Annotated[int, Column("Fail", concise=True, justify="right", style="red")]
    error: Annotated[int, Column("Error", concise=True, justify="right", style="red")]
    assistant_turns: Annotated[
        AvgMinMax, Column("Avg-Ast(min,max)", justify="right", style="blue")
    ]
    user_turns: Annotated[
        AvgMinMax, Column("Avg-Usr(min,max)", justify="right", style="blue")
    ]
    tool_turns: Annotated[
        AvgMinMax, Column("Avg-TC(min,max)", justify="right", style="blue")
    ]
    output_tokens: Annotated[
        AvgMinMax,
        Column("Avg-OutTkns(min,max)", justify="right", style="blue"),
    ]
    reasoning_tokens: Annotated[
        AvgMinMax,
        Column("Avg-ReaTkns(min,max)", justify="right", style="blue"),
    ]
    char_lengths: Annotated[
        AvgMinMax,
        Column("Avg-CharLength(min,max)", justify="right", style="blue"),
    ]
    exec_time: Annotated[
        AvgMinMax,
        Column(
            "Avg-ExecTime(min,max)",
            format=lambda x: x.to_str(2, 2),
            justify="right",
            style="blue",
        ),
    ]
    success_pct: Annotated[
        float,
        Column(
            "Success%",
            concise=True,
            format=lambda x: f"{x*100:.2f}",
            justify="right",
            style="blue",
        ),
    ]
    in_goldilocks_zone: Annotated[
        bool,
        Column(
            "GZ@95%",
            concise=True,
            format=lambda x: "YES" if x else "NO",
            justify="right",
            style="blue",
        ),
    ]
    likelihood_in_goldilocks_zone: Annotated[
        float,
        Column(
            "P(GZ|Data)",
            concise=True,
            format=lambda x: f"{x:.2f}",
            justify="right",
            style="blue",
        ),
    ]


PASS_METRICS_K = [1, 5, 10, 20, 50, 100]


class Metrics(BaseModel):
    num_tests: int = 0
    runs_per_test: int = -1
    total_runs: int = 0
    mean_pass: float = 0.0
    mean_pass_ci_low: float = 0.0
    mean_pass_ci_high: float = 0.0
    unbiased_pass_at_k: list[tuple[int, float]] = Field(default_factory=list)
    unbiased_pass_power_k: list[tuple[int, float]] = Field(default_factory=list)


def pass_at_k_unbiased(n, c, k: int):
    """
    Compute the unbiased estimate of pass@k for a set of test results.

    pass@k is the probability that at least one correct solution is found
    among k randomly selected samples from the results. This metric is commonly
    used in code generation and evaluation tasks to measure the effectiveness
    of a model in producing correct outputs within a limited number of attempts.

    The unbiased estimator corrects for the bias that occurs when the number of
    correct solutions is small compared to the total number of samples. The algorithm
    is based on the formula:
        pass@k = 1 - C(n - c, k) / C(n, k)
    where n is the total number of samples, c is the number of correct samples,
    and C(a, b) is the binomial coefficient "a choose b".

    Args:
        n (int): Total number of runs
        c (int): Number of correct outcomes
        k (int): Number of samples to consider.

    Returns:
        float: Unbiased estimate of pass@k.
    """
    assert k > 0, "pass@k requires k > 0"
    assert c <= n, "pass@k requires c <= n"
    assert k <= n, "pass@k requires k <= n"
    if n - c < k:
        return 1.0
    # 1 - C(n - c, k) / C(n, k)
    denom_range = np.arange(n - c + 1, n + 1, dtype=float)
    return 1.0 - float(np.prod(1.0 - k / denom_range))


def pass_power_k_unbiased(n: int, c: int, k: int):
    """
    Computes the unbiased pass^k metric for a set of test results.

    pass^k is defined as the probability that a test case passes in all of k independent runs,
    assuming unbiased sampling. It is calculated as (mean pass rate) ** k, i.e., the k-th power
    of the fraction of runs that passed.

    This differs from pass@k, which is the probability that at least one of k runs passes.
    pass@k is computed as 1 minus the probability that all k runs fail, whereas pass^k is the
    probability that all k runs succeed.

    Args:
        n (int): Total number of runs
        c (int): Number of correct outcomes
        k (int): Number of independent runs to consider.

    Returns:
        float: The unbiased pass^k metric.
    """
    assert k > 0, "pass^k requires k > 0"
    assert c <= n, "pass^k requires c <= n"
    if n == 0:
        return 0.0
    return (c / n) ** k


def make_per_test_table(
    test_results: dict[str, PerTestCaseResults],
) -> list[PerTestCaseTableRow]:
    table: list[PerTestCaseTableRow] = []
    for uid, tr in test_results.items():
        assert isinstance(tr, PerTestCaseResults)
        passed = sum(tr.test_returned_vals)
        row = PerTestCaseTableRow(
            uid=uid,
            runs=tr.runs,
            passed=passed,
            failed=len(tr.failed_assertions),
            error=len(tr.failed_decodings),
            assistant_turns=AvgMinMax.from_list(tr.assistant_turns),
            user_turns=AvgMinMax.from_list(tr.user_turns),
            tool_turns=AvgMinMax.from_list(tr.tool_turns),
            output_tokens=AvgMinMax.from_list(tr.output_tokens),
            reasoning_tokens=AvgMinMax.from_list(tr.reasoning_tokens),
            char_lengths=AvgMinMax.from_list(tr.content_char_length),
            exec_time=AvgMinMax.from_list(tr.execution_time),
            success_pct=passed / tr.runs if tr.runs else 0,
            in_goldilocks_zone=bool(tr.in_goldilocks_zone),
            likelihood_in_goldilocks_zone=tr.likelihood_in_goldilocks_zone,
        )
        table.append(row)
    return table


def print_per_test_table(table: list[PerTestCaseTableRow], concise: bool = False):
    rich_table = Table(title="Test Results", box=box.DOUBLE)
    columns = PerTestCaseTableRow.columns()
    if concise:
        columns = {name: col for name, col in columns.items() if col.concise}
    for _, col in columns.items():
        rich_table.add_column(col.name, **col.kwargs)
    for row in table:
        row_values = [
            RichText(col.format(getattr(row, name))) for name, col in columns.items()
        ]
        rich_table.add_row(*row_values)
    console = Console()
    console.print(rich_table)


def get_decoded_result_stats(result: DecodeResult) -> dict[str, Any]:
    """
    Get relevant statistics from a decoded result.

    Returns a dictionary with retrieved statistics:
    - turns: total number of turns in the conversation
    - user_turns: number of turns from the user
    - assistant_turns: number of turns from the assistant
    - tool_turns: number of turns from the tool responses
    - test_returned_vals: boolean indicating if the test returned values
    - failed_assertions: string with the line number and content of the failed assertion text
    - execution_time: execution time in seconds from result metadata
    - output_tokens: total number of output tokens generated
    - reasoning_tokens: total number of reasoning tokens generated
    """
    turns = 0
    user_turns = 0
    assistant_turns = 0
    tool_turns = 0
    content_char_length = 0
    output_token_count = 0
    reasoning_token_count = 0
    for msg in result.messages:
        turns += 1
        if isinstance(msg, Text):
            content_char_length += len(msg.content)
            if msg.role == "assistant":
                assistant_turns += 1
            elif msg.role == "user":
                user_turns += 1
        elif isinstance(msg, ToolResponse):
            tool_turns += 1
    ret = {
        "turns": turns,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "tool_turns": tool_turns,
    }

    if result.is_system_error:
        # error during decoding
        try:
            error_message = ErrorInfo(**result.metadata.get("error", {})).message
        except (TypeError, ValueError):
            # keep backward compatibility, let tb agg work on old results for a while
            error_message = str(result.metadata.get("error"))
        ret["failed_decodings"] = error_message
    else:
        # decoding did not fail, check the test result
        tr = result.test_result

        if tr is None:
            # it's none, unexpected since no decoding error
            # treat it like a decoding error, but we don't know the error...
            ret["failed_decodings"] = "No decode error but no test_result, unexpected"
        elif tr.is_system_error:
            # it's a test error (not a test failure)
            ret["failed_decodings"] = (
                f"Test error, caused by line {tr.lineno}:" + f" {tr.line_content}"
            )
        else:
            # test result exists and is not a test error, check the result
            ret["test_returned_vals"] = tr.result
            if not tr.result:
                ret["failed_assertions"] = f"lineno:{tr.lineno}\t{tr.line_content}"
    ret["content_char_length"] = content_char_length
    # Add execution time from result metadata
    ret["execution_time"] = result.metadata.get("execution_time", 0.0)

    # Add tokens
    for usage in result.usage or []:
        output_token_count += usage.output_tokens
        reasoning_token_count += usage.output_tokens_details.reasoning_tokens

    ret["output_tokens"] = output_token_count
    ret["reasoning_tokens"] = reasoning_token_count
    return ret


def get_decoding_error(result: DecodeResult) -> dict | None:
    error_message = result.metadata.get("error")
    if error_message is not None:
        return error_message
    if result.test_result is not None:
        tr = result.test_result
        if tr.is_system_error:
            return f"Test error, caused by line {tr.lineno}: {tr.line_content}"
    return None


def aggregate_results_per_test(
    results: Iterable[DecodeResult],
) -> dict[str, PerTestCaseResults]:
    """
    Aggregates results from multiple decoded results.

    Returns a dictionary with aggregated statistics for each unique test case ID.
    """
    aggregated: dict[str, PerTestCaseResults] = {}
    for r in results:
        uid = r.uid
        if uid not in aggregated:
            aggregated[uid] = PerTestCaseResults()
        aggregated[uid].runs += 1
        stats = get_decoded_result_stats(r)
        for key, value in stats.items():
            getattr(aggregated[uid], key).append(value)

    for uid, tr in aggregated.items():
        k = sum(int(v) for v in tr.test_returned_vals)
        tr.likelihood_in_goldilocks_zone = prob_in_zone(n=tr.runs, k=k)
        tr.in_goldilocks_zone = tr.likelihood_in_goldilocks_zone >= 0.95

    return aggregated


def aggregate_results(results: dict[str, PerTestCaseResults]) -> Metrics:
    out = Metrics()
    ks = []
    pass_at_k_list = {}
    pass_power_k_list = {}
    total_pass = 0
    runs = -1
    for i, r in enumerate(v for _, v in results.items()):
        if i == 0:
            runs = r.runs
            ks = [k for k in PASS_METRICS_K if k <= runs]
            pass_at_k_list = {k: [] for k in ks}
            pass_power_k_list = {k: [] for k in ks}
        if r.runs != runs:
            # do not compute if not all same # runs
            ks.clear()
            pass_at_k_list.clear()
            pass_power_k_list.clear()

        out.num_tests += 1
        out.total_runs += r.runs
        correct_runs = sum(int(v) for v in r.test_returned_vals)
        total_pass += correct_runs
        for k in ks:
            pass_at_k_list[k].append(pass_at_k_unbiased(n=runs, c=correct_runs, k=k))
            pass_power_k_list[k].append(
                pass_power_k_unbiased(n=runs, c=correct_runs, k=k)
            )

    out.runs_per_test = runs
    out.mean_pass = (total_pass / out.total_runs) if (out.total_runs) else 0.0
    if out.total_runs:
        out.mean_pass_ci_low, out.mean_pass_ci_high = cred_int(
            total_pass, out.total_runs, level=0.95, a0=0.5, b0=0.5
        )
    out.unbiased_pass_at_k = [
        (k, safe_mean(v, default=0.0)) for k, v in pass_at_k_list.items()
    ]
    out.unbiased_pass_power_k = [
        (k, safe_mean(v, default=0.0)) for k, v in pass_power_k_list.items()
    ]
    return out


def print_metrics(metrics: Metrics):
    print(f"Number of tests: {metrics.num_tests}")
    runs_per_test_str = (
        str(metrics.runs_per_test) if (metrics.runs_per_test >= 0) else "(multiple)"
    )
    print(f"Runs per test: {runs_per_test_str}")
    print(f"Total runs: {metrics.total_runs}")
    print(
        f"Mean per-sample accuracy: {metrics.mean_pass:.2f} (95% CI: {metrics.mean_pass_ci_low:.2f}-{metrics.mean_pass_ci_high:.2f})"
    )
    if metrics.unbiased_pass_at_k:
        print("Pass@k:")
        for k, v in metrics.unbiased_pass_at_k:
            print(f"  pass@{k}: {v:.2f}")
    if metrics.unbiased_pass_power_k:
        print("Pass^k:")
        for k, v in metrics.unbiased_pass_power_k:
            print(f"  pass^{k}: {v:.2f}")


@click.command()
@click.argument(
    "input_file", required=True, default="-", metavar="INPUT", type=click.File("r")
)
@click.option("--concise", is_flag=True, help="Only print Pass/Fail stats")
@click.option(
    "-f",
    "--output-format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format",
)
def agg(input_file, concise: bool, output_format: str):
    """
    Aggregate metrics from a JSONL file.

    INPUT should be a file in JSONL (multiline) format,
    or '-' to read from standard input.
    """
    # read from stdin, or enter a `with open(...)` context when reading from file
    results = iter_validate_jsonl(input_file, model=DecodeResult)

    per_test_aggregated_results = aggregate_results_per_test(results)
    per_test_table = make_per_test_table(per_test_aggregated_results)
    metrics = aggregate_results(per_test_aggregated_results)

    if output_format == "table":
        if not per_test_table:
            print("No test results to display.")
        else:
            print_per_test_table(per_test_table, concise=concise)
            print_metrics(metrics)
    elif output_format == "json":
        obj = {
            "per_test": [to_jsonable_python(obj) for obj in per_test_table],
            "metrics": to_jsonable_python(metrics),
        }
        json.dump(obj, sys.stdout, indent=2)
        print("")


def main():
    agg()
