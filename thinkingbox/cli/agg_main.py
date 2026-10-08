import json
import sys
from typing import Annotated, Any, Callable, Iterable

import click
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
from thinkingbox.common.eval_utils import prob_in_zone
from thinkingbox.common.utils import ErrorInfo, iter_validate_jsonl
from thinkingbox.cli.stats import (
    mean_and_se_of_mean,
    pass_at_k,
    pass_at_k_exact_se,
    pass_power_k,
    pass_power_k_exact_se,
    pass_rate_and_se,
)


def format_result(value: float) -> str:
    """Format a statistical result with at most four significant digits."""
    return f"{value:.4g}"


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
        self, name: str, details: bool = False, format: Callable | None = None, **kwargs
    ):
        self.name = name
        self.details = details
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
    uid: Annotated[str, Column("Test Case ID", style="cyan")]
    runs: Annotated[int, Column("Runs", justify="right", style="green")]
    passed: Annotated[int, Column("Pass", justify="right", style="green")]
    failed: Annotated[int, Column("Fail", justify="right", style="red")]
    error: Annotated[int, Column("Error", justify="right", style="red")]
    assistant_turns: Annotated[
        AvgMinMax,
        Column("Avg-Ast(min,max)", details=True, justify="right", style="blue"),
    ]
    user_turns: Annotated[
        AvgMinMax,
        Column("Avg-Usr(min,max)", details=True, justify="right", style="blue"),
    ]
    tool_turns: Annotated[
        AvgMinMax,
        Column("Avg-TC(min,max)", details=True, justify="right", style="blue"),
    ]
    output_tokens: Annotated[
        AvgMinMax,
        Column("Avg-OutTkns(min,max)", details=True, justify="right", style="blue"),
    ]
    reasoning_tokens: Annotated[
        AvgMinMax,
        Column("Avg-ReaTkns(min,max)", details=True, justify="right", style="blue"),
    ]
    char_lengths: Annotated[
        AvgMinMax,
        Column("Avg-CharLength(min,max)", details=True, justify="right", style="blue"),
    ]
    exec_time: Annotated[
        AvgMinMax,
        Column(
            "Avg-ExecTime(min,max)",
            details=True,
            format=lambda x: x.to_str(2, 2),
            justify="right",
            style="blue",
        ),
    ]
    success_pct: Annotated[
        float,
        Column(
            "Success%",
            format=lambda x: format_result(x * 100),
            justify="right",
            style="blue",
        ),
    ]
    success_pct_ste: Annotated[
        float,
        Column(
            "Success% SE",
            format=lambda x: format_result(x * 100),
            justify="right",
            style="blue",
        ),
    ]
    pass_at_n: Annotated[
        float,
        Column("Pass@N", format=format_result, justify="right", style="blue"),
    ]
    pass_at_n_ste: Annotated[
        float,
        Column("Pass@N SE", format=format_result, justify="right", style="blue"),
    ]
    pass_power_n: Annotated[
        float,
        Column("Pass^N", format=format_result, justify="right", style="blue"),
    ]
    pass_power_n_ste: Annotated[
        float,
        Column("Pass^N SE", format=format_result, justify="right", style="blue"),
    ]
    in_goldilocks_zone: Annotated[
        bool,
        Column(
            "GZ@95%",
            format=lambda x: "YES" if x else "NO",
            justify="right",
            style="blue",
        ),
    ]
    likelihood_in_goldilocks_zone: Annotated[
        float,
        Column(
            "P(GZ|Data)",
            format=format_result,
            justify="right",
            style="blue",
        ),
    ]


PASS_METRICS_K = [1, 5, 10, 20, 50, 100]


class Metrics(BaseModel):
    num_datasets: int = 0
    num_tests: int = 0
    runs_per_test: int = -1
    total_runs: int = 0
    mean_pass: float = 0.0
    mean_pass_ste: float = 0.0
    pass_at_k: list["KMetricSummary"] = Field(default_factory=list)
    pass_power_k: list["KMetricSummary"] = Field(default_factory=list)


class MetricEstimate(BaseModel):
    estimate: float
    ste: float
    count: int


class KMetricSummary(MetricEstimate):
    k: int


class InstanceMetrics(BaseModel):
    uid: str
    dataset: str
    runs: int
    successes: int
    pass_rate: MetricEstimate
    pass_at_n: MetricEstimate
    pass_power_n: MetricEstimate
    pass_at_k: list[KMetricSummary]
    pass_power_k: list[KMetricSummary]


class DatasetMetrics(BaseModel):
    dataset: str
    num_tests: int
    total_runs: int
    pass_rate: MetricEstimate
    pass_at_k: list[KMetricSummary]
    pass_power_k: list[KMetricSummary]


class BenchmarkMetrics(DatasetMetrics):
    source_datasets: list[str]


def make_per_test_table(
    test_results: dict[str, PerTestCaseResults],
    instance_metrics: list[InstanceMetrics],
) -> list[PerTestCaseTableRow]:
    metrics_by_uid = {metric.uid: metric for metric in instance_metrics}
    table: list[PerTestCaseTableRow] = []
    for uid, tr in test_results.items():
        assert isinstance(tr, PerTestCaseResults)
        metric = metrics_by_uid[uid]
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
            success_pct=metric.pass_rate.estimate,
            success_pct_ste=metric.pass_rate.ste,
            pass_at_n=metric.pass_at_n.estimate,
            pass_at_n_ste=metric.pass_at_n.ste,
            pass_power_n=metric.pass_power_n.estimate,
            pass_power_n_ste=metric.pass_power_n.ste,
            in_goldilocks_zone=bool(tr.in_goldilocks_zone),
            likelihood_in_goldilocks_zone=tr.likelihood_in_goldilocks_zone,
        )
        table.append(row)
    return table


def print_per_test_table(
    table: list[PerTestCaseTableRow], show_details: bool = False
) -> None:
    rich_table = Table(title="Test Results", box=box.DOUBLE)
    columns = PerTestCaseTableRow.columns()
    if not show_details:
        columns = {name: col for name, col in columns.items() if not col.details}
    for _, col in columns.items():
        rich_table.add_column(col.name, **col.kwargs)
    for row in table:
        row_values = [
            RichText(col.format(getattr(row, name))) for name, col in columns.items()
        ]
        rich_table.add_row(*row_values)
    console = Console()
    console.print(rich_table)


def _format_metric_estimates(metrics: list[KMetricSummary], label: str) -> str:
    return "\n".join(
        f"{label}{metric.k}: {format_result(metric.estimate)} +/- "
        f"{format_result(metric.ste)}"
        for metric in metrics
    )


def print_dataset_metrics_table(datasets: list[DatasetMetrics]) -> None:
    table = Table(title="Dataset Metrics", box=box.DOUBLE)
    table.add_column("Dataset", style="cyan")
    table.add_column("Instances", justify="right", style="green")
    table.add_column("Runs", justify="right", style="green")
    table.add_column("Pass Rate +/- SE", justify="right", style="blue")
    table.add_column("Pass@k +/- SE", justify="right", style="blue")
    table.add_column("Pass^k +/- SE", justify="right", style="blue")

    for dataset in datasets:
        table.add_row(
            dataset.dataset,
            str(dataset.num_tests),
            str(dataset.total_runs),
            f"{format_result(dataset.pass_rate.estimate)} +/- "
            f"{format_result(dataset.pass_rate.ste)}",
            _format_metric_estimates(dataset.pass_at_k, "pass@"),
            _format_metric_estimates(dataset.pass_power_k, "pass^"),
        )

    Console().print(table)


def print_benchmark_metrics_table(benchmarks: list[BenchmarkMetrics]) -> None:
    table = Table(title="Benchmark Metrics", box=box.DOUBLE)
    table.add_column("Benchmark", style="cyan")
    table.add_column("Source Datasets", style="cyan")
    table.add_column("Instances", justify="right", style="green")
    table.add_column("Runs", justify="right", style="green")
    table.add_column("Pass Rate +/- SE", justify="right", style="blue")
    table.add_column("Pass@k +/- SE", justify="right", style="blue")
    table.add_column("Pass^k +/- SE", justify="right", style="blue")

    for benchmark in benchmarks:
        table.add_row(
            benchmark.dataset,
            "\n".join(benchmark.source_datasets),
            str(benchmark.num_tests),
            str(benchmark.total_runs),
            f"{format_result(benchmark.pass_rate.estimate)} +/- "
            f"{format_result(benchmark.pass_rate.ste)}",
            _format_metric_estimates(benchmark.pass_at_k, "pass@"),
            _format_metric_estimates(benchmark.pass_power_k, "pass^"),
        )

    Console().print(table)


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


def _metric_estimate(estimates: list[tuple[float, float]]) -> MetricEstimate:
    estimate, ste, count = mean_and_se_of_mean(estimates)
    return MetricEstimate(estimate=estimate, ste=ste, count=count)


def _parse_dataset(uid: str) -> str:
    dataset, separator, instance = uid.partition(":")
    if not separator or not dataset or not instance:
        raise ValueError(
            f"Expected UID in 'dataset.py:instance_name' format, got {uid!r}"
        )
    return dataset


def build_instance_metrics(
    results: dict[str, PerTestCaseResults],
) -> tuple[list[InstanceMetrics], int]:
    if not results:
        return [], 0

    runs_per_test = {result.runs for result in results.values()}
    if len(runs_per_test) != 1:
        raise ValueError(
            "All instances must have the same number of samples; "
            f"found {sorted(runs_per_test)}"
        )
    runs = runs_per_test.pop()
    ks = [k for k in PASS_METRICS_K if k <= runs]
    instance_metrics: list[InstanceMetrics] = []

    for uid, result in results.items():
        successes = sum(int(value) for value in result.test_returned_vals)
        pass_rate, pass_rate_ste = pass_rate_and_se(runs, successes)
        pass_at_n = MetricEstimate(
            estimate=pass_at_k(runs, successes, runs),
            ste=pass_at_k_exact_se(runs, successes, runs),
            count=1,
        )
        pass_power_n = MetricEstimate(
            estimate=pass_power_k(runs, successes, runs),
            ste=pass_power_k_exact_se(runs, successes, runs),
            count=1,
        )
        instance_metrics.append(
            InstanceMetrics(
                uid=uid,
                dataset=_parse_dataset(uid),
                runs=runs,
                successes=successes,
                pass_rate=MetricEstimate(
                    estimate=pass_rate,
                    ste=pass_rate_ste,
                    count=1,
                ),
                pass_at_n=pass_at_n,
                pass_power_n=pass_power_n,
                pass_at_k=[
                    KMetricSummary(
                        k=k,
                        estimate=pass_at_k(runs, successes, k),
                        ste=pass_at_k_exact_se(runs, successes, k),
                        count=1,
                    )
                    for k in ks
                ],
                pass_power_k=[
                    KMetricSummary(
                        k=k,
                        estimate=pass_power_k(runs, successes, k),
                        ste=pass_power_k_exact_se(runs, successes, k),
                        count=1,
                    )
                    for k in ks
                ],
            )
        )

    return instance_metrics, runs


def _summarize_k_metrics(
    instances: list[InstanceMetrics] | list[DatasetMetrics],
    metric_name: str,
) -> list[KMetricSummary]:
    metrics_by_k: dict[int, list[tuple[float, float]]] = {}
    for instance in instances:
        for metric in getattr(instance, metric_name):
            metrics_by_k.setdefault(metric.k, []).append((metric.estimate, metric.ste))
    return [
        KMetricSummary(k=k, **_metric_estimate(estimates).model_dump())
        for k, estimates in sorted(metrics_by_k.items())
    ]


def aggregate_dataset_metrics(
    instance_metrics: list[InstanceMetrics],
) -> list[DatasetMetrics]:
    grouped: dict[str, list[InstanceMetrics]] = {}
    for metric in instance_metrics:
        grouped.setdefault(metric.dataset, []).append(metric)

    return [
        _aggregate_metric_group(dataset, instances)
        for dataset, instances in sorted(grouped.items())
    ]


def _aggregate_metric_group(
    dataset: str, instances: list[InstanceMetrics]
) -> DatasetMetrics:
    return DatasetMetrics(
        dataset=dataset,
        num_tests=len(instances),
        total_runs=sum(instance.runs for instance in instances),
        pass_rate=_metric_estimate(
            [
                (instance.pass_rate.estimate, instance.pass_rate.ste)
                for instance in instances
            ]
        ),
        pass_at_k=_summarize_k_metrics(instances, "pass_at_k"),
        pass_power_k=_summarize_k_metrics(instances, "pass_power_k"),
    )


def _benchmark_name(dataset: str, suffixes: tuple[str, ...]) -> str:
    stem, separator, extension = dataset.rpartition(".")
    if not separator:
        return dataset
    for suffix in suffixes:
        if stem.endswith(suffix):
            return f"{stem.removesuffix(suffix)}.{extension}"
    return dataset


def aggregate_benchmark_metrics(
    instance_metrics: list[InstanceMetrics],
    aggregate_dataset_suffixes: tuple[str, ...],
) -> list[BenchmarkMetrics]:
    """Aggregate split dataset files into benchmark totals for matching suffixes."""
    if not aggregate_dataset_suffixes:
        return []
    if any(not suffix for suffix in aggregate_dataset_suffixes):
        raise ValueError("Dataset aggregation suffixes must not be empty")

    normalized_suffixes = tuple(
        sorted(set(aggregate_dataset_suffixes), key=len, reverse=True)
    )
    grouped: dict[str, list[InstanceMetrics]] = {}
    source_datasets: dict[str, set[str]] = {}
    for metric in instance_metrics:
        benchmark = _benchmark_name(metric.dataset, normalized_suffixes)
        grouped.setdefault(benchmark, []).append(metric)
        source_datasets.setdefault(benchmark, set()).add(metric.dataset)

    benchmarks: list[BenchmarkMetrics] = []
    for benchmark, instances in sorted(grouped.items()):
        sources = sorted(source_datasets[benchmark])
        if len(sources) < 2:
            continue
        summary = _aggregate_metric_group(benchmark, instances)
        benchmarks.append(
            BenchmarkMetrics(
                **summary.model_dump(),
                source_datasets=sources,
            )
        )
    return benchmarks


def aggregate_results(
    instance_metrics: list[InstanceMetrics],
    dataset_metrics: list[DatasetMetrics],
    runs_per_test: int,
) -> Metrics:
    if not instance_metrics:
        return Metrics()

    pass_rate = _metric_estimate(
        [
            (metric.pass_rate.estimate, metric.pass_rate.ste)
            for metric in instance_metrics
        ]
    )
    return Metrics(
        num_datasets=len(dataset_metrics),
        num_tests=len(instance_metrics),
        runs_per_test=runs_per_test,
        total_runs=sum(metric.runs for metric in instance_metrics),
        mean_pass=pass_rate.estimate,
        mean_pass_ste=pass_rate.ste,
        pass_at_k=_summarize_k_metrics(instance_metrics, "pass_at_k"),
        pass_power_k=_summarize_k_metrics(instance_metrics, "pass_power_k"),
    )


def print_metrics(metrics: Metrics) -> None:
    print(f"Number of datasets: {metrics.num_datasets}")
    print(f"Number of tests: {metrics.num_tests}")
    print(f"Samples per instance: {metrics.runs_per_test}")
    print(f"Total runs: {metrics.total_runs}")
    print(
        "Mean per-instance pass rate: "
        f"{format_result(metrics.mean_pass)} +/- "
        f"{format_result(metrics.mean_pass_ste)}"
    )
    if metrics.pass_at_k:
        print("Pass@k:")
        for metric in metrics.pass_at_k:
            print(
                f"  pass@{metric.k}: {format_result(metric.estimate)} +/- "
                f"{format_result(metric.ste)}"
            )
    if metrics.pass_power_k:
        print("Pass^k:")
        for metric in metrics.pass_power_k:
            print(
                f"  pass^{metric.k}: {format_result(metric.estimate)} +/- "
                f"{format_result(metric.ste)}"
            )


@click.command()
@click.argument(
    "input_file", required=True, default="-", metavar="INPUT", type=click.File("r")
)
@click.option(
    "--show-details",
    is_flag=True,
    help="Show per-test operational averages and ranges.",
)
@click.option(
    "--aggregate-dataset-suffix",
    "aggregate_dataset_suffixes",
    multiple=True,
    help=(
        "Merge datasets whose filename stems end with this suffix into a "
        "benchmark-total row. May be given multiple times."
    ),
)
@click.option(
    "-f",
    "--output-format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format",
)
def agg(
    input_file,
    show_details: bool,
    aggregate_dataset_suffixes: tuple[str, ...],
    output_format: str,
):
    """
    Aggregate metrics from a JSONL file.

    INPUT should be a file in JSONL (multiline) format,
    or '-' to read from standard input.
    """
    # read from stdin, or enter a `with open(...)` context when reading from file
    results = iter_validate_jsonl(input_file, model=DecodeResult)

    per_test_aggregated_results = aggregate_results_per_test(results)
    instance_metrics, runs_per_test = build_instance_metrics(
        per_test_aggregated_results
    )
    per_test_table = make_per_test_table(per_test_aggregated_results, instance_metrics)
    dataset_metrics = aggregate_dataset_metrics(instance_metrics)
    benchmark_metrics = aggregate_benchmark_metrics(
        instance_metrics, aggregate_dataset_suffixes
    )
    metrics = aggregate_results(instance_metrics, dataset_metrics, runs_per_test)

    if output_format == "table":
        if not per_test_table:
            print("No test results to display.")
        else:
            print_per_test_table(per_test_table, show_details=show_details)
            print_dataset_metrics_table(dataset_metrics)
            if benchmark_metrics:
                print_benchmark_metrics_table(benchmark_metrics)
            print_metrics(metrics)
    elif output_format == "json":
        obj = {
            "per_test": [to_jsonable_python(obj) for obj in instance_metrics],
            "per_dataset": [to_jsonable_python(obj) for obj in dataset_metrics],
            "per_benchmark": [to_jsonable_python(obj) for obj in benchmark_metrics],
            "metrics": to_jsonable_python(metrics),
        }
        json.dump(obj, sys.stdout, indent=2)
        print("")


def main():
    agg()
