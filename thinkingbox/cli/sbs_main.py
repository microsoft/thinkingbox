# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io

import click
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text as RichText

from thinkingbox.cli.agg_main import (
    PerTestCaseResults,
    aggregate_results_per_test,
)
from thinkingbox.common.chat_types import DecodeResult
from thinkingbox.common.eval_utils import cred_int, prob_A_gt_B, prob_lift_gt_eps
from thinkingbox.common.utils import iter_validate_jsonl


def print_results(
    base_agg: dict[str, PerTestCaseResults],
    can_agg: dict[str, PerTestCaseResults],
    pct_lift: float,
):

    table = Table(title="Comparison between base and candidate", box=box.DOUBLE)

    table.add_column("Test Case ID", style="cyan")
    table.add_column(
        "Baseline:\nRun\nSuccess\nFail\nError", justify="left", style="green"
    )
    table.add_column(
        "Candidate:\nRun\nSuccess\nFail\nError", justify="left", style="green"
    )
    table.add_column(
        "Baseline Success%\nActual + CI@95", justify="center", style="blue"
    )
    table.add_column(
        "Candidate Success%\nActual + CI@95", justify="center", style="blue"
    )
    table.add_column("P(pCand > pBase)", justify="right", style="blue")
    table.add_column(
        f"P(pCand - pBase) > {pct_lift*100}%", justify="right", style="blue"
    )

    collected: list[tuple[float, tuple]] = []

    for uid, can_res in can_agg.items():
        base_res = base_agg.get(uid, None)
        if not base_res or (
            len(can_res.failed_decodings) == can_res.runs
            and len(base_res.failed_decodings) == base_res.runs
        ):
            continue
        cand_passed = sum(can_res.test_returned_vals)
        cand_failed = len(can_res.failed_assertions)
        cand_errors = len(can_res.failed_decodings)
        base_passed = sum(base_res.test_returned_vals)
        base_failed = len(base_res.failed_assertions)
        base_errors = len(base_res.failed_decodings)
        kA = cand_passed
        nA = can_res.runs - len(can_res.failed_decodings)
        kB = base_passed
        nB = base_res.runs - len(base_res.failed_decodings)

        if nA <= 0 or nB <= 0:
            prob = 0.0
            pct_lift_gt, se = 0.0, 0.0
            base_succ_low, base_succ_high = 0.0, 0.0
            cand_succ_low, cand_succ_high = 0.0, 0.0
        else:
            prob = prob_A_gt_B(kA, nA, kB, nB)
            pct_lift_gt, se = prob_lift_gt_eps(kA, nA, kB, nB, pct_lift)
            base_succ_low, base_succ_high = cred_int(kB, nB)
            cand_succ_low, cand_succ_high = cred_int(kA, nA)

        row_tuple = (
            uid,
            f"{base_res.runs}/{base_passed}/{base_failed}/{base_errors}",
            f"{can_res.runs}/{cand_passed}/{cand_failed}/{cand_errors}",
            (
                f"{(base_passed / base_res.runs * 100):.2f}%"
                if base_res.runs > 0
                else "0.00%"
                + f" ({base_succ_low*100:.2f}% , {base_succ_high*100:.2f}%)"
            ),
            (
                f"{(cand_passed / can_res.runs * 100):.2f}%"
                if can_res.runs > 0
                else "0.00%"
                + f" ({cand_succ_low*100:.2f}% , {cand_succ_high*100:.2f}%)"
            ),
            f"{prob*100:.2f}%",
            f"{pct_lift_gt*100:.2f}%",
        )
        collected.append((uid, row_tuple))

    # Sort by first descending #default uid
    # TODO make this configurable
    collected.sort(key=lambda x: x[0], reverse=True)

    for _, row_values in collected:
        rich_row = [RichText(str(v)) for v in row_values]
        table.add_row(*rich_row)

    console = Console()
    console.print(table)


@click.command()
@click.option(
    "-b",
    "--baseline",
    type=click.File("r", encoding="utf-8"),
    required=True,
    help="Baseline results in JSONL format.",
)
@click.option(
    "-c",
    "--candidate",
    type=click.File("r", encoding="utf-8"),
    required=True,
    help="Candidate results in JSONL format.",
)
@click.option(
    "--pct-lift",
    type=float,
    default=0.03,
    show_default=True,
    help="Threshold as a fraction (e.g., 0.03 for 3%%) to test if candidate is better than baseline.",
)
def sbs(
    baseline: io.TextIOWrapper, candidate: io.TextIOWrapper, pct_lift: float
) -> None:
    """
    Compare candidate vs baseline JSONL results and report lift statistics.
    """
    base_results = iter_validate_jsonl(baseline, model=DecodeResult)
    can_results = iter_validate_jsonl(candidate, model=DecodeResult)

    base_agg = aggregate_results_per_test(base_results)
    can_agg = aggregate_results_per_test(can_results)

    print_results(
        base_agg=base_agg,
        can_agg=can_agg,
        pct_lift=pct_lift,
    )
