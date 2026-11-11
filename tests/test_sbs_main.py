# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from rich.console import Console

from thinkingbox.cli import sbs_main
from thinkingbox.cli.agg_main import PerTestCaseResults


def _res(passed: int, runs: int) -> PerTestCaseResults:
    return PerTestCaseResults(
        runs=runs,
        test_returned_vals=[True] * passed + [False] * (runs - passed),
    )


def test_sbs_table_renders_confidence_interval(monkeypatch):
    # The success columns advertise "Actual + CI@95"; the interval must render.
    captured = {}

    def fake_console(*a, **k):
        c = Console(record=True, width=240)
        captured["c"] = c
        return c

    monkeypatch.setattr(sbs_main, "Console", fake_console)
    sbs_main.print_results({"t1": _res(2, 4)}, {"t1": _res(3, 4)}, 0.05)
    out = captured["c"].export_text()
    # " , " is the CI interval separator and appears in no column header. The
    # single row renders one interval per success cell, so both the baseline
    # and candidate intervals must be present — not just either one.
    assert out.count(" , ") == 2, out


def test_sbs_valid_side_keeps_interval_when_other_side_all_errors(monkeypatch):
    # One side with no decodable runs must not zero the other side's CI.
    captured = {}

    def fake_console(*a, **k):
        c = Console(record=True, width=240)
        captured["c"] = c
        return c

    all_errors = PerTestCaseResults(
        runs=4,
        test_returned_vals=[],
        failed_decodings=["boom"] * 4,
    )
    monkeypatch.setattr(sbs_main, "Console", fake_console)
    sbs_main.print_results({"t1": _res(3, 4)}, {"t1": all_errors}, 0.05)
    out = captured["c"].export_text()
    # The candidate side (3/4 decodable) still renders a real interval.
    assert out.count(" , ") == 2, out
    intervals = [
        seg for seg in out.split("(")[1:] if " , " in seg.split(")")[0]
    ]
    assert any(
        "0.00% , 0.00%" not in seg.split(")")[0] for seg in intervals
    ), out
