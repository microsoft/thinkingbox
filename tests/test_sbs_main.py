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
    # " , " is the CI interval separator and appears in no column header.
    assert " , " in out
