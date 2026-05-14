# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fixture for evaluating generated answers against a reference
E.g., when evaluating in a knowledge-based RAG architecture.

Covers three dimensions:
  1. Semantic correctness  — does the answer match the reference?
  2. Citation accuracy     — does the answer cite the expected document / pages?
  3. Textual quality       — is the answer concise, readable, etc.?

Dimensions 2 and 3 use RubricJudge internally.

By default, semantic mismatch (dimension 1) raises an AssertionError immediately, while dimensions
2 and 3 are evaluated as rubrics and written to metadata for analysis.
This default behavior can be altered by setting ``fail_on_mismatch=False`` (in conftest.yaml or
programmatically when calling the evaluator), in which case all three dimensions are evaluated as
rubrics and the overall reward is a weighted average (50% semantic correctness, 50% other criteria).


Typical conftest.yaml entry::

    fixtures:
      answer_evaluator:
        type: thinkingbox.fixtures.GeneratedAnswerEvaluator
        fail_on_mismatch: true
        max_workers: 4

Typical usage in test case::

def test_foo_bar(
    x: TestContext, judge: Judge, answer_evaluator: GeneratedAnswerEvaluator
):
...
    # Provide reference answer and (optionally) source document + location for citation checks
    answer_evaluator.evaluate(reference="$125", ref_doc="bar.pdf", ref_loc="p3,p32")

Citation location format (ref_loc)
-----------------------------------
See docs/citation_format.md for the full specification. Summary:

  Atoms:
    p<N>              — single PDF page          e.g. p3
    p<N>-<M>          — PDF page range           e.g. p95-100
    r<N>              — single spreadsheet row   e.g. r42
    r<N>-<M>          — spreadsheet row range    e.g. r7-180
    <Sheet>:r<N>      — row on named sheet       e.g. July2024:r42
    <Sheet>:r<N>-<M>  — row range on named sheet e.g. Justin:r47-62
    Multiple_Sheets   — data spans all sheets, no specific location

  Default sheet (no prefix) is the first sheet of the workbook.

  Operators (+ binds tighter than ,):
    +   AND — all locations must be cited   e.g. p3+p8
    ,   OR  — any one alternative suffices  e.g. p3,p8

See docs/generated_answer_evaluator.md for the full reference covering all three evaluation
dimensions, evaluation modes, and the complete citation location format specification.
"""

from __future__ import annotations

import re

from thinkingbox.common.chat_types import TestContext
from thinkingbox.common.config_types import RubricConfig
from thinkingbox.common.judge import Judge, safe_tag_encode
from thinkingbox.common.rubrics_judge import RubricJudge

_RUBRIC_WEIGHT = 1.0

# Patterns for various citation reference location specifiers
_ATOM_PAGE_RANGE = re.compile(r"^p(\d+)-(\d+)$", re.IGNORECASE)
_ATOM_PAGE = re.compile(r"^p(\d+)$", re.IGNORECASE)
_ATOM_ROW_RANGE = re.compile(r"^r(\d+)-(\d+)$", re.IGNORECASE)
_ATOM_ROW = re.compile(r"^r(\d+)$", re.IGNORECASE)
_ATOM_SHEET_ROW_RANGE = re.compile(r"^(.+):r(\d+)-(\d+)$", re.IGNORECASE)
_ATOM_SHEET_ROW = re.compile(r"^(.+):r(\d+)$", re.IGNORECASE)


def _format_atom(token: str) -> str:
    """Convert a single ref_loc atom to a natural language description.

    Handles: p<N>, p<N>-<M>, r<N>, r<N>-<M>, <Sheet>:r<N>, <Sheet>:r<N>-<M>,
    Multiple_Sheets.  Falls back to returning the token as-is for unrecognised
    formats.
    """
    t = token.strip()
    if t == "Multiple_Sheets":
        return "multiple sheets"
    m = _ATOM_PAGE_RANGE.match(t)
    if m:
        return f"pages {m.group(1)} through {m.group(2)}"
    m = _ATOM_PAGE.match(t)
    if m:
        return f"page {m.group(1)}"
    m = _ATOM_SHEET_ROW_RANGE.match(t)
    if m:
        return f"rows {m.group(2)} through {m.group(3)} of sheet '{m.group(1)}'"
    m = _ATOM_SHEET_ROW.match(t)
    if m:
        return f"row {m.group(2)} of sheet '{m.group(1)}'"
    m = _ATOM_ROW_RANGE.match(t)
    if m:
        return f"rows {m.group(1)} through {m.group(2)}"
    m = _ATOM_ROW.match(t)
    if m:
        return f"row {m.group(1)}"
    return t


def _format_location_requirement(ref_loc: str) -> str:
    """Convert a ref_loc expression to a natural language description.

    Format:
      - Comma-separated alternatives — any one is sufficient.
      - Within each alternative, '+'-joined atoms — all must be cited.

    Examples:
      "p3"              → "page 3"
      "p3-5"            → "pages 3 through 5"
      "p3+p8"           → "both page 3 and page 8"
      "p3,p8"           → "page 3, or page 8"
      "p1+p2,p3+p4"     → "both page 1 and page 2, or both page 3 and page 4"
      "p3+p8+p12"       → "page 3, page 8, and page 12"
      "r7-180"          → "rows 7 through 180"
      "Justin:r47-62"   → "rows 47 through 62 of sheet 'Justin'"
      "Multiple_Sheets" → "multiple sheets"

    TODO Support cross-document citations — e.g. p3 of A.pdf OR p6 of B.pdf
    """
    options = [opt.strip() for opt in ref_loc.split(",")]
    described: list[str] = []
    for opt in options:
        atoms = [_format_atom(a) for a in opt.split("+")]
        if len(atoms) == 1:
            described.append(atoms[0])
        elif len(atoms) == 2:
            described.append("both " + " and ".join(atoms))
        else:
            described.append(", ".join(atoms[:-1]) + ", and " + atoms[-1])
    return ", or ".join(described)


# ---------------------------------------------------------------------------
# Deprecated aliases — kept for backward compatibility
# ---------------------------------------------------------------------------


def _page_label(p: str) -> str:
    """Deprecated: use _format_atom instead."""
    return _format_atom(p)


def _format_page_requirement(ref_pages: str) -> str:
    """Deprecated: use _format_location_requirement instead."""
    return _format_location_requirement(ref_pages)


_SYSTEM_COMPARE_TO_REFERENCE = """\
You are a judge, your task is to read the contents of a reference answer and a candidate answer, and determine if the candidate answer communicates the full and complete semantics of the reference.
You will be presented with the query (for context) and the two answers as follows:
<query>
Content of the query
</query>
<reference>
content of the reference answer (usually short)
</reference>
<candidate>
content of the candidate answer (often longer than the reference)
</candidate>
Does the candidate answer communicate the full and complete semantics of the reference?
You may consider the content of the query for context on the meaning of each answer.
Presence or absence of formatting in numeric answers should not impact your judgement.
Addition of currency markers in generated answers are allowed (if the units are reasonable for the query).
If the reference answer contains a currency marker, the candidate answer should *also* contain the appropriate currency marker to be considered a match.
Answer with a single word: Yes or No."""

_USER_COMPARE_TO_REFERENCE = """\
<query>
{query}
</query>
<reference>
{reference}
</reference>
<candidate>
{candidate}
</candidate>
"""


class GeneratedAnswerEvaluator:
    """Evaluate a generated answer on correctness, citations, and textual quality.

    ``judge`` and ``x`` are injected automatically by the fixture system.

    Args:
        judge: Active Judge instance (auto-injected).
        x: TestContext for the current test run (auto-injected).
        fail_on_mismatch: If True, raise AssertionError when the semantic
            reference check fails. Can be overridden per call.
        rubrics_threshold: Minimum rubric reward required to pass (0.0–1.0).
            Raises AssertionError if the reward falls below this value.
            Set to 0.0 to record rubric scores without enforcing a threshold.
            Can be overridden per call.
        max_workers: Thread-pool size passed to the internal RubricJudge.
    """

    def __init__(
        self,
        judge: Judge,
        x: TestContext,
        fail_on_mismatch: bool = True,
        rubrics_threshold: float = 0.7,
        max_workers: int = 2,
    ):
        self._judge = judge
        self._x = x
        self._fail_on_mismatch = fail_on_mismatch
        self._rubrics_threshold = rubrics_threshold
        self._rubric_judge = RubricJudge(judge=judge, x=x, max_workers=max_workers)

    def candidate_answer_matches_reference(
        self,
        query: str,
        reference: str,
        candidate: str,
        fail_on_mismatch: bool = True,
    ) -> bool:
        """Return True if a candidate answer is semantically equivalent to the reference.

        Args:
            query: The original user query.
            reference: The reference answer (usually short).
            candidate: The candidate answer (often longer than the reference).
            fail_on_mismatch: If True, raise AssertionError when the candidate
                does not match the reference.
        """
        user = _USER_COMPARE_TO_REFERENCE.format(
            query=safe_tag_encode(query),
            reference=safe_tag_encode(reference),
            candidate=safe_tag_encode(candidate),
        )
        match = self._judge.evaluate_bool(_SYSTEM_COMPARE_TO_REFERENCE, user)
        if fail_on_mismatch and not match:
            max_len = 200
            cand_display = (
                candidate if len(candidate) <= max_len else candidate[:max_len] + "..."
            )
            ref_display = (
                reference if len(reference) <= max_len else reference[:max_len] + "..."
            )
            raise AssertionError(
                f"Answer Mismatch:\n{cand_display}\ndoes not match reference:\n{ref_display}"
            )
        return match

    def evaluate(
        self,
        reference: str,
        ref_doc: str | None = None,
        ref_loc: str | None = None,
        fail_on_mismatch: bool | None = None,
        rubrics_threshold: float | None = None,
        *,
        ref_pages: str | None = None,
    ) -> bool:
        """Evaluate the response stored in ``x.response``.

        When ``fail_on_mismatch`` is True, semantic correctness is checked
        first via :meth:`candidate_answer_matches_reference` and an
        AssertionError is raised immediately on mismatch.  Citation fidelity
        and textual quality rubrics are then run and written to metadata.

        When ``fail_on_mismatch`` is False, semantic correctness is included
        as a rubric criterion weighted at 50% (equal to the combined weight of
        all other criteria).  Rubric scores and reward are written to metadata.

        Args:
            reference: Expected answer text (usually short / atomic).
            ref_doc: Expected source document filename, if known.
            ref_loc: Expected citation location (e.g. ``"p3"``, ``"r7-180"``,
                ``"Justin:r47-62"``).  See docs/citation_format.md for the
                full format specification.
            fail_on_mismatch: Override the instance-level ``fail_on_mismatch``
                for the semantic check in this call only.
            rubrics_threshold: Override the instance-level ``rubrics_threshold``
                for the rubric check in this call only. Set to 0.0 to disable
                threshold enforcement for this call.
            ref_pages: Deprecated alias for ``ref_loc``.  Use ``ref_loc``
                instead.  Ignored when ``ref_loc`` is also supplied.

        Returns:
            True if the semantic correctness check passes.
        """
        if ref_pages is not None and ref_loc is None:
            ref_loc = ref_pages
        fom = (
            fail_on_mismatch if fail_on_mismatch is not None else self._fail_on_mismatch
        )
        thr = (
            rubrics_threshold
            if rubrics_threshold is not None
            else self._rubrics_threshold
        )
        response = self._x.response
        query = self._x.query()

        # Build citation fidelity + textual quality rubrics (always present)
        rubrics: list[RubricConfig] = []

        if ref_doc is not None:
            rubrics.append(
                RubricConfig(
                    criterion=(
                        f"Does the response cite, mention, or reference the source document"
                        f" '{ref_doc}'?"
                    ),
                    weight=_RUBRIC_WEIGHT,
                )
            )

        if ref_loc is not None:
            loc_req = _format_location_requirement(ref_loc)
            rubrics.append(
                RubricConfig(
                    criterion=(
                        f"Does the response indicate that the information comes from"
                        f" {loc_req} of the source document?"
                    ),
                    weight=_RUBRIC_WEIGHT,
                )
            )

        rubrics.append(
            RubricConfig(
                criterion=(
                    "Is the response written in clear, readable language that directly"
                    " addresses the query without being confusing or hard to follow?"
                ),
                weight=_RUBRIC_WEIGHT,
            )
        )

        rubrics.append(
            RubricConfig(
                criterion=(
                    "Is the response concise — does it avoid unnecessary verbosity,"
                    " filler phrases, or repetition while remaining complete?"
                ),
                weight=_RUBRIC_WEIGHT,
            )
        )

        if fom:
            # Semantic check may throw before we reach the rubric evaluation
            match = self.candidate_answer_matches_reference(
                query, reference, response, fail_on_mismatch=True
            )
            self._rubric_judge.evaluate(
                response, rubrics, global_threshold=thr, throw_on_failure=True
            )
            return match
        else:
            # Fold correctness into the rubric at 50% weight so that the overall
            # rubric reward is 50% correctness + 50% everything else.
            other_weight = sum(r.weight for r in rubrics)
            rubrics.insert(
                0,
                RubricConfig(
                    criterion=(
                        f'For the query: "{query}"\n'
                        f'The reference answer is: "{reference}"\n'
                        f"Does the response communicate the full and complete semantics"
                        f" of the reference answer?"
                    ),
                    weight=other_weight,
                ),
            )
            result = self._rubric_judge.evaluate(
                response, rubrics, global_threshold=thr, throw_on_failure=True
            )
            # scores list preserves insertion order — index 0 is correctness
            return bool(result.scores[0].rating)
