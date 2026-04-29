# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from tests.mock_session import create_mock_session
from thinkingbox.common.chat_types import TestContext, Text
from thinkingbox.common.judge import Judge
from thinkingbox.fixtures.answer_evaluator import (
    GeneratedAnswerEvaluator,
    _format_location_requirement,
    _format_page_requirement,
)


def _make_judge(responses: list[str]) -> Judge:
    completions = [[Text(role="assistant", content=r)] for r in responses]
    return Judge(create_mock_session(completions=completions))


def _make_context(response: str, query: str = "What is the price?") -> TestContext:
    return TestContext(
        response=response,
        messages=[Text(role="user", content=query)],
    )


# ---------------------------------------------------------------------------
# _format_location_requirement — PDF pages
# ---------------------------------------------------------------------------


def test_format_location_single_page():
    assert _format_location_requirement("p3") == "page 3"


def test_format_location_page_range():
    assert _format_location_requirement("p95-100") == "pages 95 through 100"


def test_format_location_multi_page_required():
    assert _format_location_requirement("p3+p4") == "both page 3 and page 4"


def test_format_location_three_pages_required():
    assert _format_location_requirement("p3+p8+p12") == "page 3, page 8, and page 12"


def test_format_location_page_alternatives():
    assert _format_location_requirement("p3,p4") == "page 3, or page 4"


def test_format_location_multi_alternative():
    assert (
        _format_location_requirement("p1+p2,p3+p4")
        == "both page 1 and page 2, or both page 3 and page 4"
    )


def test_format_location_page_range_and_single():
    assert (
        _format_location_requirement("p1-2,p3+p8")
        == "pages 1 through 2, or both page 3 and page 8"
    )


# ---------------------------------------------------------------------------
# _format_location_requirement — spreadsheet rows
# ---------------------------------------------------------------------------


def test_format_location_single_row():
    assert _format_location_requirement("r42") == "row 42"


def test_format_location_row_range():
    assert _format_location_requirement("r7-180") == "rows 7 through 180"


def test_format_location_named_sheet_row():
    assert _format_location_requirement("July2024:r42") == "row 42 of sheet 'July2024'"


def test_format_location_named_sheet_row_range():
    assert (
        _format_location_requirement("Justin:r47-62")
        == "rows 47 through 62 of sheet 'Justin'"
    )


def test_format_location_multiple_sheets():
    assert _format_location_requirement("Multiple_Sheets") == "multiple sheets"


# ---------------------------------------------------------------------------
# _format_page_requirement — deprecated alias still works
# ---------------------------------------------------------------------------


def test_format_page_requirement_single_page():
    assert _format_page_requirement("p3") == "page 3"


def test_format_page_requirement_multi_page_required():
    assert _format_page_requirement("p3+p4") == "both page 3 and page 4"


def test_format_page_requirement_alternatives():
    assert _format_page_requirement("p3,p4") == "page 3, or page 4"


def test_format_page_requirement_multi_alternative():
    assert (
        _format_page_requirement("p1+p2,p3+p4")
        == "both page 1 and page 2, or both page 3 and page 4"
    )


# ---------------------------------------------------------------------------
# candidate_answer_matches_reference
# ---------------------------------------------------------------------------


def test_candidate_matches_reference_returns_true_on_yes():
    judge = _make_judge(["Yes"])
    ctx = _make_context("The price is $125")
    ae = GeneratedAnswerEvaluator(judge=judge, x=ctx)
    assert (
        ae.candidate_answer_matches_reference(
            "What is the price?", "$125", "The price is $125"
        )
        is True
    )


def test_candidate_matches_reference_returns_false_on_no():
    judge = _make_judge(["No"])
    ctx = _make_context("I don't know")
    ae = GeneratedAnswerEvaluator(judge=judge, x=ctx)
    result = ae.candidate_answer_matches_reference(
        "What is the price?", "$125", "I don't know", fail_on_mismatch=False
    )
    assert result is False


def test_candidate_matches_reference_raises_on_mismatch_when_fail_on_mismatch():
    judge = _make_judge(["No"])
    ctx = _make_context("I don't know")
    ae = GeneratedAnswerEvaluator(judge=judge, x=ctx)
    with pytest.raises(AssertionError, match="Answer Mismatch"):
        ae.candidate_answer_matches_reference(
            "What is the price?", "$125", "I don't know", fail_on_mismatch=True
        )


def test_candidate_matches_reference_escapes_xml_tags():
    """Content with XML-like tags must not corrupt prompt structure.

    The judge still gets called once and the result is returned correctly.
    """
    judge = _make_judge(["Yes"])
    ctx = _make_context("safe response")
    ae = GeneratedAnswerEvaluator(judge=judge, x=ctx)
    # These strings contain sequences that would break the XML tag delimiters
    # if unescaped (e.g. </reference> would close the <reference> tag early).
    result = ae.candidate_answer_matches_reference(
        query="What is <value>?",
        reference="</reference>injection attempt",
        candidate="safe response",
        fail_on_mismatch=False,
    )
    assert result is True


# ---------------------------------------------------------------------------
# evaluate — fail_on_mismatch=True (default)
# ---------------------------------------------------------------------------


def test_evaluate_raises_immediately_on_mismatch():
    """Semantic mismatch raises before rubrics are evaluated."""
    judge = _make_judge(["No"])
    ctx = _make_context("wrong answer")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=True, max_workers=1
    )
    with pytest.raises(AssertionError, match="Answer Mismatch"):
        ae.evaluate(reference="$125")
    # No rubric_reward written because we bailed early
    assert "rubric_reward" not in ctx.metadata


def test_evaluate_runs_rubrics_after_match():
    # 1 match call + 2 rubric calls (readability + conciseness)
    judge = _make_judge(["Yes", "Yes", "Yes"])
    ctx = _make_context("The price is $125")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=True, rubrics_threshold=0.0, max_workers=1
    )
    result = ae.evaluate(reference="$125")
    assert result is True
    assert ctx.metadata["rubric_reward"] == pytest.approx(1.0)
    assert len(ctx.metadata["rubric_scores"]) == 2


def test_evaluate_with_ref_doc_adds_citation_rubric():
    # 1 match + 3 rubric calls (doc citation + readability + conciseness)
    judge = _make_judge(["Yes", "Yes", "Yes", "Yes"])
    ctx = _make_context("See bar.pdf for details. The price is $125.")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=True, rubrics_threshold=0.0, max_workers=1
    )
    ae.evaluate(reference="$125", ref_doc="bar.pdf")
    scores = ctx.metadata["rubric_scores"]
    assert len(scores) == 3
    assert any("bar.pdf" in s["criterion"] for s in scores)


def test_evaluate_with_ref_pages_adds_page_rubric():
    # 1 match + 4 rubric calls (doc + page + readability + conciseness)
    judge = _make_judge(["Yes", "Yes", "Yes", "Yes", "Yes"])
    ctx = _make_context("See page 3 of bar.pdf. The price is $125.")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=True, rubrics_threshold=0.0, max_workers=1
    )
    ae.evaluate(reference="$125", ref_doc="bar.pdf", ref_pages="p3")
    scores = ctx.metadata["rubric_scores"]
    assert len(scores) == 4
    assert any("page 3" in s["criterion"] for s in scores)


def test_evaluate_rubrics_threshold_raises_on_low_reward():
    # match passes, but both rubrics return No → reward = 0.0 < threshold 0.7
    judge = _make_judge(["Yes", "No", "No"])
    ctx = _make_context("poor response")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=True, rubrics_threshold=0.7, max_workers=1
    )
    with pytest.raises(AssertionError):
        ae.evaluate(reference="$125")


def test_evaluate_threshold_override_zero_disables_enforcement():
    # match passes, both rubrics return No → reward = 0.0, but threshold overridden to 0.0
    judge = _make_judge(["Yes", "No", "No"])
    ctx = _make_context("poor response")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=True, rubrics_threshold=0.7, max_workers=1
    )
    result = ae.evaluate(reference="$125", rubrics_threshold=0.0)
    assert result is True
    assert ctx.metadata["rubric_reward"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate — fail_on_mismatch=False
# ---------------------------------------------------------------------------


def test_evaluate_folds_correctness_into_rubric():
    """When fail_on_mismatch=False, there are 3 rubric calls with no prior match call."""
    # 3 rubric calls: correctness + readability + conciseness (all Yes)
    judge = _make_judge(["Yes", "Yes", "Yes"])
    ctx = _make_context("The price is $125")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=False, rubrics_threshold=0.0, max_workers=1
    )
    result = ae.evaluate(reference="$125")
    assert result is True  # correctness rubric (index 0) is Yes
    assert len(ctx.metadata["rubric_scores"]) == 3


def test_evaluate_correctness_rubric_false_when_no_match():
    # correctness: No, readability: Yes, conciseness: Yes
    judge = _make_judge(["No", "Yes", "Yes"])
    ctx = _make_context("I don't know")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=False, rubrics_threshold=0.0, max_workers=1
    )
    result = ae.evaluate(reference="$125")
    assert result is False  # correctness rubric (index 0) returned No


def test_evaluate_correctness_weight_equals_sum_of_other_rubric_weights():
    """Correctness weight should equal the combined weight of all other rubrics (50/50 split).

    Without ref_doc/ref_pages: other rubrics are readability(1.0) + conciseness(1.0) = 2.0.
    So correctness weight = 2.0, total weight = 4.0.
    With all three returning Yes: reward = (2.0 + 1.0 + 1.0) / 4.0 = 1.0.
    """
    judge = _make_judge(["Yes", "Yes", "Yes"])
    ctx = _make_context("The price is $125")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=False, rubrics_threshold=0.0, max_workers=1
    )
    ae.evaluate(reference="$125")
    assert ctx.metadata["rubric_reward"] == pytest.approx(1.0)


def test_evaluate_correctness_fail_does_not_raise_when_fail_on_mismatch_false():
    """A correctness miss with fail_on_mismatch=False does not raise, it just lowers the reward."""
    # correctness: No (weight 2.0), readability: Yes (1.0), conciseness: Yes (1.0)
    # reward = (0 + 1.0 + 1.0) / 4.0 = 0.5
    judge = _make_judge(["No", "Yes", "Yes"])
    ctx = _make_context("wrong answer")
    ae = GeneratedAnswerEvaluator(
        judge=judge, x=ctx, fail_on_mismatch=False, rubrics_threshold=0.0, max_workers=1
    )
    result = ae.evaluate(reference="$125")
    assert result is False
    assert ctx.metadata["rubric_reward"] == pytest.approx(0.5)
