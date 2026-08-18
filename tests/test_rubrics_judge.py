# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import types
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from thinkingbox.common.config_types import RubricConfig
from thinkingbox.common.rubrics_judge import (
    SYSTEM_PROMPT_PENALTY,
    SYSTEM_PROMPT_POSITIVE,
    RubricJudge,
)


def _make_judge(return_value=True):
    m = Mock()
    m.evaluate_bool.return_value = return_value
    return m


def _make_x():
    return types.SimpleNamespace(metadata={})


def test_rubric_judge_all_pass():
    """Test when all rubrics receive YES (1)."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="First rubric", weight=10),
            RubricConfig(criterion="Second rubric", weight=20),
        ],
        global_threshold=0.5,
        throw_on_failure=False,
    )

    assert result.reward == 1.0
    assert result.passed is True
    assert mock_judge.evaluate_bool.call_count == 2


def test_rubric_judge_partial_pass():
    """Test when rubrics receive different ratings."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="First rubric", weight=10),
            RubricConfig(criterion="Second rubric", weight=10),
        ],
        global_threshold=0.3,
        throw_on_failure=False,
    )

    assert abs(result.reward - 0.5) < 0.001  # 10/20 = 0.5
    assert result.passed is True
    assert mock_judge.evaluate_bool.call_count == 2


def test_rubric_judge_below_threshold():
    """Test when reward is below threshold (should raise AssertionError)."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [False, False]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)

    with pytest.raises(AssertionError) as exc_info:
        rubric_judge.evaluate(
            response="Sample text",
            rubrics=[
                RubricConfig(criterion="First rubric", weight=10),
                RubricConfig(criterion="Second rubric", weight=10),
            ],
            global_threshold=0.7,
            throw_on_failure=True,
        )

    error_msg = str(exc_info.value)
    assert "0.0" in error_msg or "0.000" in error_msg
    assert "0.7" in error_msg or "0.700" in error_msg


def test_rubric_judge_weighted_scoring():
    """Test that weights are properly considered in scoring."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="Important rubric", weight=40),
            RubricConfig(criterion="Less important rubric", weight=10),
        ],
        global_threshold=0.6,
        throw_on_failure=False,
    )

    assert abs(result.reward - 0.8) < 0.001  # 40/50 = 0.8


def test_rubric_judge_float_weights():
    """Test that float weights are supported."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False, True]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="First rubric", weight=1.5),
            RubricConfig(criterion="Second rubric", weight=0.5),
            RubricConfig(criterion="Third rubric", weight=1.0),
        ],
        global_threshold=0.5,
        throw_on_failure=False,
    )

    assert abs(result.reward - (2.5 / 3.0)) < 0.001


def test_rubric_judge_prompt_formatting():
    """Test that prompts are formatted correctly."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    text_to_evaluate = "This is the text to evaluate"
    rubric_question = "Does it meet the criterion?"

    rubric_judge.evaluate(
        response=text_to_evaluate,
        rubrics=[RubricConfig(criterion=rubric_question, weight=10)],
        global_threshold=0.5,
        throw_on_failure=False,
    )

    mock_judge.evaluate_bool.assert_called_once()
    call_args = mock_judge.evaluate_bool.call_args

    assert call_args[1]["system"] == SYSTEM_PROMPT_POSITIVE
    user_prompt = call_args[1]["user"]
    assert text_to_evaluate in user_prompt
    assert rubric_question in user_prompt


def test_rubric_judge_penalty_prompt_formatting():
    """Test that penalty rubrics use the penalty system prompt."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    text_to_evaluate = "This is the text to evaluate"
    rubric_question = "Does the response contain harmful content?"

    rubric_judge.evaluate(
        response=text_to_evaluate,
        rubrics=[RubricConfig(criterion=rubric_question, weight=5, is_penalty=True)],
        global_threshold=0.0,
        throw_on_failure=False,
    )

    mock_judge.evaluate_bool.assert_called_once()
    call_args = mock_judge.evaluate_bool.call_args

    assert call_args[1]["system"] == SYSTEM_PROMPT_PENALTY
    user_prompt = call_args[1]["user"]
    assert text_to_evaluate in user_prompt
    assert rubric_question in user_prompt


def test_rubric_judge_empty_rubrics():
    """Test that empty rubrics list raises ValueError."""
    mock_judge = Mock()
    mock_x = _make_x()
    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)

    with pytest.raises(ValueError, match="Rubrics list cannot be empty"):
        rubric_judge.evaluate(
            response="Sample text",
            rubrics=[],
            global_threshold=0.5,
        )


def test_rubric_judge_zero_threshold():
    """Test when threshold is 0 (any non-negative reward passes)."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [False, True]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="First rubric", weight=10),
            RubricConfig(criterion="Second rubric", weight=10),
        ],
        global_threshold=0.0,
        throw_on_failure=False,
    )

    assert result.reward == 0.5
    assert result.passed is True


def test_basic_rubric_config():
    """Test basic RubricConfig usage."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)

    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="First rubric", weight=10),
            RubricConfig(criterion="Second rubric", weight=10),
        ],
        global_threshold=0.4,
        throw_on_failure=False,
    )

    assert result.reward == 0.5


def test_deduction_penalty():
    """Test that deduction penalties subtract from earned score."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, True]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    rubrics = [
        RubricConfig(criterion="Quality", weight=10.0),
        RubricConfig(
            criterion="Verbosity penalty",
            weight=5.0,
            is_penalty=True,
            penalty_type="deduction",
        ),
    ]

    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=rubrics,
        global_threshold=0.0,
        throw_on_failure=False,
    )

    assert abs(result.reward - 0.5) < 0.001  # (10 - 5) / 10 = 0.5


def test_multiplicative_penalty():
    """Test that multiplicative penalties scale the reward."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, True]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    rubrics = [
        RubricConfig(criterion="Quality", weight=10.0),
        RubricConfig(
            criterion="Safety violation",
            weight=0.5,
            is_penalty=True,
            penalty_type="multiplicative",
        ),
    ]

    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=rubrics,
        global_threshold=0.0,
        throw_on_failure=False,
    )

    assert abs(result.reward - 0.5) < 0.001  # 1.0 * (1 - 0.5) = 0.5


def test_penalty_no_effect_when_rating_zero():
    """Test that penalty has no effect when rating is NO (0)."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    rubrics = [
        RubricConfig(criterion="Quality", weight=10.0),
        RubricConfig(
            criterion="Verbosity penalty",
            weight=5.0,
            is_penalty=True,
            penalty_type="deduction",
        ),
    ]

    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=rubrics,
        global_threshold=0.0,
        throw_on_failure=False,
    )

    assert abs(result.reward - 1.0) < 0.001  # (10 - 0) / 10 = 1.0


def test_deduction_and_multiplicative_penalties_combined():
    """Deduction applies before multiplicative scaling, per docs."""
    mock_judge = Mock()
    # Quality YES, Deduction penalty YES, Multiplicative penalty YES
    mock_judge.evaluate_bool.side_effect = [True, True, True]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    rubrics = [
        RubricConfig(criterion="Quality", weight=10.0),
        RubricConfig(
            criterion="Localized issue penalty",
            weight=3.0,
            is_penalty=True,
            penalty_type="deduction",
        ),
        RubricConfig(
            criterion="Global trust penalty",
            weight=0.5,
            is_penalty=True,
            penalty_type="multiplicative",
        ),
    ]

    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=rubrics,
        global_threshold=0.0,
        throw_on_failure=False,
    )

    # base_reward = (10 - 3) / 10 = 0.7
    # reward = 0.7 * (1 - 0.5) = 0.35
    assert abs(result.reward - 0.35) < 0.001


def test_multiple_multiplicative_penalties_multiply():
    """Multiple multiplicative penalties should combine multiplicatively."""
    mock_judge = Mock()
    # Quality YES, penalty1 YES, penalty2 YES
    mock_judge.evaluate_bool.side_effect = [True, True, True]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    rubrics = [
        RubricConfig(criterion="Quality", weight=10.0),
        RubricConfig(
            criterion="Penalty 1",
            weight=0.2,
            is_penalty=True,
            penalty_type="multiplicative",
        ),
        RubricConfig(
            criterion="Penalty 2",
            weight=0.5,
            is_penalty=True,
            penalty_type="multiplicative",
        ),
    ]

    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=rubrics,
        global_threshold=0.0,
        throw_on_failure=False,
    )

    # base_reward = 1.0
    # reward = 1.0 * (1 - 0.2) * (1 - 0.5) = 0.4
    assert abs(result.reward - 0.4) < 0.001


def test_multiplicative_penalty_weight_validation():
    """Multiplicative penalties require weights in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        RubricConfig(
            criterion="Overweight multiplicative penalty",
            weight=1.5,
            is_penalty=True,
            penalty_type="multiplicative",
        )


def test_check_threshold_returns_bool():
    """Test check_threshold returns True when reward meets threshold."""
    rubric_judge = RubricJudge(judge=_make_judge(), x=_make_x(), max_workers=1)

    assert rubric_judge.check_threshold(0.8, 0.7) is True
    assert rubric_judge.check_threshold(0.7, 0.7) is True
    assert rubric_judge.check_threshold(0.5, 0.7) is False


def test_assert_threshold_raises():
    """Test check_threshold raises when throw_on_failure=True and reward is below threshold."""
    rubric_judge = RubricJudge(judge=_make_judge(), x=_make_x(), max_workers=1)

    # Should return True when passing
    assert rubric_judge.check_threshold(0.8, 0.7, throw_on_failure=True) is True

    # Should raise when below threshold with throw_on_failure=True
    with pytest.raises(AssertionError, match="below threshold"):
        rubric_judge.check_threshold(0.5, 0.7, throw_on_failure=True)


def test_max_workers_parameter():
    """Test that max_workers parameter is respected."""
    rubric_judge = RubricJudge(judge=_make_judge(), x=_make_x(), max_workers=5)
    assert rubric_judge.max_workers == 5


def test_parallel_evaluates_all_rubrics():
    """Test that parallel execution evaluates all rubrics."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)

    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="Rubric 1", weight=10),
            RubricConfig(criterion="Rubric 2", weight=10),
            RubricConfig(criterion="Rubric 3", weight=10),
        ],
        global_threshold=0.5,
        throw_on_failure=False,
    )

    assert result.reward == 1.0
    assert mock_judge.evaluate_bool.call_count == 3


def test_evaluation_result_scores():
    """Test that EvaluationResult contains all rubric scores."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="Quality", weight=10.0),
            RubricConfig(criterion="Safety", weight=10.0),
        ],
        global_threshold=0.0,
        throw_on_failure=False,
    )

    assert len(result.scores) == 2
    assert result.scores[0].criterion == "Quality"
    assert result.scores[0].rating == 1
    assert result.scores[1].criterion == "Safety"
    assert result.scores[1].rating == 0


def test_evaluation_result_assert_passed():
    """Test EvaluationResult.assert_passed() method."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[RubricConfig(criterion="Quality", weight=10)],
        global_threshold=0.5,
        throw_on_failure=False,
    )

    reward = result.assert_passed()
    assert reward == 1.0


def test_evaluation_result_assert_passed_raises():
    """Test EvaluationResult.assert_passed() raises on failure."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = False
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[RubricConfig(criterion="Quality", weight=10)],
        global_threshold=0.5,
        throw_on_failure=False,
    )

    with pytest.raises(AssertionError):
        result.assert_passed()


def test_custom_reward_function():
    """Test delegated reward function with weighted score calculation."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, True, False]
    mock_x = _make_x()

    def weighted_quality_reward(scores):
        """Custom reward that doubles quality scores and ignores safety."""
        total = 0.0
        for score in scores:
            if "quality" in score.criterion.lower():
                total += score.rating * score.weight * 2
            elif "completeness" in score.criterion.lower():
                total += score.rating * score.weight
        max_possible = sum(
            s.weight for s in scores if "safety" not in s.criterion.lower()
        )
        return total / max_possible if max_possible > 0 else 0.0

    rubric_judge = RubricJudge(
        judge=mock_judge, x=mock_x, reward_fn=weighted_quality_reward, max_workers=1
    )
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="Quality check", weight=10),
            RubricConfig(criterion="Completeness check", weight=10),
            RubricConfig(criterion="Safety check", weight=10),
        ],
        global_threshold=0.3,
        throw_on_failure=False,
    )

    # Quality: 1 * 10 * 2 = 20, Completeness: 1 * 10 = 10, Safety ignored
    # max_possible = 20 (quality + completeness weights)
    # reward = 30 / 20 = 1.5, clamped to 1.0
    assert result.reward == 1.0


def test_custom_reward_function_raises_error():
    """Test that errors from custom reward functions propagate."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False]  # Second score will be 0
    mock_x = _make_x()

    def geometric_mean_reward(scores):
        """Compute geometric mean of scores - fails on zero."""
        import math

        product = 1.0
        for score in scores:
            product *= score.score
        # This will raise ValueError for log(0) when any score is 0
        return math.exp(math.log(product) / len(scores))

    rubric_judge = RubricJudge(
        judge=mock_judge, x=mock_x, reward_fn=geometric_mean_reward, max_workers=1
    )

    with pytest.raises(ValueError):
        rubric_judge.evaluate(
            response="Sample text",
            rubrics=[
                RubricConfig(criterion="Quality", weight=10),
                RubricConfig(criterion="Safety", weight=10),
            ],
            global_threshold=0.5,
            throw_on_failure=False,
        )


def test_custom_reward_function_returns_non_numeric():
    """Test that non-numeric return from reward function raises TypeError."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    def categorical_reward(scores):
        """Return category instead of number - simulating a bug."""
        total = sum(s.score for s in scores)
        max_total = sum(s.weight for s in scores)
        ratio = total / max_total if max_total > 0 else 0
        # Bug: returns string category instead of float
        if ratio > 0.8:
            return "excellent"
        elif ratio > 0.5:
            return "good"
        return "poor"

    rubric_judge = RubricJudge(
        judge=mock_judge, x=mock_x, reward_fn=categorical_reward, max_workers=1
    )

    with pytest.raises(ValueError, match="could not convert string to float"):
        rubric_judge.evaluate(
            response="Sample text",
            rubrics=[RubricConfig(criterion="Quality", weight=10)],
            global_threshold=0.5,
            throw_on_failure=False,
        )


def test_evaluation_result_has_global_threshold():
    """Test that EvaluationResult stores global_threshold."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[RubricConfig(criterion="Quality", weight=10)],
        global_threshold=0.75,
        throw_on_failure=False,
    )

    assert result.global_threshold == 0.75


def test_delegated_reward_clamped_to_range():
    """Test that delegated reward functions are clamped to [0, 1]."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    def bad_reward(scores):
        return 1.5

    rubric_judge = RubricJudge(
        judge=mock_judge, x=mock_x, reward_fn=bad_reward, max_workers=1
    )
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[RubricConfig(criterion="Quality", weight=10)],
        global_threshold=0.5,
        throw_on_failure=False,
    )

    assert result.reward == 1.0


def test_delegated_reward_handles_negative():
    """Test that delegated reward functions handle negative values."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = True
    mock_x = _make_x()

    def negative_reward(scores):
        return -0.5

    rubric_judge = RubricJudge(
        judge=mock_judge, x=mock_x, reward_fn=negative_reward, max_workers=1
    )
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[RubricConfig(criterion="Quality", weight=10)],
        global_threshold=0.0,
        throw_on_failure=False,
    )

    # Should be clamped to 0.0
    assert result.reward == 0.0


def test_evaluate_writes_metadata():
    """evaluate() writes rubric_reward and rubric_scores to x.metadata."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)
    result = rubric_judge.evaluate(
        response="Sample text",
        rubrics=[
            RubricConfig(criterion="Criterion A", weight=10),
            RubricConfig(criterion="Criterion B", weight=10),
        ],
    )

    assert mock_x.metadata["rubric_reward"] == result.reward
    assert mock_x.metadata["rubric_reward"] == 0.5
    scores = mock_x.metadata["rubric_scores"]
    assert len(scores) == 2
    ratings = {s["criterion"]: s["rating"] for s in scores}
    assert ratings["Criterion A"] == 1
    assert ratings["Criterion B"] == 0


def test_evaluate_overwrites_metadata_on_second_call():
    """Second evaluate() call overwrites earlier metadata values."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.side_effect = [True, False, False]
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)

    rubric_judge.evaluate(
        response="First response",
        rubrics=[RubricConfig(criterion="Criterion A", weight=10)],
    )
    assert mock_x.metadata["rubric_reward"] == 1.0

    rubric_judge.evaluate(
        response="Second response",
        rubrics=[
            RubricConfig(criterion="Criterion B", weight=10),
            RubricConfig(criterion="Criterion C", weight=10),
        ],
    )
    assert mock_x.metadata["rubric_reward"] == 0.0
    criteria = {s["criterion"] for s in mock_x.metadata["rubric_scores"]}
    assert criteria == {"Criterion B", "Criterion C"}


def test_evaluate_writes_metadata_even_when_throw_on_failure():
    """Metadata is written before the AssertionError when throw_on_failure=True."""
    mock_judge = Mock()
    mock_judge.evaluate_bool.return_value = False
    mock_x = _make_x()

    rubric_judge = RubricJudge(judge=mock_judge, x=mock_x, max_workers=1)

    with pytest.raises(AssertionError):
        rubric_judge.evaluate(
            response="Sample text",
            rubrics=[RubricConfig(criterion="Quality", weight=10)],
            global_threshold=0.5,
            throw_on_failure=True,
        )

    # Metadata must be written despite the AssertionError
    assert "rubric_reward" in mock_x.metadata
    assert mock_x.metadata["rubric_reward"] == 0.0
    assert "rubric_scores" in mock_x.metadata
