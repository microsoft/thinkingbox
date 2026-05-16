# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tests.mock_session import create_mock_session
from thinkingbox.common.chat_types import Text
from thinkingbox.common.judge import Judge, _parse_yesno


def _mock_judge_llm(responses: list[str]):
    completions = [[Text(role="assistant", content=resp)] for resp in responses]
    return create_mock_session(completions=completions)


def test_simple_yes_no():
    """Test simple 'yes' and 'no' responses."""
    assert _parse_yesno("yes") is True
    assert _parse_yesno("Yes") is True
    assert _parse_yesno("no") is False
    assert _parse_yesno("maybe") is False


def test_punctuation_removal():
    """Test that punctuation is properly removed.

    This is the main test case for the bug fix. The old implementation
    used text.replace(".,;:\n", " ") which tried to replace the entire
    string as one unit, not each character individually.
    """
    assert _parse_yesno("Yes.") is True
    assert _parse_yesno("Yes,") is True
    assert _parse_yesno("Yes;") is True
    assert _parse_yesno("Yes:") is True
    assert _parse_yesno("Yes\n") is True
    assert _parse_yesno("Yes, I agree.") is True


def test_yes_at_beginning_or_end():
    """Test 'yes' at the beginning or end of a sentence."""
    assert _parse_yesno("yes, I agree") is True
    assert _parse_yesno("The answer is yes") is True
    assert _parse_yesno("The answer is yes.") is True


def test_text_yesno_with_motivation_parses_json():
    responses = ['{"motivation": "Answer is stated", "answer": "Yes"}']
    judge = Judge(_mock_judge_llm(responses))

    result = judge.text_yesno_with_motivation("Assistant reply", "Did they comply?")

    assert result["answer"] is True
    assert result["motivation"] == "Answer is stated"
    assert judge.drain_decisions() == [
        {
            "question": "Did they comply?",
            "answer": True,
            "motivation": "Answer is stated",
        }
    ]


def test_text_yesno_with_motivation_handles_invalid_json():
    judge = Judge(_mock_judge_llm(["Yes, absolutely"]))

    result = judge.text_yesno_with_motivation("Assistant reply", "Is it correct?")

    assert result["answer"] is False
    assert result["motivation"] == "Failed to parse response"
    assert judge.drain_decisions() == [
        {
            "question": "Is it correct?",
            "answer": False,
            "motivation": "Failed to parse response",
        }
    ]


def test_text_yesno_with_motivation_handles_single_quoted_keys():
    responses = ["{'motivation': \"Looks good\", 'answer': \"Yes\"}"]
    judge = Judge(_mock_judge_llm(responses))

    result = judge.text_yesno_with_motivation(
        "Assistant reply", "Does it look correct?"
    )

    assert result["answer"] is True
    assert result["motivation"] == "Looks good"


def test_text_yesno_with_motivation_handles_unquoted_keys():
    judge = Judge(_mock_judge_llm(['{motivation: "Clear", answer: "Yes"}']))

    result = judge.text_yesno_with_motivation("Assistant reply", "Is it ready?")

    assert result["answer"] is True
    assert result["motivation"] == "Clear"


def test_text_yesno_returns_boolean_and_records_decision():
    judge = Judge(
        _mock_judge_llm(['{"motivation": "Insufficient info", "answer": "No"}']),
        judge_type="motivation",
    )

    assert judge.text_yesno("Assistant reply", "Was it compliant?") is False
    assert judge.drain_decisions() == [
        {
            "question": "Was it compliant?",
            "answer": False,
            "motivation": "Insufficient info",
        }
    ]
