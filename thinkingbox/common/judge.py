# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import re
from typing import Any

from .chat_types import Message, Text
from .llm_session_base import LLMSessionBase


class JudgeException(Exception):
    pass


class Judge:
    def __init__(self, llm: LLMSessionBase, judge_type: str = "legacy"):
        self.llm = llm

        if judge_type not in ("legacy", "motivation"):
            raise ValueError(f"Invalid judge_type: {judge_type}")

        self.judge_type = judge_type
        self._decisions: list[dict[str, Any]] = []

    def _get_text_completion(self, messages: list[Message]) -> Text:
        try:
            out = self.llm.get_completion_sync(
                conversation=messages,
                update_conversation=False,
            )
            # If it's a reasoning model there could be a think message,
            # so only take the last message which is the final answer.
            # But we shouldn't need reasoning models as judge
            if len(out) >= 1 and isinstance(out[-1], Text):
                return out[-1]
            return Text(role="assistant", content="")
        except Exception as e:
            raise JudgeException(f"Error getting Judge completion: {e}") from e

    def _grade(
        self,
        system_prompt: str,
        user_prompt: str,
        format_dict: dict[str, Any],
        encode_keys: tuple[str, ...] = ("message", "first", "second"),
    ) -> str:
        """Format prompt with given dictionary and return model response content.

        All callers must supply ``format_dict``; selected keys are HTML tag encoded.
        """
        # Encode selected keys (copy dict to avoid side-effects)
        fd = dict(format_dict)
        for k in encode_keys:
            v = fd.get(k)
            if isinstance(v, str):
                fd[k] = safe_tag_encode(v)

        user_msg = user_prompt.format(**fd)
        msg = self._get_text_completion(
            messages=[
                Text(role="system", content=system_prompt),
                Text(role="user", content=user_msg),
            ]
        )
        return msg.content

    def evaluate_bool(self, system: str, user: str) -> bool:
        """Generic yes/no evaluation with custom system and user prompts."""
        msg = self._get_text_completion(
            messages=[
                Text(role="system", content=system),
                Text(role="user", content=user),
            ]
        )
        return _parse_yesno(msg.content)

    def text_yesno_with_motivation(self, text: str, question: str) -> dict[str, Any]:
        """
        Grade the given text with a yes/no answer to the question
        Returns a dictionary with the answer and motivation.

        Args:
            text: The text to be graded.
            question: The question to ask about the text.
        Returns:
            A dictionary with 'answer' and 'motivation'.
        """
        msg = self._grade(
            system_prompt=SYSTEM_TEXT_YESNO_MOTIVATION,
            user_prompt=USER_TEXT_YESNO,
            format_dict={"message": text, "question": question},
        )

        answer_raw = _quoted_keys(msg)

        try:
            answer = json.loads(answer_raw)
            answer["answer"] = _parse_yesno(answer["answer"])
        except json.JSONDecodeError:
            answer = {
                # If the model failed to follow the JSON format contract we
                # cannot trust the raw response, so default to "No".
                "answer": False,
                "motivation": "Failed to parse response",
            }

        self._decisions.append(
            {
                "question": question,
                "answer": bool(answer.get("answer")),
                "motivation": answer.get("motivation", ""),
            }
        )

        return answer

    def text_yesno_legacy(self, text: str, question: str) -> bool:
        user_msg = USER_TEXT_YESNO.format(
            message=safe_tag_encode(text),
            question=question,
        )
        return self.evaluate_bool(system=SYSTEM_TEXT_YESNO, user=user_msg)

    def text_yesno(self, text: str, question: str) -> bool:
        """
        Grade the given text with a yes/no answer to the question.
        This is a legacy method, only returning the boolean answer,
        it will deprecate soon.

        Args:
            text: The text to be graded.
            question: The question to ask about the text.
        Returns:
            bool: wether the proposed solution conform to expectation
                or not.
        """
        if self.judge_type == "legacy":
            answer = self.text_yesno_legacy(text, question)

        else:
            answer_motivation = self.text_yesno_with_motivation(text, question)
            answer = answer_motivation["answer"]

        return answer

    def compare_yesno(self, first: str, second: str, question: str) -> bool:
        response = self._grade(
            system_prompt=SYSTEM_COMPARE_YESNO,
            user_prompt=USER_COMPARE_YESNO,
            format_dict={"first": first, "second": second, "question": question},
        )
        return _parse_yesno(response)

    def yesno(self, question: str) -> bool:
        # Keep simple wrapper using _grade for consistency
        response = self._grade(
            system_prompt=SYSTEM_YESNO,
            user_prompt="{question}",
            format_dict={"question": question},
            encode_keys=(),  # no tag encoding needed
        )
        return _parse_yesno(response)

    def no_repeat(self, first: str, second_list: list[str]) -> bool:
        for second in second_list:
            if self.compare_yesno(
                first=first,
                second=second,
                question=QUESTION_FIRST_REPEATS_SECOND,
            ):
                return False
        return True

    def drain_decisions(self) -> list[dict[str, Any]]:
        """Return and clear any stored judge decisions."""

        decisions = self._decisions
        self._decisions = []
        return decisions


def safe_tag_encode(text: str):
    """Encode < and > and any other characters used internally by prompt templates to avoid interfering with prompt formatting and model interpretation.

    Args:
        text (str): The text to encode.

    Returns:
        str: The encoded text.
    """
    return text.replace("<", "&lt;").replace(">", "&gt;")


_PUNCTUATION_TRANSLATION = str.maketrans({c: " " for c in ".,;:\n"})


def _parse_yesno(text: str):
    cleaned = text.translate(_PUNCTUATION_TRANSLATION)
    tokens = cleaned.strip().split()
    if tokens:
        if tokens[0].lower() == "yes":
            return True
        if tokens[-1].lower() == "yes":
            return True
    return False


def _quoted_keys(raw: str) -> str:
    """Ensure JSON-style double quotes around specific keys if they appear.
    Only rewrites occurrences of answer and motivation keys when they are:
      - single quoted: 'answer': or 'motivation':
      - unquoted: answer: or motivation:
    Leaves already correctly double-quoted keys intact.
    Returns the modified string (idempotent on properly quoted input).
    """

    def fix_key(s: str, key: str) -> str:
        # 1. Replace single-quoted key -> double quoted
        s = re.sub(rf"'({key})'\s*:", r'"\1":', s)
        # 2. Replace unquoted key at start of object or after { or ,
        #    Using a lookbehind for { or , or start-of-string to avoid matching in values.
        s = re.sub(rf"(?:(?<=\{{)|(?<=,)|^)(\s*)({key})(\s*:)", r'\1"\2"\3', s)
        return s

    for k in ("answer", "motivation"):
        raw = fix_key(raw, k)

    return raw


SYSTEM_YESNO = """\
You are a judge, your task is to answer a question.
Answer with a single word: Yes or No.
"""

SYSTEM_TEXT_YESNO_MOTIVATION = """\
You are a judge, your task is to read the contents of a message and answer a question about the message.
The message may contain a brief interaction between the assistant and the user. When answering, consider
only the assistant responses across the conversation.
A message will be presented between <message> and </message> tags, and a question will be asked afterwards.
Answer the question as follows:

{
    "motivation": a brief explanation of your answer (1-2 sentences) in quotes,
    "answer": a single word: Yes or No in quotes.
}
"""

SYSTEM_TEXT_YESNO = """\
You are a judge, your task is to read the contents of a message and answer a question about the message.
A message will be presented between <message> and </message> tags, and a question will be asked afterwards.
Answer the question with a single word: Yes or No."""

USER_TEXT_YESNO = """\
<message>
{message}
</message>
{question}"""

SYSTEM_COMPARE_YESNO = """\
You are a judge, your task is to read the contents of two messages and answer a question about the messages.
The two messages will be presented as follows:
<first>
content of the first message
</first>
<second>
content of the second message
</second>
A question will be asked afterwards.
Answer the question with a single word: Yes or No."""

USER_COMPARE_YESNO = """\
<first>
{first}
</first>
<second>
{second}
</second>
{question}"""


QUESTION_FIRST_REPEATS_SECOND = """\
Does the content of the first message repeat the content of the second message?
"""
