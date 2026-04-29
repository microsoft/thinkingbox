# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import uuid
from html import escape as escape_html
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from thinkingbox.common.tag_types import TestCaseTags
from thinkingbox.common.usage_types import Usage


def _pp_tag(tag, title, message, html=False):
    if html and tag:
        # called tool names can contain reserved characters too if the model
        # makes a mistake
        title = escape_html(title)

        message = escape_html(message)
        return f"<{tag}>[{title}]</{tag}>\n{message}".rstrip("\n") + "\n"
    return f"[{title}]\n{message}".rstrip("\n") + "\n"


class Message(BaseModel):
    T: Literal["Message"] = "Message"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def pp(self, html: bool = False):
        return [str(self)]


TextMessageTag = Literal[
    # Reasoning message, not visible to user
    "think",
    # Text response to user
    "text",
    # Injected text message, visible to user
    "direct",
]


class Text(Message):
    T: Literal["Text"] = "Text"
    role: Literal["system", "user", "assistant"]
    content: str

    @property
    def tag(self) -> TextMessageTag:
        # Keep it in metadata for backwards compatibility
        # TODO promote to top level attribute?
        return self.metadata.get("tag", "text")

    @property
    def is_visible(self) -> bool:
        return self.tag != "think"

    def pp(self, html: bool = False) -> list[str]:
        msg_type = self.tag
        tag = "think" if msg_type == "think" else self.role
        return [_pp_tag(tag, f"{self.role}::{msg_type}", self.content, html=html)]


class ToolCall(Message):
    T: Literal["ToolCall"] = "ToolCall"
    name: str
    arguments: dict
    id: str = Field(default_factory=lambda: "call_" + uuid.uuid4().hex)

    def pp(self, html: bool = False) -> list[str]:
        error = self.metadata.get("error")
        if error:
            return [_pp_tag("error", "assistant::call", error, html=html)]
        args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in self.arguments.items())
        return [
            _pp_tag(
                "toolcall", "assistant::call", f"{self.name}({args_str})", html=html
            )
        ]


class ParallelToolCall(Message):
    T: Literal["ParallelToolCall"] = "ParallelToolCall"
    tool_calls: list[ToolCall]

    def pp(self, html: bool = False) -> list[str]:
        out = []
        for tc in self.tool_calls:
            out.extend(tc.pp(html=html))
        return out


class ToolResponse(Message):
    T: Literal["ToolResponse"] = "ToolResponse"
    name: str
    content: str
    id: str = Field(default_factory=lambda: "call_" + uuid.uuid4().hex)

    def pp(self, html: bool = False) -> list[str]:
        out = []
        out.append(
            _pp_tag("toolresp", f"{self.name}::result", self.content, html=html),
        )
        direct_response = self.metadata.get("direct_response")
        if direct_response is not None:
            out.append(
                _pp_tag(
                    "assistant", f"{self.name}::direct", direct_response, html=html
                ),
            )
        return out


MessageT = Text | ToolCall | ParallelToolCall | ToolResponse


class Conversation(BaseModel):
    messages: list[MessageT] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=lambda: {"usage": []})

    def pp(self, html: bool = False) -> list[str]:
        out = []
        for msg in self.messages:
            out.extend(msg.pp(html=html))
        return out


class ToolCallResponse(BaseModel):
    tool_call: ToolCall
    tool_response: ToolResponse


class ToolDef(BaseModel):
    name: str
    description: str
    input_schema: dict
    direct_response: str | None = None
    is_end_turn: bool = False


class MessageConfig(BaseModel):
    tag_done: str

    def is_done(self, text: str):
        # is it: content <DONE> ?
        return text.strip().endswith(self.tag_done)


def get_conversation_transcript(
    messages: list[Message],
    limit: int = 0,
    content_transform_fn: Callable[[str], str] | None = None,
) -> str:
    """Get the conversation transcript as a string, including only the messages
    visible to the user.

    If limit is 0, get the full transcript from beginning to end.
    Otherwise, get the tail of the transcript, including up to `limit`
    assistant messages, stopping at the last assistant message to be included.

    e.g. limit = 1 -> only the last assistant message
    limit = 2 -> assistant, user, assistant
    """
    if limit < 0:
        raise ValueError("limit must be >= 0")

    lines = []
    assistant_count = 0

    for m in reversed(messages):
        # filter only user and assistant text messages
        if not (isinstance(m, Text) and (m.role in ("user", "assistant"))):
            continue

        # ignore Text messages that should not be visible to User
        if not m.is_visible:
            continue

        # ignore dummy messages inserted for thinking-only responses
        if m.metadata.get("is_dummy"):
            continue

        if m.role == "assistant":
            assistant_count += 1

        content = m.content
        if content_transform_fn is not None:
            content = content_transform_fn(m.content)
        label = "Assistant" if (m.role == "assistant") else "User"
        lines.append(f"{label}: {content}")

        if limit > 0 and assistant_count >= limit:
            break

    lines.reverse()
    return "\n\n".join(lines)


class TestContext(BaseModel):
    __test__ = False  # prevent pytest from collecting this as a test class

    # The last assistant text message
    response: str = ""

    # The list of direct responses emitted as assistant text messages
    # from formatted tool responses
    tool_direct_responses: list[str] = Field(default_factory=list)

    # mapping of (server name -> effects) where effects is the object
    # returned by the special 'geteffects' function of each server
    effects: dict[str, Any] = Field(default_factory=dict)

    # list of all tool calls with their respective responses
    tool_calls: list[ToolCallResponse] = Field(default_factory=list)
    messages: list[MessageT] = Field(default_factory=list)

    def transcript(self, limit: int = 0) -> str:
        return get_conversation_transcript(self.messages, limit=limit)

    # mapping of (server name -> init result) where init result is the
    # object returned by the special 'init' function of each server
    # (minus the status key)
    init_result: dict[str, Any] = Field(default_factory=dict)

    # Session ID within the session proxy
    session_id: str = "00000000-0000-0000-0000-000000000000"

    # arbitrary metadata written by test code and fixtures during execution
    metadata: dict[str, Any] = Field(default_factory=dict)

    def query(self) -> str:
        """Get the original user query from the first user message in the transcript."""
        # This skips dummy messages - in the same way as in get_conversation_transcript() - but may still
        # be brittle to history.
        for m in self.messages:
            if (
                isinstance(m, Text)
                and m.role == "user"
                and not m.metadata.get("is_dummy", False)
                and getattr(m, "tag", None) != "think"
            ):
                return m.content
        raise ValueError("No user message found in messages")


class TestResult(BaseModel):
    # the test result.
    # If the test passes (no exceptions or failed assertions): True
    # If the test fails (assertion raised): False
    # If there is an error (exception other than assertion raised): also False, but
    #   ideally should be ignored because is_system_error=True.
    result: bool = Field(description="True if the test passed, False otherwise")
    reward: float = Field(ge=0.0, le=1.0, description="Reward value between 0 and 1")

    # If true, the test execution raised an exception that was not an
    #   assertion. This is likely a bug in the test. When this happens,
    #   the value stored in result is irrelevant
    is_system_error: bool = False

    # traceback
    tb: str = ""

    # printed string if capturing print
    prints: str = ""

    # line number where the test failed
    lineno: int = -1

    # content of the line where the test failed
    line_content: str = ""

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata related to the test result, such as judge motivations",
    )

    def pp(self, html: bool = False) -> list[str]:
        maybe_system_error = "" if not self.is_system_error else " (error in test!)"
        tag = "good" if self.result else "error"
        msg = [f"Result: {self.result}{maybe_system_error}"]
        if self.prints:
            msg.append("Prints:\n" + self.prints)

        judge_motivation = self.metadata.get("judge_motivation")
        if judge_motivation:
            motiv_lines = []
            for item in judge_motivation:
                question = item.get("question", "")
                answer_bool = item.get("answer")
                answer = "Yes" if answer_bool else "No"
                motivation = item.get("motivation", "")
                motiv_lines.append(f"- {answer}: {question} :: {motivation}")
            msg.append("Judge motivation:\n" + "\n".join(motiv_lines))

        if not self.result:
            msg.append(self.tb)
            msg.append(f"Line content ({self.lineno}):\n{self.line_content}")
        return [
            _pp_tag(tag, "test::result", "\n".join(msg), html=html),
        ]


FinishReason = Literal[
    "done",
    "end_turn_tool",
    "agent_error",
    "agent_limit",
    "user_limit",
    "user_done",
    "no_user_llm",
    "skipped",
]


class DecodeResult(BaseModel):
    uid: str
    messages: list[MessageT]
    test_result: TestResult | None = None
    test_context: TestContext | None = None
    test_tags: TestCaseTags | None = None
    tools: list[ToolDef] | None = None
    raw_messages: list[dict] | None = None
    user_llm_history: list[list[MessageT]] | None = None
    usage: list[Usage] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # If true, there was an error decoding this, and it is expected to be incomplete.
    # metadata["error"]["message"] might contain a string describing the error.
    # Note that even if this is False, test result can still contain errors related to
    # the test (test_result.is_system_error)
    is_system_error: bool = False
    finish_reason: FinishReason = "done"
