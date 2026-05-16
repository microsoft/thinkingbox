# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from thinkingbox.common.chat_types import (
    Message,
    MessageT,
    ToolDef,
)
from thinkingbox.common.llm_session_base import LLMSessionBase


class MockSessionLog(BaseModel):
    instance_created: bool = False
    remaining_completions: int = -1


class MockSessionConfig(BaseModel):
    completions: list[list[MessageT]]
    log: MockSessionLog | None = None


class MockSession(LLMSessionBase):
    def __init__(
        self,
        completions: list[list[Message]],
        log: MockSessionLog | None = None,
    ):
        self.conversation: list[Message] = []
        self.completions = completions
        self._tool_names = set()
        self.log = log
        if log is not None:
            if log.instance_created:
                raise RuntimeError(
                    "MockSessionLog.instance_created already set; "
                    "a second session was created with the same log"
                )
            log.instance_created = True
            log.remaining_completions = len(completions)

    @property
    def tool_names(self) -> set[str]:
        return self._tool_names

    def add_tools(self, tools: list[ToolDef]) -> None:
        for t in tools:
            self._tool_names.add(t.name)

    def reset_tools(self) -> None:
        self._tool_names.clear()

    def add_messages(self, messages: list[Message]) -> None:
        self.conversation.extend(messages)

    def reset_messages(self) -> None:
        self.conversation.clear()

    def get_internal_conversation(self) -> Any:
        return self.conversation

    async def get_completion(
        self,
        conversation: list[Message] | None = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[Message]:
        return self.get_completion_sync(
            conversation=conversation,
            update_conversation=update_conversation,
        )

    def get_completion_sync(
        self,
        conversation: list[Message] | None = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[Message]:
        out = self.completions.pop(0)
        if self.log is not None:
            self.log.remaining_completions = len(self.completions)
        if not conversation and update_conversation:
            for msg in out:
                self.conversation.append(msg.model_copy(deep=True))
        return out

    @classmethod
    def from_config(
        cls,
        config: MockSessionConfig,
    ) -> MockSession:
        return MockSession(
            completions=config.completions,
            log=config.log,
        )


class ErrorOnCompletionMockSession(MockSession):
    def __init__(
        self,
        completions,
        log: MockSessionLog | None = None,
        error_type=Exception,
        error_message="Simulated error",
        error_on_call=0,
    ):
        super().__init__(completions, log=log)
        self.call_count = 0
        self.error_on_call = error_on_call
        self.error_message = error_message
        self.error_type = error_type

    def get_completion_sync(self, *args, **kwargs) -> list[Message]:
        if self.call_count >= self.error_on_call:
            raise self.error_type(self.error_message)
        self.call_count += 1
        return super().get_completion_sync(*args, **kwargs)


def create_mock_session(**kwargs):
    return MockSession.from_config(MockSessionConfig(**kwargs))
