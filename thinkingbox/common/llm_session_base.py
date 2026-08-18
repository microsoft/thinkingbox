# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, AsyncIterator

import httpx
from pydantic import BaseModel

from thinkingbox.common.chat_types import Message, ToolDef
from thinkingbox.common.http_client import (
    BackoffAsyncClient,
    create_custom_ssl_context,
    get_httpx_client,
)
from thinkingbox.common.usage_types import Usage
from thinkingbox.common.utils import CredentialBase


class LLMSessionBase(ABC):
    """
    Base class for LLM session, which can be initialized with tool definitions
        and used to get completions from the LLM.

    It should keep a representation of the conversation and should update it with new
        completions. This is used to:
    - keep a history to continue the conversation, after appending a user response or tool response
    - retrieve the final completed conversation for training
    """

    last_usage: Usage | None = None

    @property
    @abstractmethod
    def tool_names(self) -> Collection[str]:
        """Collection of available tool names"""
        ...

    @abstractmethod
    def add_tools(self, tools: list[ToolDef]) -> None:
        """Add new tools definitions (schemas). If a tool already exists, replace it"""
        ...

    @abstractmethod
    def reset_tools(self) -> None:
        """Remove all tools"""
        ...

    @abstractmethod
    def add_messages(self, messages: list[Message]) -> None:
        """Add new messages"""
        ...

    @abstractmethod
    def reset_messages(self) -> None:
        """Reset messages"""
        ...

    @abstractmethod
    def get_internal_conversation(self) -> Any:
        """Get the conversation in the "native" format used to communicate with
        the decoder interface
        """
        ...

    @abstractmethod
    async def get_completion(
        self,
        conversation: list[Message] | None = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[Message]:
        """
        Get a completion.
        A completion may be mapped to a sequence of messages, therefore this method returns a list.
        If conversation is not None, use it as prefix and do not update
            the internal conversation
        If conversation is None, use the internal conversation as prefix and,
            if update_conversation, update it with the new completion
        """
        ...

    @abstractmethod
    def get_completion_sync(
        self,
        conversation: list[Message] | None = None,
        update_conversation: bool = True,
        response_schema: dict | None = None,
    ) -> list[Message]:
        """
        Same as get_completion but synchronous
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: BaseModel,
    ) -> "LLMSessionBase":
        """
        Create a session from a configuration
        """
        ...


class HTTPLLMSessionBase(LLMSessionBase):
    def __init__(
        self,
        credential: CredentialBase | None = None,
        client_certificate: str | None = None,
        trust_ca_path: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        use_dns_cache: bool = False,
        max_retries_server_error: int = 5,
        retryable_server_errors: tuple[int | str, ...] = (502, 503, 504),
    ):
        self.credential = credential
        self.client_certificate = client_certificate
        self.trust_ca_path = trust_ca_path
        self.headers = headers.copy() if headers else {}
        self.timeout_config = httpx.Timeout(
            timeout=timeout,
            connect=30.0,  # Connection timeout (includes DNS resolution)
            read=timeout,
            write=30.0,
        )
        self.use_dns_cache = use_dns_cache
        self.max_retries_server_error = max_retries_server_error
        self.retryable_server_errors = retryable_server_errors

        self.ssl_context = None
        if client_certificate is not None or trust_ca_path is not None:
            self.ssl_context = create_custom_ssl_context(
                client_certificate=client_certificate,
                trust_ca_path=trust_ca_path,
            )

    @contextlib.asynccontextmanager
    async def get_client(self) -> AsyncIterator[BackoffAsyncClient]:
        headers = self.headers.copy()
        verify = self.ssl_context if self.ssl_context else True
        async with get_httpx_client(
            use_dns_cache=self.use_dns_cache,
            headers=headers,
            timeout=self.timeout_config,
            verify=verify,
        ) as client:
            yield BackoffAsyncClient(
                client,
                credential=self.credential,
                max_retries_server_error=self.max_retries_server_error,
                retryable_server_errors=self.retryable_server_errors,
            )
