# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
import json
import uuid
from typing import Any

import httpx

from thinkingbox.common.chat_types import ToolDef
from thinkingbox.common.config_types import SessionProxyConfig
from thinkingbox.common.credential_factory import create_credential
from thinkingbox.common.http_client import (
    BackoffAsyncClient,
    create_custom_ssl_context,
    get_httpx_client,
)
from thinkingbox.common.utils import (
    CredentialBase,
    raise_for_status_with_error,
)


class MCPProxyClient:
    def __init__(
        self,
        endpoint: str,
        session_id: str | None = None,
        timeout: float = 120.0,
        use_dns_cache: bool = False,
        credential: CredentialBase | None = None,
        client_certificate: str | None = None,
        trust_ca_path: str | None = None,
        headers: dict[str, str] | None = None,
        max_retries_server_error: int = 5,
        retryable_server_errors: tuple[int | str, ...] = (502, 503),
        always_json_output: bool = False,
        geteffects_proxy_info: bool = False,
    ):
        self.endpoint = endpoint
        self.credential = credential
        self.session_id = session_id if session_id is not None else str(uuid.uuid4())
        self.timeout = timeout
        self.timeout_config = httpx.Timeout(
            timeout=timeout,
            connect=30.0,
            read=timeout,
            write=30.0,
        )
        self.use_dns_cache = use_dns_cache
        self.headers = headers
        self.max_retries_server_error = max_retries_server_error
        self.retryable_server_errors = retryable_server_errors
        self.always_json_output = always_json_output
        self.geteffects_proxy_info = geteffects_proxy_info
        self.ssl_context = None
        if client_certificate is not None or trust_ca_path is not None:
            self.ssl_context = create_custom_ssl_context(
                client_certificate=client_certificate,
                trust_ca_path=trust_ca_path,
            )

    @contextlib.asynccontextmanager
    async def _get_http_client(self):
        verify = self.ssl_context if self.ssl_context else True
        async with get_httpx_client(
            use_dns_cache=self.use_dns_cache,
            timeout=self.timeout_config,
            headers=self.headers,
            verify=verify,
        ) as client:
            yield BackoffAsyncClient(
                client,
                credential=self.credential,
                max_retries_server_error=self.max_retries_server_error,
                retryable_server_errors=self.retryable_server_errors,
                # on timeout, the operation might have happened on the server.
                # Do not re-try.
                max_retries_timeout=0,
            )

    async def session_create(
        self,
        server_config: dict[str, dict],
        available_tools: list[str],
    ) -> dict:
        url = f"{self.endpoint}/session_create"
        payload = {
            "session_id": self.session_id,
            "server_config": server_config,
            "available_tools": available_tools,
            # Give the server a bit less time than the client timeout with default and pegging
            "timeout": max(self.timeout * 0.9, self.timeout - 30.0),
        }
        if self.always_json_output:
            payload["always_json_output"] = True
        if self.geteffects_proxy_info:
            payload["geteffects_proxy_info"] = True

        async with self._get_http_client() as client:
            response = await client.post(url, json=payload)
        raise_for_status_with_error(response)
        out = response.json()
        return out["content"]["value"]

    async def session_destroy(self):
        url = f"{self.endpoint}/session_destroy"
        payload = {
            "session_id": self.session_id,
        }

        async with self._get_http_client() as client:
            response = await client.post(url, json=payload)
        raise_for_status_with_error(response)

    async def session_destroy_all(self):
        url = f"{self.endpoint}/session_destroy_all"

        async with self._get_http_client() as client:
            response = await client.post(url, json={})
        raise_for_status_with_error(response)

    async def session_info(
        self, update: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.endpoint}/session_info"
        payload = {
            "session_id": self.session_id,
            "update": update or {},
        }
        async with self._get_http_client() as client:
            response = await client.post(url, json=payload)
        raise_for_status_with_error(response)
        out = response.json()
        return out["content"]["value"]

    async def call_tool(self, tool_name, /, **kwargs):
        url = f"{self.endpoint}/call_tool"
        payload = {
            "session_id": self.session_id,
            "tool_name": tool_name,
            "arguments": kwargs,
        }
        async with self._get_http_client() as client:
            response = await client.post(url, json=payload)
        raise_for_status_with_error(response)
        out = response.json()
        return out["content"]["value"]

    async def list_tools(self) -> list[ToolDef]:
        url = f"{self.endpoint}/list_tools"
        payload = {
            "session_id": self.session_id,
        }
        async with self._get_http_client() as client:
            response = await client.post(url, json=payload)
        raise_for_status_with_error(response)
        out = response.json()
        tools_dict = out["content"]["value"]["tools"]
        tools_list = []
        for name, schema in tools_dict.items():
            tools_list.append(
                ToolDef(
                    name=name,
                    description=schema["description"] or "",
                    input_schema=schema["inputSchema"],
                )
            )
        return tools_list

    async def get_effects(self) -> dict[str, Any]:
        url = f"{self.endpoint}/get_effects"
        payload = {
            "session_id": self.session_id,
        }
        async with self._get_http_client() as client:
            response = await client.post(url, json=payload)
        raise_for_status_with_error(response)
        out = response.json()
        return out["content"]["value"]

    @staticmethod
    @contextlib.asynccontextmanager
    async def session_context(
        endpoint: str,
        server_config: dict[str, dict],
        available_tools: list[str],
        session_id: str | None = None,
        timeout: float = 120.0,
        use_dns_cache: bool = False,
        credential: CredentialBase | None = None,
        client_certificate: str | None = None,
        trust_ca_path: str | None = None,
        headers: dict[str, str] | None = None,
        max_retries_server_error: int = 5,
        retryable_server_errors: tuple[int | str, ...] = (502, 503),
        **kwargs,
    ):
        mcp = MCPProxyClient(
            endpoint,
            session_id=session_id,
            timeout=timeout,
            use_dns_cache=use_dns_cache,
            credential=credential,
            client_certificate=client_certificate,
            trust_ca_path=trust_ca_path,
            headers=headers,
            max_retries_server_error=max_retries_server_error,
            retryable_server_errors=retryable_server_errors,
            **kwargs,
        )
        init_result = await mcp.session_create(server_config, available_tools)
        context = MCPProxyContext(client=mcp, init_result=init_result)
        try:
            yield context
        finally:
            if not context.should_linger:
                await mcp.session_destroy()

    @staticmethod
    @contextlib.asynccontextmanager
    async def session_context_from_config(
        config: SessionProxyConfig,
        server_config: dict[str, dict],
        available_tools: list[str],
    ):
        credential = None
        if config.credential is not None:
            credential = create_credential(config.credential)
        async with MCPProxyClient.session_context(
            endpoint=config.endpoint_url,
            timeout=config.timeout,
            use_dns_cache=config.use_dns_cache,
            credential=credential,
            client_certificate=config.client_certificate,
            trust_ca_path=config.trust_ca_path,
            headers=config.headers,
            max_retries_server_error=config.max_retries_server_error,
            retryable_server_errors=config.retryable_server_errors,
            server_config=server_config,
            available_tools=available_tools,
            always_json_output=config.always_json_output,
            geteffects_proxy_info=config.geteffects_proxy_info,
        ) as context:
            yield context


class MCPProxyContext:
    def __init__(
        self,
        client: MCPProxyClient,
        init_result: dict | None = None,
    ):
        self.client = client
        self.session_id = client.session_id
        self._init_result: dict = init_result or {}
        self._linger = False

    def linger(self):
        self._linger = True

    @property
    def should_linger(self):
        return self._linger

    @property
    def init_result(self):
        return self._init_result

    async def call_tool(self, tool_name, /, **kwargs):
        return await self.client.call_tool(tool_name, **kwargs)

    async def list_tools(self) -> list[ToolDef]:
        return await self.client.list_tools()

    async def get_effects(self) -> dict[str, Any]:
        return await self.client.get_effects()


class SessionClientFixture:
    def __init__(
        self,
        endpoint: str,
        timeout: float = 120.0,
    ):
        """Synchronous client to be used as fixture"""
        self.endpoint = endpoint
        self.session_id: str | None = None
        self.timeout_config = httpx.Timeout(
            timeout=timeout,
            connect=30.0,
            read=timeout,
            write=30.0,
        )

    def set_session(self, session_id: str):
        self.session_id = session_id

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ) -> str:
        """Call a tool and return the response string"""
        if self.session_id is None:
            raise RuntimeError("SessionClientFixture: Session ID was not set")
        url = f"{self.endpoint}/call_tool"
        payload = {
            "session_id": self.session_id,
            "tool_name": "__reserved__server_tool",
            "arguments": {
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments,
            },
        }
        # TODO consider reusing the wrapper in http_client.py
        with httpx.Client(timeout=self.timeout_config) as http_client:
            response = http_client.post(url, json=payload)
        raise_for_status_with_error(response)
        out = response.json()
        return out["content"]["value"]

    def call_json_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ) -> dict | list:
        """Call a tool and return the response parsed as JSON"""
        result = self.call_tool(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
        )
        out = json.loads(result)
        if not isinstance(out, (list, dict)):
            raise ValueError(f"Result is not dict or list: {result!r}")
        return out
