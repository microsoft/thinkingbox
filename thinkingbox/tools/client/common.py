# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, Union

import yaml
from mcp.types import CallToolResult
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from thinkingbox.common.azure_interactive_browser_credential import (
    AzureInteractiveBrowserCredential,
)
from thinkingbox.common.utils import CredentialBase


class MCPServerError(Exception):
    pass


class InitResponse(NamedTuple):
    success: bool
    obj: dict


class ToolServerConfigBase(BaseModel):
    type: Literal[""] = ""
    name: str = ""


class MCPProcessConfig(ToolServerConfigBase):
    type: Literal["mcp-process"] = "mcp-process"
    command: list[str]
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    transport: Literal["stdio"] = "stdio"


class MCPRemoteCredentialConfigBase(BaseModel):
    requires_startup_authentication: ClassVar[bool] = False
    type: Literal[""] = ""


class MCPRemoteAzureInteractiveCredentialConfig(MCPRemoteCredentialConfigBase):
    requires_startup_authentication: ClassVar[bool] = True
    type: Literal["azure-interactive"] = "azure-interactive"
    scope: str | None = None
    client_id: str | None = None
    tenant_id: str | None = None
    purpose: str | None = None
    account_hint: str | None = None


MCPRemoteCredentialConfigT = Union[MCPRemoteAzureInteractiveCredentialConfig]


class MCPCredentialFactory:
    @staticmethod
    def from_config(
        credential_config: MCPRemoteCredentialConfigT | None,
    ) -> CredentialBase | None:
        if credential_config is None:
            return None
        if isinstance(credential_config, MCPRemoteAzureInteractiveCredentialConfig):
            return AzureInteractiveBrowserCredential(
                # todo name handling
                name=credential_config.purpose.replace(" ", "_").lower(),
                scope=credential_config.scope,
                purpose=credential_config.purpose,
                account_hint=credential_config.account_hint,
                client_id=credential_config.client_id,
                tenant_id=credential_config.tenant_id,
            )
        else:
            raise ValueError(
                f"Unsupported credential config type: {type(credential_config)}"
            )


class MCPRemoteConfig(ToolServerConfigBase):
    type: Literal["mcp-remote"] = "mcp-remote"
    endpoint: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 60.0
    transport: Literal["sse", "streamable-http"] = "streamable-http"
    credential: MCPRemoteCredentialConfigT | None = None


ToolServerConfigT = Union[MCPProcessConfig, MCPRemoteConfig]


class ServersConfig(BaseModel):
    # whether to automatically add thinkingbox/thinkingbox/tools/mcp_*.py
    use_internal_servers: bool = True
    servers: dict[str, ToolServerConfigT] = Field(default_factory=dict)


def load_servers_config(
    path: str | Path | None,
    internal_servers_root: str | Path | None,
):
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        config = ServersConfig(**config_dict)
    else:
        config = ServersConfig()

    if internal_servers_root is not None and config.use_internal_servers:
        prefix = "mcp_"
        suffix = ".py"
        for f in Path(internal_servers_root).iterdir():
            if f.is_file() and f.name.startswith(prefix) and f.name.endswith(suffix):
                server_name = f.name[len(prefix) : -len(suffix)]
                if server_name not in config.servers:
                    config.servers[server_name] = MCPProcessConfig(
                        command=[
                            sys.executable,
                            str(f.resolve()),
                        ]
                    )

    command_fmt_kwargs = {
        "python": sys.executable,
        "env": os.environ,
    }

    for server_name, server_cfg in config.servers.items():
        server_cfg.name = server_name
        if isinstance(server_cfg, MCPProcessConfig):
            for i in range(len(server_cfg.command)):
                part = server_cfg.command[i]
                server_cfg.command[i] = part.format(**command_fmt_kwargs)

    return config


class ToolsInterface(ABC):
    @abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> CallToolResult: ...

    @abstractmethod
    async def list_tools(self) -> dict[str, dict]: ...


class MultiToolsInterface(ABC):
    @abstractmethod
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult: ...

    @abstractmethod
    async def list_tools(self) -> dict[str, dict[str, dict]]: ...

    @property
    @abstractmethod
    def servers(self) -> list[str]: ...


class ServerToolCall(BaseModel):
    server_name: str
    tool_name: str
    arguments: dict[str, Any]


class ToolInfo(NamedTuple):
    client_index: int
    server_name: str
    tool_def: dict


class ServerInfo(NamedTuple):
    client_index: int
    tools: dict[str, dict]


class ToolDispatcher(ToolsInterface):
    RESERVED_SERVER_TOOL = "__reserved__server_tool"

    def __init__(self):
        self.clients: list[MultiToolsInterface] = []
        self.tools: dict[str, ToolInfo] = {}
        self.server_info: dict[str, ServerInfo] = {}

    async def add(
        self,
        client: MultiToolsInterface,
        server_to_priority: dict[str, int],
        visible_tools: set[str] | None = None,
    ):
        index = len(self.clients)
        multi_tools = await client.list_tools()
        self.clients.append(client)
        for server_name, tools in multi_tools.items():
            self.server_info[server_name] = ServerInfo(
                client_index=index,
                tools=tools,
            )
            priority = server_to_priority.get(server_name, 0)
            for tool_name, tool_def in tools.items():
                if (visible_tools is not None) and (tool_name not in visible_tools):
                    # not in list_tools and can't be used in call_tool
                    continue

                existing_tool = self.tools.get(tool_name)
                if existing_tool is None:
                    # if tool does not exist yet, just add it
                    self.tools[tool_name] = ToolInfo(
                        client_index=index,
                        server_name=server_name,
                        tool_def=tool_def,
                    )
                else:
                    # insert only if it should take priority
                    if priority > server_to_priority.get(existing_tool.server_name, 0):
                        self.tools[tool_name] = ToolInfo(
                            client_index=index,
                            server_name=server_name,
                            tool_def=tool_def,
                        )

    def get_server_tool_def(self, server_name: str, tool_name: str) -> dict | None:
        server_info = self.server_info.get(server_name)
        if server_info is not None:
            return server_info.tools.get(tool_name)
        return None

    async def call_server_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        info = self.server_info.get(server_name)
        if info is None or (tool_name not in info.tools):
            raise MCPServerError(f"Tool {server_name} > {tool_name} does not exist")
        client = self.clients[info.client_index]
        return await client.call_tool(server_name, tool_name, arguments)

    async def _handle_server_tool_call(self, arguments: dict[str, Any]) -> str:
        tc = ServerToolCall(**arguments)
        return await self.call_server_tool(tc.server_name, tc.tool_name, tc.arguments)

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        if tool_name == ToolDispatcher.RESERVED_SERVER_TOOL:
            return await self._handle_server_tool_call(arguments)
        info = self.tools.get(tool_name)
        if info is None:
            raise MCPServerError(f"Tool {tool_name} does not exist")
        client = self.clients[info.client_index]
        return await client.call_tool(info.server_name, tool_name, arguments)

    async def list_tools(self) -> dict[str, dict]:
        return {name: info.tool_def for name, info in self.tools.items()}


def success_response(**kwargs) -> str:
    obj = {
        "status": "ok",
        **to_jsonable_python(kwargs),
    }
    return json.dumps(obj)


def error_response(exc) -> str:
    traceback.print_exc()
    return "Error!\n" + str(exc)


def parse_init_response(text: str):
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return InitResponse(success=False, obj={})
    if not isinstance(obj, dict):
        return InitResponse(success=False, obj={})
    if obj == {}:
        # for backwards compatibility
        return InitResponse(success=True, obj={})
    status = obj.pop("status", None)
    if status == "ok":
        return InitResponse(
            success=True,
            obj=obj,
        )
    return InitResponse(success=False, obj={})
