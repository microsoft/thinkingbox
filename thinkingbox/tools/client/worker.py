# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import copy
import os
from abc import abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Literal, Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool

from thinkingbox.common.utils import CredentialBase
from thinkingbox.tools.client.common import (
    MCPCredentialFactory,
    MCPProcessConfig,
    MCPRemoteConfig,
    MCPServerError,
    MultiToolsInterface,
    ToolServerConfigT,
    ToolsInterface,
)


@dataclass
class AddServersRequest:
    server_configs: list[ToolServerConfigT]


@dataclass
class ListToolsRequest:
    pass


@dataclass
class CallToolRequest:
    server_name: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class Task:
    request: AddServersRequest | ListToolsRequest | CallToolRequest
    future: asyncio.Future
    timeout: float


WORKER_TIMEOUT_MARGIN = 2.0


class MCPWorkerRef(MultiToolsInterface):
    def __init__(self, q: asyncio.Queue, task: asyncio.Task, timeout: float = 120.0):
        self.q = q
        self.task = task
        self._servers: list[str] = []
        self._timeout = timeout

    async def _run_task(self, request):
        task = Task(
            request=request,
            future=asyncio.get_event_loop().create_future(),
            timeout=self._timeout,
        )
        await self.q.put(task)

        try:
            w_timeout = self._timeout + WORKER_TIMEOUT_MARGIN
            result = await asyncio.wait_for(task.future, w_timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"_run_task request timed out after {task.timeout} seconds: {request}"
            ) from e
        if isinstance(result, asyncio.CancelledError):
            raise RuntimeError("The internal worker tasks was cancelled unexpectedly")
        elif isinstance(result, Exception):
            raise result
        return result

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        return await self._run_task(CallToolRequest(server_name, tool_name, arguments))

    async def list_tools(self) -> dict[str, dict[str, dict]]:
        return await self._run_task(ListToolsRequest())

    async def add_servers(self, server_configs: list[MCPProcessConfig]):
        res = await self._run_task(AddServersRequest(server_configs))
        for server_cfg in server_configs:
            self._servers.append(server_cfg.name)
        return res

    @property
    def servers(self):
        return self._servers

    async def stop(self):
        # Signal worker to stop
        await self.q.put(None)
        try:
            await asyncio.wait_for(
                self.task, timeout=self._timeout + WORKER_TIMEOUT_MARGIN
            )
        except asyncio.TimeoutError:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


def format_exception_group(exc_group: Exception) -> str:
    """
    Recursively formats all exceptions in an ExceptionGroup into a single error string.
    """
    messages = []

    def collect_messages(exc):
        if isinstance(exc, Exception):
            exceptions = getattr(exc, "exceptions", None)
            if isinstance(exceptions, (tuple, list)):
                for sub_exc in exceptions:
                    collect_messages(sub_exc)
                    return
        messages.append(f"{type(exc).__name__}: {exc}")

    collect_messages(exc_group)

    return "\n".join(f"[Exception {i + 1}] {msg}" for i, msg in enumerate(messages))


async def mcp_worker(q: asyncio.Queue):
    # need to keep everything in a single asyncio task because of the async context...
    exception_context = ""
    task = None
    try:
        async with AsyncExitStack() as exit_stack:
            mcp_client = MCPMultiClient(exit_stack)
            while True:
                task: Task | None = await q.get()
                if task is None:
                    break
                result = None
                exception_context = task.request

                if isinstance(task.request, AddServersRequest):
                    # if this fails, let the stack exit instead of catching here,
                    # or we don't get the exceptions.
                    # This is probably because of how the mcp client library handles
                    # its internal tasks...
                    result = await mcp_client.add_servers(task.request.server_configs)
                    if not task.future.done():
                        task.future.set_result(result)
                    continue

                try:
                    if isinstance(task.request, ListToolsRequest):
                        result = copy.deepcopy(await mcp_client.list_tools())
                    elif isinstance(task.request, CallToolRequest):
                        try:
                            result = await asyncio.wait_for(
                                mcp_client.call_tool(
                                    server_name=task.request.server_name,
                                    tool_name=task.request.tool_name,
                                    arguments=task.request.arguments,
                                ),
                                timeout=task.timeout,
                            )
                        except asyncio.TimeoutError:
                            result = TimeoutError(
                                f"Tool call timed out after {task.timeout}s"
                            )
                except Exception as e:
                    result = e
                if not task.future.done():
                    task.future.set_result(result)

    except Exception as e:
        # TODO should have an option to log or print the stack traces here
        # and for any failed request
        # import traceback
        # traceback.print_exc()
        msg = format_exception_group(e)
        if task is not None and not task.future.done():
            task.future.set_result(
                MCPServerError(
                    f"Errors encountered while processing request {exception_context!r}: {msg}"
                ),
            )


def jsonschema_remove_title_inplace(schema: dict):
    if "title" in schema:
        del schema["title"]
    properties: dict = schema.get("properties")
    if properties:
        for k, v in properties.items():
            if "title" in v:
                del v["title"]


def jsonschema_dereference(root: dict):
    def resolve_pointer(ptr: str):
        # ptr like '#/$defs/PaymentMethod'
        if not ptr.startswith("#/"):
            return None
        parts = ptr[2:].split("/")
        node = root
        for p in parts:
            # Unescape JSON Pointer tokens (~1 => /, ~0 => ~)
            p = p.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict):
                node = node.get(p)
            else:
                return None
        return node

    def walk(node, visiting=None):
        if visiting is None:
            visiting = set()

        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                if ref in visiting:
                    raise ValueError(f"Circular reference detected: {ref}")
                target = resolve_pointer(ref)
                if target is None:
                    raise ValueError(f"Missing definition for reference: {ref}")
                visiting.add(ref)
                # Deep copy target before recursion
                result = walk(copy.deepcopy(target), visiting)
                visiting.remove(ref)
                return result
            return {k: walk(v, visiting) for k, v in node.items() if k != "$ref"}
        if isinstance(node, list):
            return [walk(x, visiting) for x in node]
        return node

    return walk(root)


def convert_and_add_tools(out: list[dict], tools: list[Tool]):
    for tool in tools:
        input_schema = jsonschema_dereference(tool.inputSchema)
        input_schema.pop("$defs", None)
        jsonschema_remove_title_inplace(input_schema)
        if tool.name in out:
            raise MCPServerError(f"Duplicate tool {tool.name}")
        out[tool.name] = {
            "description": tool.description,
            "inputSchema": input_schema,
        }


class MCPSessionClientInterface(ToolsInterface):
    @abstractmethod
    async def enter_session(self, exit_stack: AsyncExitStack): ...

    @abstractmethod
    async def initialize(self): ...

    @property
    @abstractmethod
    def session(self) -> ClientSession: ...


class MCPProcessClient(MCPSessionClientInterface):
    def __init__(
        self,
        command: list[str],
        env: dict[str, str],
        cwd: str | None = None,
    ):
        # Initialize session and client objects
        self._session: Optional[ClientSession] = None
        self.command = command
        self.env = env
        self.cwd = cwd
        self.tools: dict[str, dict] = {}

    async def enter_session(self, exit_stack: AsyncExitStack):
        command = self.command[0]
        args = self.command[1:]
        env = os.environ.copy()
        env.update(self.env)
        env.setdefault("FASTMCP_LOG_LEVEL", "WARNING")
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=self.cwd,
        )

        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self._session = await exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

    async def initialize(self):
        await self.session.initialize()
        response = await self.session.list_tools()
        convert_and_add_tools(self.tools, response.tools)

    async def call_tool(self, tool_name: str, tool_args) -> CallToolResult:
        result = await self.session.call_tool(tool_name, tool_args)
        return result

    async def list_tools(self) -> dict[str, dict]:
        return self.tools

    @property
    def session(self):
        return self._session


class MCPRemoteClient(MCPSessionClientInterface):
    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        transport: Literal["sse", "stremable-http"] = "streamable-http",
        credential: CredentialBase | None = None,
    ):
        # Initialize session and client objects
        self._session: Optional[ClientSession] = None
        self.endpoint = endpoint
        self.headers = headers
        self.timeout = timeout
        self.transport = transport
        self.tools: dict[str, dict] = {}
        self.credential = credential

    @staticmethod
    def create_httpx_client(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {"follow_redirects": True}
        kwargs["timeout"] = httpx.Timeout(60.0) if timeout is None else timeout
        if headers is not None:
            kwargs["headers"] = headers
        if auth is not None:
            kwargs["auth"] = auth
        return httpx.AsyncClient(**kwargs)

    async def enter_session(self, exit_stack: AsyncExitStack):
        headers = self.headers.copy() if self.headers else {}
        if self.credential is not None:
            headers["Authorization"] = f"Bearer {await self.credential.get_token()}"

        if self.transport == "sse":
            (
                self.read_stream,
                self.write_stream,
            ) = await exit_stack.enter_async_context(
                sse_client(
                    url=self.endpoint,
                    headers=headers,
                    timeout=self.timeout,
                    sse_read_timeout=60 * 5,
                    httpx_client_factory=MCPRemoteClient.create_httpx_client,
                    auth=None,
                )
            )
        else:
            assert self.transport == "streamable-http"
            http_client = MCPRemoteClient.create_httpx_client(
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
            )
            (
                self.read_stream,
                self.write_stream,
                _,
            ) = await exit_stack.enter_async_context(
                streamable_http_client(
                    url=self.endpoint,
                    http_client=http_client,
                )
            )

        self._session = await exit_stack.enter_async_context(
            ClientSession(self.read_stream, self.write_stream)
        )

    async def initialize(self):
        await self.session.initialize()
        response = await self.session.list_tools()
        convert_and_add_tools(self.tools, response.tools)

    async def call_tool(self, tool_name: str, tool_args) -> CallToolResult:
        result = await self.session.call_tool(tool_name, tool_args)
        return result

    async def list_tools(self) -> dict[str, dict]:
        return self.tools

    @property
    def session(self):
        return self._session


class MCPMultiClient(MultiToolsInterface):
    def __init__(self, exit_stack: AsyncExitStack):
        self.exit_stack = exit_stack
        self.clients: dict[str, MCPSessionClientInterface] = {}
        # map(server_name, map(tool_name, tool_def))
        self.tools: dict[str, dict[str, dict]] = {}

    def create_session(self, cfg: ToolServerConfigT):
        if isinstance(cfg, MCPProcessConfig):
            return MCPProcessClient(
                command=cfg.command,
                env=cfg.env,
                cwd=cfg.cwd,
            )
        if isinstance(cfg, MCPRemoteConfig):
            return MCPRemoteClient(
                endpoint=cfg.endpoint,
                headers=cfg.headers,
                timeout=cfg.timeout,
                transport=cfg.transport,
                credential=MCPCredentialFactory.from_config(cfg.credential),
            )
        raise RuntimeError(f"Invalid configuration type: {type(cfg)!r}")

    async def add_servers(self, server_configs: list[ToolServerConfigT]) -> None:
        # can't enter the async context in a gather
        # and initialize() can take a significant portion of a second...
        mcp_clients: dict[str, MCPSessionClientInterface] = {}
        for cfg in server_configs:
            client = self.create_session(cfg)
            await client.enter_session(self.exit_stack)
            mcp_clients[cfg.name] = client

        await asyncio.gather(
            *[client.initialize() for client in mcp_clients.values()],
        )

        self.clients.update(mcp_clients)
        for server_name, client in mcp_clients.items():
            self.tools[server_name] = await client.list_tools()

    @property
    def servers(self):
        return list(self.clients)

    async def call_tool(
        self, server_name: str, tool_name: str, arguments
    ) -> CallToolResult:
        client = self.clients.get(server_name)
        if client is None:
            # this should not happen
            raise MCPServerError("Unknown Error")
        result = await client.call_tool(tool_name=tool_name, tool_args=arguments)
        return result

    async def list_tools(self):
        return self.tools
