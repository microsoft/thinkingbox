#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

"""
A server that allows creating / using / destroying MCP server sessions

GET /health
    Health check endpoint for Azure probes

POST /session_create
    Create a new session with a given set of tools and initial configurations,
    this spawns MCP server processes

POST /session_destroy
    Destroy a session,
    this terminates the associated MCP processes

POST /session_destroy_all
    Destroy all the sessions

POST /session_info
    Get / Update session info

POST /get_effects
    Get a JSON object containing the state or list of side effects

POST /call_tool
    Call a tool with arguments in a MCP server session

POST /list_tools
    List tools in a MCP server session

POST /garbage_collect
    Destroy inactive sessions

POST /mcp
    MCP streamable-http endpoint.
    Supports tools/list and tools/call methods.
    Forwards requests to the session in header X-TB-Session-Id.
"""

import asyncio
import datetime
import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple

import click
import uvicorn
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware as FastMCPMiddleware
from fastmcp.tools.tool import Tool, ToolResult
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import to_jsonable_python
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route

from thinkingbox.common.utils import syncify
from thinkingbox.tools.client.common import (
    MCPCredentialFactory,
    MCPRemoteConfig,
    MCPServerError,
    ServersConfig,
    ToolDispatcher,
    ToolServerConfigT,
    load_servers_config,
    parse_init_response,
)
from thinkingbox.tools.client.worker import (
    MCPWorkerRef,
    mcp_worker,
)

INTERNAL_MCP_SERVERS_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

TB_SESSION_ID_HEADER = "X-TB-Session-Id"


def datetime_now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


class AppConfig(BaseModel):
    log_status_task_interval_seconds: float = 5.0
    gc_enable: bool = False
    gc_minutes: float = 15.0
    gc_task_interval_seconds: float = 60.0
    servers_config: ServersConfig = Field(default_factory=ServersConfig)
    require_auth: bool = False
    api_key: str | None = None


class SessionCreateRequest(BaseModel):
    session_id: str
    server_config: dict[str, dict]
    available_tools: list[str]
    timeout: float = 120.0
    always_json_output: bool = False
    geteffects_proxy_info: bool = False


class SessionDestroyRequest(BaseModel):
    session_id: str


class SessionDestroyAllRequest(BaseModel):
    pass


class SessionInfoRequest(BaseModel):
    session_id: str
    update: dict[str, Any] = Field(default_factory=dict)


class CallToolRequest(BaseModel):
    session_id: str
    tool_name: str
    arguments: dict[str, Any]


class GetEffectsRequest(BaseModel):
    session_id: str


class ListToolsRequest(BaseModel):
    session_id: str


class GarbageCollectRequest(BaseModel):
    inactivity_minutes: float


class ToolDispatcherExt(ToolDispatcher):
    async def _call_reserved_tool_if_exists(
        self,
        server_name: str,
        tool_name: str,
        session_id: str,
        arguments: dict,
    ):
        tool_def = self.get_server_tool_def(server_name, tool_name)
        if tool_def is None:
            return None
        tool_args = []
        try:
            tool_args.extend(list(tool_def["inputSchema"]["properties"]))
        except (ValueError, KeyError, TypeError):
            # no arguments
            pass
        arguments = arguments.copy()
        if "session_id" in tool_args:
            arguments["session_id"] = session_id
        res = await self.call_server_tool(
            server_name,
            tool_name,
            arguments,
        )
        logger.debug("%s.%s -> %s", server_name, tool_name, res.content[0].text)
        return res

    async def init(self, server_name: str, session_id: str, config: dict) -> dict:
        res = await self._call_reserved_tool_if_exists(
            server_name=server_name,
            tool_name="__reserved__init",
            session_id=session_id,
            arguments={
                "config": config,
            },
        )
        if res is None:
            # __reserved__init does not exist
            return {}

        init_response = parse_init_response(res.content[0].text)
        if not init_response.success:
            response_text = str(res.content[0].text)
            raise MCPServerError(f"Error calling {server_name} init: {response_text}")
        return init_response.obj

    async def get_effects(self, server_name: str, session_id: str) -> dict:
        res = await self._call_reserved_tool_if_exists(
            server_name=server_name,
            tool_name="__reserved__geteffects",
            session_id=session_id,
            arguments={},
        )
        effects_obj = {}
        if res is not None:
            try:
                obj = json.loads(res.content[0].text)
                if isinstance(obj, dict):
                    effects_obj = obj
            except (TypeError, ValueError) as e:
                logger.error(f"Error calling {server_name} geteffects: {e}")
        return effects_obj

    async def teardown(self, server_name: str, session_id: str) -> None:
        _ = await self._call_reserved_tool_if_exists(
            server_name=server_name,
            tool_name="__reserved__teardown",
            session_id=session_id,
            arguments={},
        )


class SessionProxyMiddleware(FastMCPMiddleware):
    """
    Middleware that routes MCP requests to session-isolated tool dispatchers.
    """

    async def on_list_tools(self, context, call_next):
        headers = get_http_headers(include_all=True)
        session_id = headers.get(TB_SESSION_ID_HEADER.lower())

        if not session_id:
            raise MCPServerError(f"Missing {TB_SESSION_ID_HEADER} header")

        async with sessions_lock:
            session = sessions.get(session_id)

        if session is None:
            raise MCPServerError(f"Session not found: {session_id}")

        tools = []
        for name, tool_def in session.tools.items():
            tools.append(
                Tool(
                    name=name,
                    description=tool_def.get("description", ""),
                    parameters=tool_def.get("inputSchema", {}),
                )
            )

        return tools

    async def on_call_tool(self, context, call_next):
        headers = get_http_headers(include_all=True)
        session_id = headers.get(TB_SESSION_ID_HEADER.lower())

        if not session_id:
            raise MCPServerError(f"Missing {TB_SESSION_ID_HEADER} header")

        async with sessions_lock:
            session = sessions.get(session_id)

        if session is None:
            raise MCPServerError(f"Session not found: {session_id}")

        result = await session.call_tool(
            context.message.name, context.message.arguments
        )

        structured = result.structuredContent
        if structured is None and result.content:
            # Synthesize structuredContent from JSON text content
            try:
                text = result.content[0].text
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    structured = parsed
            except (AttributeError, IndexError, ValueError, TypeError):
                pass

        return ToolResult(content=result.content, structured_content=structured)


class Session:
    def __init__(
        self,
        session_id: str,
        timeout: float = 120.0,
        enable_geteffects_proxy_info: bool = False,
        always_json_output: bool = False,
    ):
        self.session_id: str = session_id
        self.timeout: float = timeout
        self.enable_geteffects_proxy_info = enable_geteffects_proxy_info
        self.always_json_output = always_json_output
        self.tools: dict[str, dict] = {}
        mcp_worker_queue = asyncio.Queue()
        mcp_worker_task = asyncio.create_task(mcp_worker(mcp_worker_queue))
        self.mcp_worker = MCPWorkerRef(
            mcp_worker_queue, mcp_worker_task, timeout=timeout
        )
        self.dispatcher = ToolDispatcherExt()
        self.initialized = False
        self.servers = []
        self.last_event = datetime_now_utc()
        self.tool_call_log: list[dict] = []
        self.info: dict[str, Any] = {}

    async def initialize(
        self,
        init_config: dict[str, dict],
        available_tools: list[str],
    ) -> dict[str, dict]:
        self.last_event = datetime_now_utc()
        available_tools_set = set(available_tools)
        init_result: dict[str, dict] = {}

        server_to_priority: dict[str, int] = {}
        server_configs: list[ToolServerConfigT] = []
        for i, server_name in enumerate(init_config.keys()):
            mcp_config = app_config.servers_config.servers.get(server_name)
            if mcp_config is None:
                raise MCPServerError(f"Server {server_name} does not exist")
            server_configs.append(mcp_config)
            server_to_priority[server_name] = i

        # batch initialize the process server+clients and
        # remote clients in the worker task
        await self.mcp_worker.add_servers(server_configs)
        await self.dispatcher.add(
            self.mcp_worker,
            server_to_priority=server_to_priority,
            visible_tools=available_tools_set,
        )

        # check that all available tools were found
        self.tools.update(await self.dispatcher.list_tools())
        for tool_name in available_tools:
            if tool_name not in self.tools:
                raise MCPServerError(f"Tool {tool_name} does not exist")

        # call the init function
        for server_name, init_dict in init_config.items():
            self.servers.append(server_name)
            init_result[server_name] = await self.dispatcher.init(
                server_name,
                session_id=self.session_id,
                config=init_dict,
            )
        self.initialized = True
        self.last_event = datetime_now_utc()
        return init_result

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        self.last_event = datetime_now_utc()
        # Use the session's configured timeout
        res = await self.dispatcher.call_tool(tool_name, arguments)
        if self.always_json_output:
            if res.content and hasattr(res.content[0], "text"):
                text = res.content[0].text
                try:
                    parsed = json.loads(text)
                except (ValueError, TypeError):
                    parsed = {"result": text}
                    res.content = [TextContent(type="text", text=json.dumps(parsed))]
                res.structuredContent = parsed
        if self.enable_geteffects_proxy_info:
            self.tool_call_log.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "response": res.content[0].text,
                }
            )
        self.last_event = datetime_now_utc()
        return res

    async def get_effects(self):
        self.last_event = datetime_now_utc()
        out = {}
        for server_name in self.servers:
            effects_obj = await self.dispatcher.get_effects(
                server_name,
                session_id=self.session_id,
            )
            out[server_name] = effects_obj
        if self.enable_geteffects_proxy_info:
            out["__reserved__proxy_info"] = {
                "tool_calls": self.tool_call_log,
            }
        self.last_event = datetime_now_utc()
        return out

    async def destroy(self):
        self.last_event = datetime_now_utc()
        for server_name in self.servers:
            try:
                await self.dispatcher.teardown(
                    server_name,
                    session_id=self.session_id,
                )
            except Exception as e:
                logger.warning(
                    "(session %s) Failed to call %s.teardown: %s",
                    self.session_id,
                    server_name,
                    e,
                )
        await self.mcp_worker.stop()


# Global dictionary to store sessions
sessions: Dict[str, Session] = {}
sessions_lock = asyncio.Lock()  # Lock to protect concurrent access to sessions


class StringContent(BaseModel):
    type: Literal["string"]
    value: str


class ObjContent(BaseModel):
    type: Literal["object"]
    value: dict


class SessionProxyResponse(BaseModel):
    success: bool
    content: StringContent | ObjContent

    def get_error_text(self) -> str:
        if self.success:
            raise ValueError("There is no error, success=True")
        return str(self.content.value)

    def get_text(self) -> str:
        if not self.success:
            raise ValueError("There is no text, success=False")
        if not isinstance(self.content, StringContent):
            raise ValueError("There is no text, content.type!=string")
        return self.content.value

    def get_object(self) -> dict:
        if not self.success:
            raise ValueError("There is no object, success=False")
        if not isinstance(self.content, ObjContent):
            raise ValueError("There is no object, content.type!=object")
        return self.content.value


class ServerResponse(NamedTuple):
    data: SessionProxyResponse
    status_code: int = 200

    @classmethod
    def success_text(cls, text: str) -> ServerResponse:
        return cls(
            SessionProxyResponse(
                success=True, content=StringContent(type="string", value=text)
            )
        )

    @classmethod
    def success_obj(cls, obj: dict) -> ServerResponse:
        return cls(
            SessionProxyResponse(
                success=True, content=ObjContent(type="object", value=obj)
            )
        )

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(to_jsonable_python(self.data), status_code=self.status_code)


class ServerError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def _handle(request, model_cls, inner_fn):
    """Handler wrapper: parse request, call inner function, serialize response."""
    try:
        try:
            data = model_cls(**(await request.json()))
        except ValidationError:
            raise ServerError("Invalid input")
        result = await inner_fn(data)
    except ServerError as e:
        resp = SessionProxyResponse(
            success=False, content=StringContent(type="string", value=e.message)
        )
        return JSONResponse(to_jsonable_python(resp), status_code=e.status_code)
    return result.to_json_response()


async def _session_create_inner(data: SessionCreateRequest) -> ServerResponse:
    async with sessions_lock:
        if data.session_id in sessions:
            raise ServerError("Session already exists")
        new_session = Session(
            session_id=data.session_id,
            timeout=data.timeout,
            enable_geteffects_proxy_info=data.geteffects_proxy_info,
            always_json_output=data.always_json_output,
        )
        sessions[data.session_id] = new_session

    init_result: dict[str, dict] = {}
    init_error = None
    # try to initialize it
    try:
        init_result = await new_session.initialize(
            init_config=data.server_config, available_tools=data.available_tools
        )
    except Exception as e:
        init_error = str(e)

    if init_error is not None:
        # something went wrong, destroy the session and respond with an error
        # suppress exceptions here or they will stop the original
        # one from propagating to the response

        async with sessions_lock:
            try:
                del sessions[data.session_id]
            except KeyError:
                pass

        try:
            await new_session.destroy()
        except Exception as e:
            logger.warning(
                "Error deleting new session %s which had failed to initialize: %s",
                data.session_id,
                e,
            )
        raise ServerError(init_error)

    return ServerResponse.success_obj(init_result)


async def session_create(request):
    return await _handle(request, SessionCreateRequest, _session_create_inner)


async def _session_info_inner(data: SessionInfoRequest) -> ServerResponse:
    async with sessions_lock:
        session = sessions.get(data.session_id)
    if session is None:
        raise ServerError("Session does not exist")
    if data.update:
        session.info.update(data.update)
    return ServerResponse.success_obj(session.info)


async def session_info(request):
    return await _handle(request, SessionInfoRequest, _session_info_inner)


async def _session_destroy_inner(data: SessionDestroyRequest) -> ServerResponse:
    async with sessions_lock:
        try:
            to_destroy = sessions[data.session_id]
            del sessions[data.session_id]
        except KeyError:
            raise ServerError("Session does not exist")
    await to_destroy.destroy()
    return ServerResponse.success_text("Ok")


async def session_destroy(request):
    return await _handle(request, SessionDestroyRequest, _session_destroy_inner)


async def _session_destroy_all_inner(data: SessionDestroyAllRequest) -> ServerResponse:
    to_destroy: list[Session] = []
    async with sessions_lock:
        for s in sessions.values():
            to_destroy.append(s)
        sessions.clear()
    for s in to_destroy:
        await s.destroy()
    return ServerResponse.success_text("Ok")


async def session_destroy_all(request):
    return await _handle(request, SessionDestroyAllRequest, _session_destroy_all_inner)


async def _call_tool_inner(data: CallToolRequest) -> ServerResponse:
    async with sessions_lock:
        session = sessions.get(data.session_id)
    if session is None:
        raise ServerError("Session does not exist")
    try:
        tool_resp = await session.call_tool(data.tool_name, data.arguments)
    except TimeoutError:
        raise ServerError(f"Tool call timed out after {session.timeout} seconds", 504)
    except Exception as e:
        raise ServerError(str(e))
    return ServerResponse.success_text(tool_resp.content[0].text)


async def call_tool(request):
    return await _handle(request, CallToolRequest, _call_tool_inner)


async def _get_effects_inner(data: GetEffectsRequest) -> ServerResponse:
    async with sessions_lock:
        session = sessions.get(data.session_id)
    if session is None:
        raise ServerError("Session does not exist")
    effects = await session.get_effects()
    return ServerResponse.success_obj(effects)


async def get_effects(request):
    return await _handle(request, GetEffectsRequest, _get_effects_inner)


async def _list_tools_inner(data: ListToolsRequest) -> ServerResponse:
    async with sessions_lock:
        session = sessions.get(data.session_id)
    if session is None:
        raise ServerError("Session does not exist")
    return ServerResponse.success_obj({"tools": session.tools})


async def list_tools(request):
    return await _handle(request, ListToolsRequest, _list_tools_inner)


async def _garbage_collect_inner(data: GarbageCollectRequest) -> ServerResponse:
    session_ids = await _destroy_inactive_sessions(minutes=data.inactivity_minutes)
    return ServerResponse.success_obj(
        {
            "count": len(session_ids),
            "session_ids": list(session_ids),
        }
    )


async def garbage_collect(request):
    return await _handle(request, GarbageCollectRequest, _garbage_collect_inner)


async def health_check(request: Request):
    """Health check endpoint"""
    return PlainTextResponse("OK")


async def _destroy_inactive_sessions(minutes: float) -> list[str]:
    dt_threshold = datetime_now_utc() - datetime.timedelta(minutes=minutes)
    to_destroy: list[tuple[str, Session | None]] = []
    async with sessions_lock:
        for s_id, s in sessions.items():
            if s.last_event < dt_threshold:
                to_destroy.append((s_id, s))
        for s_id, _ in to_destroy:
            # remove from the global dict while holding the lock
            del sessions[s_id]

    # actually destroy the detached sessions
    for i in range(len(to_destroy)):
        s_id, s = to_destroy[i]
        to_destroy[i] = (s_id, None)
        if s is None:
            logger.error(f"Session {s_id} is None and cannot be destroyed.")
            continue
        try:
            await s.destroy()
        except Exception:
            pass

    # return ids of destroyed sessions
    return [s_id for (s_id, _) in to_destroy]


mcp = FastMCP("SessionProxy")
mcp.add_middleware(SessionProxyMiddleware())
mcp_app = mcp.http_app(path="/")

# Define routes
routes = [
    # Existing ThinkingBox endpoints
    Route("/session_create", session_create, methods=["POST"]),
    Route("/session_info", session_info, methods=["POST"]),
    Route("/session_destroy", session_destroy, methods=["POST"]),
    Route("/session_destroy_all", session_destroy_all, methods=["POST"]),
    Route("/call_tool", call_tool, methods=["POST"]),
    Route("/get_effects", get_effects, methods=["POST"]),
    Route("/list_tools", list_tools, methods=["POST"]),
    Route("/garbage_collect", garbage_collect, methods=["POST"]),
    Route("/health", health_check, methods=["GET"]),
    Mount("/mcp", mcp_app),
]


class LoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        log_tag = getattr(request.state, "log_tag", None)
        log_tag_text = f" [{log_tag}]" if log_tag else ""
        logger.info(
            f"{request.method} {request.url.path}{log_tag_text} {response.status_code} ({duration:.2f} s)"
        )
        return response


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Bearer token authentication middleware.

    When require_auth is enabled in app_config, requires Authorization: Bearer <key>
    header on all requests except /health endpoint. When require_auth is disabled,
    all requests are allowed (local dev).
    """

    async def dispatch(self, request, call_next):
        # Skip auth for health checks
        if request.url.path == "/health":
            return await call_next(request)

        if app_config.require_auth:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    {"success": False, "error": "Unauthorized"}, status_code=401
                )
            token = auth_header[len("Bearer ") :]
            if token != app_config.api_key:
                return JSONResponse(
                    {"success": False, "error": "Unauthorized"}, status_code=401
                )

        return await call_next(request)


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager that initializes FastMCP and background tasks."""
    async with mcp_app.lifespan(app):
        # Start background tasks
        asyncio.create_task(
            log_status_every(app_config.log_status_task_interval_seconds)
        )
        if app_config.gc_enable:
            asyncio.create_task(gc_task(app_config.gc_task_interval_seconds))
        yield


async def log_status_every(seconds):
    num_sessions_prev = None
    while True:
        await asyncio.sleep(seconds)
        async with sessions_lock:
            num_sessions = len(sessions)
        if num_sessions != num_sessions_prev:
            logger.info(f"Session Count {num_sessions}")
            num_sessions_prev = num_sessions


async def gc_task(seconds):
    while True:
        await asyncio.sleep(seconds)
        if not app_config.gc_enable:
            continue
        try:
            session_ids = await _destroy_inactive_sessions(app_config.gc_minutes)
            if session_ids:
                logger.info("gc_task: removed sessions %r", session_ids)
            else:
                logger.info("gc_task: no sessions to remove")
        except Exception:
            logger.warning(
                "gc_task: error in _destroy_inactive_sessions: %s",
                traceback.format_exc(),
            )


def create_app(extra_routes=None):
    all_routes = list(routes)
    if extra_routes:
        all_routes.extend(extra_routes)
    return Starlette(
        debug=False,
        routes=all_routes,
        lifespan=lifespan,
        middleware=[
            Middleware(BearerAuthMiddleware),
            Middleware(LoggerMiddleware),
        ],
    )


app: Starlette | None = None
app_config = AppConfig()


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    httpx_log = logging.getLogger("httpx")
    httpx_log.setLevel(logging.WARN)


@click.command()
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host interface to bind the server.",
)
@click.option(
    "--port",
    type=int,
    default=7111,
    show_default=True,
    help="Port to bind the server.",
)
@click.option(
    "--gc-enable",
    is_flag=True,
    help="Enable auto-deleting inactive sessions.",
)
@click.option(
    "--gc-minutes",
    type=float,
    default=15.0,
    show_default=True,
    help="Inactivity time before auto-deleting sessions (minutes).",
)
@click.option(
    "--servers",
    type=click.Path(path_type=Path, exists=False),
    default=None,
    help="Path to servers configuration file.",
)
@click.option(
    "--require-auth",
    is_flag=True,
    help="Require Bearer token authentication on all requests (except /health).",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="API key for Bearer authentication.",
)
@click.option(
    "--ssl-pem",
    type=click.Path(exists=True),
    default=None,
    help="Path to PEM file containing both SSL certificate and private key for HTTPS.",
)
@click.option(
    "--bd-connectors",
    is_flag=True,
    help="Enable BotDesigner connector routes under /connectors/.",
)
def mcp_start(
    host: str,
    port: int,
    gc_enable: bool,
    gc_minutes: float,
    servers: Path | None,
    require_auth: bool,
    api_key: str | None,
    ssl_pem: str | None,
    bd_connectors: bool,
) -> None:
    """
    Start the MCP server.
    """
    global app

    setup_logging()
    os.environ.setdefault("FASTMCP_LOG_LEVEL", "WARNING")

    # Validate auth configuration
    if not require_auth and api_key is not None:
        raise click.UsageError(
            "--api-key cannot be set when --require-auth is not enabled."
        )

    if require_auth:
        if api_key is None:
            api_key = os.environ.get("THINKINGBOX_SESSION_PROXY_KEY", "").strip()
        if not api_key:
            raise click.UsageError(
                "--require-auth is enabled but no API key provided. "
                "Set --api-key or the THINKINGBOX_SESSION_PROXY_KEY environment variable."
            )
        app_config.require_auth = True
        app_config.api_key = api_key
        logger.info("Authentication is enabled and required")

    tb_data = os.environ.get("THINKINGBOX_DATA")
    if not tb_data or not os.path.isdir(tb_data):
        logger.warning(
            "THINKINGBOX_DATA not defined or does not exist, some tools may not work properly"
        )

    app_config.gc_enable = gc_enable
    app_config.gc_minutes = gc_minutes
    app_config.servers_config = load_servers_config(
        servers,
        internal_servers_root=INTERNAL_MCP_SERVERS_DIR,
    )

    logger.info(
        "Available servers: %s", ", ".join(app_config.servers_config.servers.keys())
    )
    logger.info("Found %d servers", len(app_config.servers_config.servers))

    # Ensure that the servers which require authentication are authenticated at startup
    for _, server_config in app_config.servers_config.servers.items():
        if (
            isinstance(server_config, MCPRemoteConfig)
            and server_config.credential is not None
            and server_config.credential.requires_startup_authentication
        ):
            credential = MCPCredentialFactory.from_config(server_config.credential)
            if credential is None:
                raise Exception("Failed to get credential")

            syncify(credential.get_token())

    # Build the app, optionally with BotDesigner connector routes
    extra_routes = None
    if bd_connectors:
        from thinkingbox.tools.botdesigner_connector_proxy import (
            handle_operation as bd_handle_operation,
        )

        extra_routes = [
            Route(
                "/connectors/{session_id}/{operation_id}",
                bd_handle_operation,
                methods=["POST"],
            ),
        ]
        logger.info("BotDesigner connector routes enabled at /connectors/")

    app = create_app(extra_routes=extra_routes)

    uvicorn_kwargs: dict[str, Any] = {}
    if ssl_pem:
        uvicorn_kwargs["ssl_certfile"] = ssl_pem
        uvicorn_kwargs["ssl_keyfile"] = ssl_pem

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        access_log=False,
        **uvicorn_kwargs,
    )
