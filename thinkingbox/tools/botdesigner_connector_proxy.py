# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Forwards BotDesigner connector redirect requests to the ThinkingBox
Session Proxy.
"""

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from thinkingbox.tools.session_proxy import (
    CallToolRequest,
    ServerError,
    _call_tool_inner,
)

logger = logging.getLogger(__name__)


MCP_OPERATION = "mcp_b354464ac7e246a7aa29fe3304a9a3ba"


# TODO what if a tool is called "register"?
# this needs to be fixed in BotDesigner! However it seems that
# BotDesigner is not calling "register" ever
REGISTER_OPERATION = "register"


def get_tool_response_as_obj(response_text: str) -> dict:
    try:
        obj = json.loads(response_text)
    except (ValueError, TypeError):
        return {"response": response_text}
    if not isinstance(obj, dict):
        return {"response": obj}
    return obj


async def handle_operation(request: Request) -> JSONResponse:
    try:
        operation_id = request.path_params["operation_id"]

        if operation_id == MCP_OPERATION:
            raise ServerError("Not implemented")

        if operation_id == REGISTER_OPERATION:
            return JSONResponse({})

        return await ConnectorForward().handle_request(request)
    except ServerError as e:
        return JSONResponse({"error": e.message}, status_code=e.status_code)


class ConnectorForward:
    def __init__(self, redirect_ust: bool = True):
        self.redirect_ust = redirect_ust

    async def handle_request(self, request: Request) -> JSONResponse:
        session_id = request.path_params["session_id"]
        operation_id = request.path_params["operation_id"]
        body = await request.body()

        req_text = body.decode("utf-8").strip()
        if req_text:
            try:
                req_obj = json.loads(req_text)
            except json.JSONDecodeError:
                logger.info("Client sent invalid JSON object: %s", json.dumps(req_text))
                raise ServerError("Invalid JSON")
        else:
            logger.info("Client sent empty string, interpreting as '{}'")
            req_obj = {}

        call_tool_request = CallToolRequest(
            session_id=session_id,
            tool_name=operation_id,
            arguments=req_obj,
        )
        tool_resp = await self.call_tool(call_tool_request)
        return JSONResponse(tool_resp)

    async def call_tool(self, call_tool_request: CallToolRequest) -> dict:
        result = await _call_tool_inner(call_tool_request)
        text_content = result.data.get_text()
        return get_tool_response_as_obj(text_content)
