# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field
from pydantic_core import to_jsonable_python

mcp = FastMCP("notepad")
g_text = ""


def success_response(**kwargs) -> dict:
    obj = {
        "status": "ok",
        **to_jsonable_python(kwargs),
    }
    return obj


@mcp.tool(name="__reserved__init")
async def initialize(config: dict):
    global g_text
    g_text = config["text"]
    return json.dumps(
        {
            "status": "ok",
            "query_fmt": "ABC",
            "bot_instructions_fmt": "DEF",
            "user_context_fmt": "XYZ",
            "dont_format": "NO",
        }
    )


@mcp.tool(name="__reserved__geteffects")
async def geteffects() -> dict:
    obj = {"text": g_text}
    return to_jsonable_python(obj)


@mcp.tool(name="read_notepad", description="Get the text content of the notepad")
async def read_notepad() -> dict:
    return success_response(text=g_text)


@mcp.tool(name="write_notepad", description="Write text to the notepad")
async def write_notepad(
    text: Annotated[
        str,
        Field(
            description="New text to write to the notepad, which will replace the previous content"
        ),
    ],
) -> dict:

    global g_text
    g_text = text
    return success_response()


if __name__ == "__main__":
    mcp.run(transport="stdio")
