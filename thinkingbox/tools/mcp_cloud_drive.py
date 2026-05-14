# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import traceback
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from thinkingbox.tools.toolslib.cloud_drive import CloudDrive, CloudDriveError, File

mcp = FastMCP("cloud_drive")
db = CloudDrive(files=[])


class SuccessResponse(BaseModel):
    status: Literal["ok"]


def error_response(exc) -> str:
    if isinstance(exc, CloudDriveError):
        return "Error!\n" + str(exc)
    traceback.print_exc()
    return "Error!"


def success_response(**kwargs) -> str:
    obj = {
        "status": "ok",
        **to_jsonable_python(kwargs),
    }
    return json.dumps(obj)


@mcp.tool(name="__reserved__init")
async def initialize(config: dict):
    files = config["files"]
    db.files.clear()
    for f in files:
        db.files.append(File(**f))
    return json.dumps({})


@mcp.tool(name="__reserved__geteffects")
async def geteffects():
    obj = {"effects": db.effects, "files": db.files}
    return json.dumps(to_jsonable_python(obj))


@mcp.tool(name="get_metadata", description="Get metadata of a file")
async def get_metadata(
    path: Annotated[str, Field(description="The path of the file")],
) -> str:
    try:
        res = db.get_metadata(path=path)
        return success_response(metadata=res)
    except Exception as e:
        return error_response(e)


@mcp.tool(name="upload_file", description="Upload a file to the cloud drive")
async def upload_file(
    path: Annotated[
        str,
        Field(
            description="The path where the file will be uploaded, any parent directories are automatically created"
        ),
    ],
    text_content: Annotated[str, Field(description="The content of the file")],
    overwrite: Annotated[
        bool, Field(description="Whether to overwrite the file if it already exists")
    ],
) -> str:
    try:
        db.upload_file(path=path, text_content=text_content, overwrite=overwrite)
        return success_response()
    except Exception as e:
        return error_response(e)


@mcp.tool(name="delete_file", description="Delete a file from the cloud drive")
async def delete_file(
    path: Annotated[str, Field(description="The path of the file to delete")],
) -> str:
    try:
        db.delete_file(path=path)
        return success_response()
    except Exception as e:
        return error_response(e)


@mcp.tool(name="get_text_content", description="Get the text content of a file")
async def get_text_content(
    path: Annotated[str, Field(description="The path of the file")],
) -> str:
    try:
        res = db.get_text_content(path=path)
        return success_response(text_content=res)
    except Exception as e:
        return error_response(e)


@mcp.tool(
    name="list_files",
    description="List files whose paths start with a given prefix. note: the directory separator is /",
)
async def list_files(
    prefix: Annotated[str, Field(description="The prefix to filter files by")],
    sort: Annotated[
        Literal["name", "created", "modified"],
        Field(description="The attribute to sort files by, defaults to 'name'"),
    ] = "name",
    order: Annotated[
        Literal["ascending", "descending"],
        Field(description="The sort order, defaults to 'ascending'"),
    ] = "ascending",
) -> str:
    try:
        res = db.list_files(prefix=prefix, sort=sort, order=order)
        return success_response(files=res)
    except Exception as e:
        return error_response(e)


@mcp.tool(
    name="search_files",
    description="List files whose full paths match a given wildcard pattern",
)
async def search_files(
    pattern: Annotated[
        str,
        Field(
            description="The pattern to match file paths against. e.g. 'Documents/*.txt' or '**/example.txt'"
        ),
    ],
    sort: Annotated[
        Literal["name", "created", "modified"],
        Field(description="The attribute to sort files by, defaults to 'name'"),
    ] = "name",
    order: Annotated[
        Literal["ascending", "descending"],
        Field(description="The sort order, defaults to 'ascending'"),
    ] = "ascending",
) -> str:
    try:
        res = db.find_files(pattern=pattern, sort=sort, order=order)
        return success_response(files=res)
    except Exception as e:
        return error_response(e)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
