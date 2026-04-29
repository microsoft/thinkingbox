# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MCP server with tools that exercise every inputType conversion path.

The purpose is to verify that ``jsonschema_to_connector_schema`` produces
YAML that BotDesigner can round-trip back to a compatible JSON Schema.
"""

import json
from enum import Enum
from typing import Annotated, Any, Literal

import pytest
from fastmcp import Client, FastMCP
from pydantic import BaseModel, Field

from thinkingbox.botdesigner.connector_schema import jsonschema_to_connector_schema
from thinkingbox.tools.client.worker import jsonschema_dereference

mcp = FastMCP("schema_test")


@mcp.tool(
    name="scalar_types",
    description="Covers all scalar types: string, number, integer, boolean",
)
async def scalar_types(
    name: Annotated[str, Field(description="A plain string")],
    score: Annotated[float, Field(description="A floating-point number")],
    count: Annotated[int, Field(description="An integer value")],
    active: Annotated[bool, Field(description="A boolean flag")],
) -> str:
    return json.dumps({})


@mcp.tool(
    name="string_formats",
    description="Covers string format mappings: date-time, byte (ignored), and bd-file",
)
async def string_formats(
    timestamp: Annotated[
        str,
        Field(
            description="A date-time value", json_schema_extra={"format": "date-time"}
        ),
    ],
    byte_data: Annotated[
        str,
        Field(
            description="Base64 content",
            json_schema_extra={"format": "byte"},
        ),
    ],
    bd_file: Annotated[
        str,
        Field(
            description="A File reference",
            json_schema_extra={"format": "bd-file"},
        ),
    ],
    plain: Annotated[str, Field(description="A plain string with no format")],
) -> str:
    return json.dumps({})


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


@mcp.tool(
    name="enum_values",
    description="Covers enum properties",
)
async def enum_values(
    priority: Annotated[Priority, Field(description="Priority level")],
    label: Annotated[str, Field(description="A required label")],
) -> str:
    return json.dumps({})


@mcp.tool(
    name="required_optional",
    description="Mix of required and optional parameters",
)
async def required_optional(
    required_field: Annotated[str, Field(description="This field is required")],
    optional_field: Annotated[
        str, Field(description="This field is optional")
    ] = "default",
) -> str:
    return json.dumps({})


class Address(BaseModel):
    street: Annotated[str, Field(description="Street name")]
    city: Annotated[str, Field(description="City name")]
    zip_code: Annotated[str, Field(description="Postal code")]


@mcp.tool(
    name="nested_object",
    description="Covers nested object (Record) conversion",
)
async def nested_object(
    name: Annotated[str, Field(description="Person name")],
    address: Annotated[Address, Field(description="Mailing address")],
) -> str:
    return json.dumps({})


@mcp.tool(
    name="primitive_array",
    description="Covers array of primitive items (Table with Value column)",
)
async def primitive_array(
    tags: Annotated[list[str], Field(description="A list of string tags")],
    scores: Annotated[list[int], Field(description="A list of integer scores")],
) -> str:
    return json.dumps({})


class LineItem(BaseModel):
    product: Annotated[str, Field(description="Product name")]
    quantity: Annotated[int, Field(description="Quantity ordered")]
    price: Annotated[float, Field(description="Unit price")]


@mcp.tool(
    name="object_array",
    description="Covers array of objects (Table with item properties)",
)
async def object_array(
    order_id: Annotated[str, Field(description="The order identifier")],
    items: Annotated[list[LineItem], Field(description="Line items in the order")],
) -> str:
    return json.dumps({})


class InnerDetail(BaseModel):
    key: Annotated[str, Field(description="Detail key")]
    value: Annotated[str, Field(description="Detail value")]


class MiddleLayer(BaseModel):
    label: Annotated[str, Field(description="Layer label")]
    detail: Annotated[InnerDetail, Field(description="Nested detail")]


@mcp.tool(
    name="deep_nesting",
    description="Covers multiple levels of object nesting",
)
async def deep_nesting(
    top_name: Annotated[str, Field(description="Top-level name")],
    layer: Annotated[MiddleLayer, Field(description="A nested structure")],
) -> str:
    return json.dumps({})


@mcp.tool(
    name="optional_nullable",
    description="Optional[T] fields (Pydantic emits anyOf[T, null])",
)
async def optional_nullable(
    required_field: Annotated[str, Field(description="Always required")],
    optional_plain: Annotated[str | None, Field(description="Optional string")] = None,
    optional_literal: Annotated[
        Literal["a", "b"] | None, Field(description="Optional enum")
    ] = None,
    optional_address: Annotated[
        Address | None, Field(description="Optional nested record")
    ] = None,
) -> str:
    return json.dumps({})


@mcp.tool(
    name="no_params",
    description="Tool with no input parameters (empty schema)",
)
async def no_params() -> str:
    return json.dumps({})


async def _get_tool_schema(tool_name: str) -> dict[str, Any]:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        for tool in tools:
            if tool.name == tool_name:
                return jsonschema_dereference(tool.inputSchema)
    raise ValueError(f"Tool not found: {tool_name}")


@pytest.mark.asyncio
async def test_scalar_types():
    schema = await _get_tool_schema("scalar_types")
    bd_schema = jsonschema_to_connector_schema(schema)
    assert bd_schema == {
        "kind": "Record",
        "properties": {
            "name": {
                "type": "String",
                "displayName": "name",
                "description": "A plain string",
                "isRequired": True,
                "order": 0,
            },
            "score": {
                "type": "Number",
                "displayName": "score",
                "description": "A floating-point number",
                "isRequired": True,
                "order": 1,
            },
            "count": {
                "type": "Number",
                "displayName": "count",
                "description": "An integer value",
                "isRequired": True,
                "order": 2,
            },
            "active": {
                "type": "Boolean",
                "displayName": "active",
                "description": "A boolean flag",
                "isRequired": True,
                "order": 3,
            },
        },
    }


@pytest.mark.asyncio
async def test_string_formats():
    schema = await _get_tool_schema("string_formats")
    bd_schema = jsonschema_to_connector_schema(schema)
    assert bd_schema == {
        "kind": "Record",
        "properties": {
            "timestamp": {
                "type": "DateTime",
                "displayName": "timestamp",
                "description": "A date-time value",
                "isRequired": True,
                "order": 0,
            },
            "byte_data": {
                "type": "String",
                "displayName": "byte_data",
                "description": "Base64 content",
                "isRequired": True,
                "order": 1,
            },
            "bd_file": {
                "type": "File",
                "displayName": "bd_file",
                "description": "A File reference",
                "isRequired": True,
                "order": 2,
            },
            "plain": {
                "type": "String",
                "displayName": "plain",
                "description": "A plain string with no format",
                "isRequired": True,
                "order": 3,
            },
        },
    }


@pytest.mark.asyncio
async def test_enum_values():
    schema = await _get_tool_schema("enum_values")
    bd_schema = jsonschema_to_connector_schema(schema)
    assert bd_schema == {
        "kind": "Record",
        "properties": {
            "priority": {
                "type": "String",
                "displayName": "priority",
                "description": "Priority level",
                "isRequired": True,
                "order": 0,
                "enumValues": ["low", "medium", "high"],
            },
            "label": {
                "type": "String",
                "displayName": "label",
                "description": "A required label",
                "isRequired": True,
                "order": 1,
            },
        },
    }


@pytest.mark.asyncio
async def test_nested_object():
    schema = await _get_tool_schema("nested_object")
    bd_schema = jsonschema_to_connector_schema(schema)
    assert bd_schema == {
        "kind": "Record",
        "properties": {
            "name": {
                "type": "String",
                "displayName": "name",
                "description": "Person name",
                "isRequired": True,
                "order": 0,
            },
            "address": {
                "displayName": "address",
                "description": "Mailing address",
                "isRequired": True,
                "order": 1,
                "type": {
                    "kind": "Record",
                    "properties": {
                        "street": {
                            "type": "String",
                            "displayName": "street",
                            "description": "Street name",
                            "isRequired": True,
                            "order": 0,
                        },
                        "city": {
                            "type": "String",
                            "displayName": "city",
                            "description": "City name",
                            "isRequired": True,
                            "order": 1,
                        },
                        "zip_code": {
                            "type": "String",
                            "displayName": "zip_code",
                            "description": "Postal code",
                            "isRequired": True,
                            "order": 2,
                        },
                    },
                },
            },
        },
    }


@pytest.mark.asyncio
async def test_primitive_array():
    schema = await _get_tool_schema("primitive_array")
    bd_schema = jsonschema_to_connector_schema(schema)
    assert bd_schema == {
        "kind": "Record",
        "properties": {
            "tags": {
                "displayName": "tags",
                "description": "A list of string tags",
                "isRequired": True,
                "order": 0,
                "type": {"kind": "Table", "properties": {"Value": "String"}},
            },
            "scores": {
                "displayName": "scores",
                "description": "A list of integer scores",
                "isRequired": True,
                "order": 1,
                "type": {"kind": "Table", "properties": {"Value": "Number"}},
            },
        },
    }


@pytest.mark.asyncio
async def test_deep_nesting():
    schema = await _get_tool_schema("deep_nesting")
    bd_schema = jsonschema_to_connector_schema(schema)
    assert bd_schema == {
        "kind": "Record",
        "properties": {
            "top_name": {
                "type": "String",
                "displayName": "top_name",
                "description": "Top-level name",
                "isRequired": True,
                "order": 0,
            },
            "layer": {
                "displayName": "layer",
                "description": "A nested structure",
                "isRequired": True,
                "order": 1,
                "type": {
                    "kind": "Record",
                    "properties": {
                        "label": {
                            "type": "String",
                            "displayName": "label",
                            "description": "Layer label",
                            "isRequired": True,
                            "order": 0,
                        },
                        "detail": {
                            "displayName": "detail",
                            "description": "Nested detail",
                            "isRequired": True,
                            "order": 1,
                            "type": {
                                "kind": "Record",
                                "properties": {
                                    "key": {
                                        "type": "String",
                                        "displayName": "key",
                                        "description": "Detail key",
                                        "isRequired": True,
                                        "order": 0,
                                    },
                                    "value": {
                                        "type": "String",
                                        "displayName": "value",
                                        "description": "Detail value",
                                        "isRequired": True,
                                        "order": 1,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


@pytest.mark.asyncio
async def test_optional_nullable():
    schema = await _get_tool_schema("optional_nullable")
    bd_schema = jsonschema_to_connector_schema(schema)
    assert bd_schema == {
        "kind": "Record",
        "properties": {
            "required_field": {
                "type": "String",
                "displayName": "required_field",
                "description": "Always required",
                "isRequired": True,
                "order": 0,
            },
            "optional_plain": {
                "type": "String",
                "displayName": "optional_plain",
                "description": "Optional string",
                "order": 1,
            },
            "optional_literal": {
                "type": "String",
                "displayName": "optional_literal",
                "description": "Optional enum",
                "order": 2,
                "enumValues": ["a", "b"],
            },
            "optional_address": {
                "displayName": "optional_address",
                "description": "Optional nested record",
                "order": 3,
                "type": {
                    "kind": "Record",
                    "properties": {
                        "street": {
                            "type": "String",
                            "displayName": "street",
                            "description": "Street name",
                            "isRequired": True,
                            "order": 0,
                        },
                        "city": {
                            "type": "String",
                            "displayName": "city",
                            "description": "City name",
                            "isRequired": True,
                            "order": 1,
                        },
                        "zip_code": {
                            "type": "String",
                            "displayName": "zip_code",
                            "description": "Postal code",
                            "isRequired": True,
                            "order": 2,
                        },
                    },
                },
            },
        },
    }


@pytest.mark.asyncio
async def test_no_params():
    schema = await _get_tool_schema("no_params")
    bd_schema = jsonschema_to_connector_schema(schema)
    assert bd_schema == {"kind": "Record", "properties": {}}


"""
To test this against BotDesigner:

1. Run mcp-start with the following servers.yaml

```
servers:
  schema_test:
    type: mcp-process
    command: ["{python}", "tests/botdesigner/test_mcp_schema.py"]
```

2. Run any test against BotDesigner with the schema_test tool in world_state

3. Retrieve the transcript with LLM API calls and get the converted schemas from tools

For reference, below are the tools produced for this server by BotDesigner (chat completions)

```
tools:
- type: function
  function:
    name: scalar_types
    description: 'Covers all scalar types: string, number, integer, boolean'
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        active:
          type: boolean
          description: A boolean flag
        name:
          type: string
          description: A plain string
        score:
          type: number
          description: A floating-point number
        count:
          type: number
          description: An integer value
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
- type: function
  function:
    name: string_formats
    description: 'Covers string format mappings: date-time, byte (ignored), and bd-file'
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        byte_data:
          type: string
          description: Base64 content
        plain:
          type: string
          description: A plain string with no format
        bd_file:
          type: string
          pattern: ^f_[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$
          description: A File reference
        timestamp:
          type: string
          format: date-time
          description: A date-time value
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
- type: function
  function:
    name: enum_values
    description: Covers enum properties
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        label:
          type: string
          description: A required label
        priority:
          type: string
          description: Priority level
          enum:
          - low
          - medium
          - high
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
- type: function
  function:
    name: required_optional
    description: Mix of required and optional parameters
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        required_field:
          type: string
          description: This field is required
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
- type: function
  function:
    name: nested_object
    description: Covers nested object (Record) conversion
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        name:
          type: string
          description: Person name
        address:
          type: object
          properties:
            city:
              type: string
              description: City name
            street:
              type: string
              description: Street name
            zip_code:
              type: string
              description: Postal code
          required:
          - city
          - street
          - zip_code
          description: Mailing address
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
- type: function
  function:
    name: primitive_array
    description: Covers array of primitive items (Table with Value column)
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        scores:
          type: array
          items:
            type: number
          description: A list of integer scores
        tags:
          type: array
          items:
            type: string
          description: A list of string tags
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
- type: function
  function:
    name: object_array
    description: Covers array of objects (Table with item properties)
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        order_id:
          type: string
          description: The order identifier
        items:
          type: array
          items:
            type: object
            properties:
              price:
                type: number
                description: Unit price
              product:
                type: string
                description: Product name
              quantity:
                type: number
                description: Quantity ordered
            required:
            - price
            - product
            - quantity
          description: Line items in the order
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
- type: function
  function:
    name: deep_nesting
    description: Covers multiple levels of object nesting
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        top_name:
          type: string
          description: Top-level name
        layer:
          type: object
          properties:
            detail:
              type: object
              properties:
                key:
                  type: string
                  description: Detail key
                value:
                  type: string
                  description: Detail value
              required:
              - key
              - value
              description: Nested detail
            label:
              type: string
              description: Layer label
          required:
          - detail
          - label
          description: A nested structure
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
- type: function
  function:
    name: no_params
    description: Tool with no input parameters (empty schema)
    parameters:
      type: object
      required:
      - explanation_of_tool_call
      properties:
        explanation_of_tool_call:
          type: string
          description: Provide a 1-3 sentence ...
```

"""


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
