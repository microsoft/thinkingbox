# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

from thinkingbox.common.chat_types import ToolDef
from thinkingbox.common.config_types import (
    ToolDefOverride,
    update_tools_with_client_config,
)


def _make_tool(name="my_tool", description="Original desc", **schema_props):
    properties = {
        k: {"type": "string", "description": v} for k, v in schema_props.items()
    }
    return ToolDef(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties},
    )


def test_override_description_applied():
    tool = _make_tool()
    override = ToolDefOverride(name="my_tool", override_description="New desc")
    update_tools_with_client_config([tool], [override])
    assert tool.description == "New desc"


def test_override_description_none_not_clobbered():
    tool = _make_tool()
    override = ToolDefOverride(name="my_tool")
    update_tools_with_client_config([tool], [override])
    assert tool.description == "Original desc"


def test_override_arg_description_overrides_parameter():
    tool = _make_tool(city="The city name")
    override = ToolDefOverride(
        name="my_tool", override_arg_description={"city": "City for weather lookup"}
    )
    update_tools_with_client_config([tool], [override])
    assert (
        tool.input_schema["properties"]["city"]["description"]
        == "City for weather lookup"
    )


def test_override_arg_description_preserves_non_overridden_params():
    tool = _make_tool(city="The city name", country="The country")
    override = ToolDefOverride(
        name="my_tool", override_arg_description={"city": "Updated city"}
    )
    update_tools_with_client_config([tool], [override])
    assert tool.input_schema["properties"]["city"]["description"] == "Updated city"
    assert tool.input_schema["properties"]["country"]["description"] == "The country"


def test_unknown_param_name_emits_warning(caplog):
    tool = _make_tool(city="The city name")
    override = ToolDefOverride(
        name="my_tool", override_arg_description={"nonexistent": "Desc"}
    )
    with caplog.at_level(logging.WARNING):
        update_tools_with_client_config([tool], [override])
    assert "nonexistent" in caplog.text
    assert tool.input_schema["properties"]["city"]["description"] == "The city name"


def test_no_override_tool_unchanged():
    tool = _make_tool(city="The city name")
    update_tools_with_client_config([tool], [])
    assert tool.description == "Original desc"
    assert tool.input_schema["properties"]["city"]["description"] == "The city name"


def test_all_fields_combined():
    tool = _make_tool(city="The city name", units="Temperature units")
    override = ToolDefOverride(
        name="my_tool",
        override_description="Weather lookup tool",
        override_arg_description={"city": "Updated city desc"},
        direct_response="The weather is {weather}",
        is_end_turn=True,
    )
    update_tools_with_client_config([tool], [override])
    assert tool.description == "Weather lookup tool"
    assert tool.input_schema["properties"]["city"]["description"] == "Updated city desc"
    assert (
        tool.input_schema["properties"]["units"]["description"] == "Temperature units"
    )
    assert tool.direct_response == "The weather is {weather}"
    assert tool.is_end_turn is True
