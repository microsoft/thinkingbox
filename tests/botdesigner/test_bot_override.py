# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime

import pytest

from thinkingbox.botdesigner.bot_override import (
    BotOverrideTemplate,
    TemplateError,
    YAMLTemplate,
    load_template_file,
    resolve_variable,
)
from thinkingbox.common.chat_types import ToolDef

# ---------------------------------------------------------------------------
# ~value directive
# ---------------------------------------------------------------------------


def test_value():
    v = {
        "a_int": 99,
        "a_list": [1, 2, 3],
        "a_nested": {"key": "nested"},
    }
    base = {
        "out_int": {"~value": "a_int"},
        "out_list": {"~value": "a_list"},
        "out_nested": {"~value": "a_nested[key]"},
    }
    expected = {
        "out_int": 99,
        "out_list": [1, 2, 3],
        "out_nested": "nested",
    }
    out = YAMLTemplate(base).apply(v)
    assert out == expected


def test_value_undefined_variable():
    """~value raises TemplateError with path for undefined variables."""
    tpl = YAMLTemplate({"a": {"b": {"~value": "missing"}}})
    with pytest.raises(TemplateError, match="missing") as exc_info:
        tpl.apply()
    assert exc_info.value.path == "a.b"


def test_value_invalid():
    """~value must be the only key in the dict, and its value must be a string"""
    tpl = YAMLTemplate({"x": {"~value": "v", "extra": 1}})
    with pytest.raises(TemplateError, match="extra"):
        tpl.apply({"v": 1})

    tpl = YAMLTemplate({"x": {"~value": 123}})
    with pytest.raises(TemplateError, match="string"):
        tpl.apply()


# ---------------------------------------------------------------------------
# ~format directive
# ---------------------------------------------------------------------------


def test_format():
    """~format performs Python string formatting."""
    v = {
        "name": "world",
        "n": 7,
        "d": {"k": "val"},
    }
    base = {
        "msg": {"~format": "hello {name}"},
        "id": {"~format": "item-{n:03d}"},
        "out": {"~format": "{d[k]}!"},
    }
    expected = {
        "msg": "hello world",
        "id": "item-007",
        "out": "val!",
    }
    out = YAMLTemplate(base).apply(v)
    assert out == expected


def test_format_missing_variable():
    """~format raises TemplateError with path for missing variables."""
    tpl = YAMLTemplate({"x": {"~format": "{nope}"}})
    with pytest.raises(TemplateError, match="nope") as exc_info:
        tpl.apply()
    assert exc_info.value.path == "x"


def test_format_invalid():
    """~format must be the only key in the dict, and its value must be a string"""
    tpl = YAMLTemplate({"x": {"~format": "{v}", "extra": 1}})
    with pytest.raises(TemplateError, match="extra"):
        tpl.apply({"v": 1})

    tpl = YAMLTemplate({"x": {"~format": 42}})
    with pytest.raises(TemplateError, match="string"):
        tpl.apply()


# ---------------------------------------------------------------------------
# ~variables section
# ---------------------------------------------------------------------------


def test_variables():
    """~variables entries are available as defaults."""
    base = {
        "~variables": {
            "x": 10,
            "a": "hi",
            "b": {"~value": "a"},
            "c": {
                "key": {"~value": "x"},
            },
        },
        "out_x": {"~value": "x"},
        "out_b": {"~value": "b"},
        "out_c": {"~value": "c"},
    }
    expected = {
        "out_x": 10,
        "out_b": "hi",
        "out_c": {"key": 10},
    }
    out = YAMLTemplate(base).apply()
    assert out == expected


def test_variables_replaced_by_overrides():
    """Caller-supplied variables override ~variables defaults."""
    v = {
        "x": 99,
    }
    base = {
        "~variables": {
            "x": 1,
        },
        "v": {"~value": "x"},
    }
    out = YAMLTemplate(base).apply()
    assert out == {"v": 1}

    out = YAMLTemplate(base).apply(v)
    assert out == {"v": 99}


def test_variables_do_not_expand_overrides():
    """Caller-supplied variables override ~variables defaults."""
    v = {
        "x": 99,
    }
    base = {
        "~variables": {
            "y": {"~value": "x"},
        },
        "v": {"~value": "y"},
    }
    with pytest.raises(TemplateError, match="undefined variable"):
        YAMLTemplate(base).apply(v)


def test_variables_invalid():
    """~variables must be a mapping."""
    tpl = YAMLTemplate({"~variables": [1, 2]})
    with pytest.raises(TemplateError, match="mapping"):
        tpl.apply()


# ---------------------------------------------------------------------------
# ~relative-date directive
# ---------------------------------------------------------------------------


def _check_ts(ts, before: datetime.datetime, after: datetime.datetime, delta: int):
    assert isinstance(ts, datetime.datetime)
    assert (
        before + datetime.timedelta(seconds=delta)
        <= ts
        <= after + datetime.timedelta(seconds=delta)
    )


def test_relative_date_produces_datetime():
    """~relative-date returns a UTC datetime offset by the given seconds."""
    base = {
        "~variables": {"ts_var": {"~relative-date": -1800}},
        "ts_past": {"~relative-date": -60},
        "ts_future": {"~relative-date": 300},
        "ts_from_var": {"~value": "ts_var"},
    }
    before = datetime.datetime.now(datetime.timezone.utc)
    out = YAMLTemplate(base).apply()
    after = datetime.datetime.now(datetime.timezone.utc)

    _check_ts(out["ts_past"], before, after, -60)
    _check_ts(out["ts_future"], before, after, 300)
    _check_ts(out["ts_from_var"], before, after, -1800)


def test_relative_date_invalid():
    """~relative-date must be the only key in the dict, and its value must be a number"""
    tpl = YAMLTemplate({"ts": {"~relative-date": -60, "extra": 1}})
    with pytest.raises(TemplateError, match="extra"):
        tpl.apply()

    tpl = YAMLTemplate({"ts": {"~relative-date": "bad"}})
    with pytest.raises(TemplateError, match="number"):
        tpl.apply()


# ---------------------------------------------------------------------------
# ~update directive
# ---------------------------------------------------------------------------


def test_update():
    """~update merges a variable dict into the containing dict."""
    v = {
        "extra": {"added": "yes"},
        "overrides": {"mode": "custom"},
    }
    base = {
        "merged": {"base": True, "~update": "extra"},
        "overridden": {"mode": "default", "~update": "overrides"},
        "noop": {"base": True, "~update": "optional"},
    }
    expected = {
        "merged": {"base": True, "added": "yes"},
        "overridden": {"mode": "custom"},
        "noop": {"base": True},
    }
    out = YAMLTemplate(base).apply(v)
    assert out == expected


def test_update_invalid():
    """~update argument must be a string and must resolve to a mapping."""
    tpl = YAMLTemplate({"cfg": {"~update": "bad"}})
    with pytest.raises(TemplateError, match="mapping"):
        tpl.apply({"bad": "not a dict"})

    tpl = YAMLTemplate({"cfg": {"~update": 123}})
    with pytest.raises(TemplateError, match="string"):
        tpl.apply()


# ---------------------------------------------------------------------------
# ~unpack directive
# ---------------------------------------------------------------------------


def test_unpack():
    """~unpack expands list items with access to element and index."""
    v = {
        "connectors": [{"name": "alpha"}, {"name": "beta"}],
        "empty": [],
    }
    base = {
        "rows": [
            {"static": True},
            {
                "~unpack": "c : connectors",
                "label": {"~value": "c[name]"},
                "id": {"~format": "conn-{index:03d}"},
            },
        ],
        "empty_items": [
            {"~unpack": "x : empty", "v": {"~value": "x"}},
        ],
    }
    expected = {
        "rows": [
            {"static": True},
            {"label": "alpha", "id": "conn-000"},
            {"label": "beta", "id": "conn-001"},
        ],
        "empty_items": [],
    }
    out = YAMLTemplate(base).apply(v)
    assert out == expected


def test_unpack_undefined_list_variable():
    """~unpack raises TemplateError when the list variable is missing."""
    tpl = YAMLTemplate({"items": [{"~unpack": "x : missing"}]})
    with pytest.raises(TemplateError, match="missing") as exc_info:
        tpl.apply()
    assert "items[0]" in exc_info.value.path


def test_unpack_invalid():
    """~unpack requires valid syntax and a sequence variable."""
    tpl = YAMLTemplate({"items": [{"~unpack": "x : bad"}]})
    with pytest.raises(TemplateError, match="sequence"):
        tpl.apply({"bad": "not a list"})

    tpl = YAMLTemplate({"items": [{"~unpack": "no colon here"}]})
    with pytest.raises(TemplateError, match="syntax"):
        tpl.apply()

    tpl = YAMLTemplate({"items": [{"~unpack": " : xs"}]})
    with pytest.raises(TemplateError, match="syntax"):
        tpl.apply({"xs": []})


# ---------------------------------------------------------------------------
# Template immutability
# ---------------------------------------------------------------------------


def test_template_immutability():
    """apply() does not mutate the stored template or leak references."""
    tpl = YAMLTemplate(
        {"~variables": {"x": 1}, "items": [1, 2, 3], "v": {"~value": "x"}}
    )
    tpl.apply({"x": 100})
    assert tpl.apply() == {"items": [1, 2, 3], "v": 1}

    out = tpl.apply()
    out["items"].append(4)
    assert tpl.apply() == {"items": [1, 2, 3], "v": 1}


# ---------------------------------------------------------------------------
# Recursive / nested expansion
# ---------------------------------------------------------------------------


def test_nested_expansion():
    """Directives expand through nested dicts and lists; scalars pass through."""
    v = {"x": "deep", "y": 2}
    base = {
        "a": {"b": {"c": {"~value": "x"}}},
        "items": [{"~value": "x"}, "static", {"~value": "y"}],
        "s": "hello",
        "n": 42,
        "b": True,
        "nil": None,
    }
    expected = {
        "a": {"b": {"c": "deep"}},
        "items": ["deep", "static", 2],
        "s": "hello",
        "n": 42,
        "b": True,
        "nil": None,
    }
    out = YAMLTemplate(base).apply(v)
    assert out == expected


# ---------------------------------------------------------------------------
# Error path accuracy
# ---------------------------------------------------------------------------


def test_error_path():
    """TemplateError.path reflects the full path to the failing node."""
    tpl = YAMLTemplate({"a": {"b": {"c": {"~value": "nope"}}}})
    with pytest.raises(TemplateError) as exc_info:
        tpl.apply()
    assert exc_info.value.path == "a.b.c"

    tpl = YAMLTemplate({"items": ["ok", {"~value": "nope"}]})
    with pytest.raises(TemplateError) as exc_info:
        tpl.apply()
    assert exc_info.value.path == "items[1]"

    tpl = YAMLTemplate(
        {"items": [{"~unpack": "c : cs", "name": {"~value": "c[missing_key]"}}]}
    )
    with pytest.raises(TemplateError) as exc_info:
        tpl.apply({"cs": [{"other": "val"}]})
    assert "items[0]" in exc_info.value.path
    assert "c=0" in exc_info.value.path

    tpl = YAMLTemplate(
        {"~variables": {"bad": {"~value": "undefined"}}, "out": {"~value": "bad"}}
    )
    with pytest.raises(TemplateError) as exc_info:
        tpl.apply()
    assert "~variables.bad" in exc_info.value.path


# ---------------------------------------------------------------------------
# Full template file (integration)
# ---------------------------------------------------------------------------


NUM_COMPONENTS_BEFORE_CONNECTORS = 2


def test_full_template_expansion():
    """Expand the real bot_template.yaml with representative variables."""
    tpl = BotOverrideTemplate(
        base=load_template_file(),
        variables={"icon": "test_icon_b64", "solution_id": "sol-001"},
    )
    connector_tools = [
        ToolDef(name="get_weather", description="Get weather", input_schema={}),
        ToolDef(name="search", description="Search things", input_schema={}),
    ]
    result = tpl.render(
        bot_instructions="Test instructions",
        recognizer_kind="TestRecognizer",
        add_mcp_connector=False,
        connector_tools=connector_tools,
    )

    # Root-level structure
    assert result["kind"] == "BotDefinition"

    # ~value resolved from ~variables
    entity = result["entity"]
    assert entity["auditInfo"]["createdBy"] == entity["auditInfo"]["modifiedBy"]

    # ~value with caller variable
    assert entity["iconBase64"] == "test_icon_b64"
    assert entity["managedProperties"]["solutionId"] == "sol-001"

    # ~unpack: 2 connectors
    components = result["components"][NUM_COMPONENTS_BEFORE_CONNECTORS:]
    assert len(components) == 2
    assert components[0]["displayName"] == "get_weather"
    assert components[1]["displayName"] == "search"

    # ~format with connector index (starts at 1)
    assert components[0]["id"].endswith("001")
    assert components[1]["id"].endswith("002")

    # connectionReferences unpacked
    assert len(result["connectionReferences"]) == 2

    # connectorDefinitions operations unpacked
    ops = result["connectorDefinitions"][0]["operations"]
    assert len(ops) == 2
    assert ops[0]["operationId"] == "get_weather"
    assert ops[1]["operationId"] == "search"


def test_full_template_update_extension_point():
    """~update extension points accept caller-provided overrides."""
    tpl = BotOverrideTemplate(base=load_template_file())
    result = tpl.render(
        bot_instructions="Test instructions",
        recognizer_kind="TestRecognizer",
        add_mcp_connector=False,
        variables={"ai_settings": {"customKey": "customVal"}},
    )
    ai = result["entity"]["configuration"]["aISettings"]
    # Default keys preserved
    assert ai["useModelKnowledge"] is True
    # Extension merged
    assert ai["customKey"] == "customVal"
