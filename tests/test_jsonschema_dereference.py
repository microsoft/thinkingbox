# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy

import pytest

from thinkingbox.tools.client.worker import jsonschema_dereference


def test_simple_reference():
    """Test resolving a simple $ref."""
    schema = {
        "type": "object",
        "properties": {"payment": {"$ref": "#/$defs/PaymentMethod"}},
        "$defs": {
            "PaymentMethod": {
                "type": "string",
                "enum": ["credit_card", "debit_card", "paypal"],
            }
        },
    }

    result = jsonschema_dereference(schema)

    assert result["properties"]["payment"]["type"] == "string"
    assert result["properties"]["payment"]["enum"] == [
        "credit_card",
        "debit_card",
        "paypal",
    ]
    assert "$ref" not in result["properties"]["payment"]


def test_nested_references():
    """Test resolving nested references (a $ref that points to a definition containing another $ref)."""
    schema = {
        "type": "object",
        "properties": {"address": {"$ref": "#/$defs/Address"}},
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"country": {"$ref": "#/$defs/Country"}},
            },
            "Country": {"type": "string", "minLength": 2, "maxLength": 2},
        },
    }

    result = jsonschema_dereference(schema)

    assert result["properties"]["address"]["type"] == "object"
    assert result["properties"]["address"]["properties"]["country"] == {
        "type": "string",
        "minLength": 2,
        "maxLength": 2,
    }
    assert "$ref" not in result["properties"]["address"]
    assert "$ref" not in result["properties"]["address"]["properties"]["country"]


def test_reference_in_array_items():
    """Test resolving $ref within array items."""
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        "$defs": {
            "Item": {"type": "object", "properties": {"name": {"type": "string"}}}
        },
    }

    result = jsonschema_dereference(schema)

    assert result["properties"]["items"]["items"]["type"] == "object"
    assert (
        result["properties"]["items"]["items"]["properties"]["name"]["type"] == "string"
    )
    assert "$ref" not in result["properties"]["items"]["items"]


def test_missing_definition():
    """Test that ValueError is raised when a referenced definition doesn't exist."""
    schema = {
        "type": "object",
        "properties": {"payment": {"$ref": "#/$defs/NonExistent"}},
        "$defs": {"PaymentMethod": {"type": "string"}},
    }

    with pytest.raises(ValueError, match="Missing.*"):
        jsonschema_dereference(schema)


def test_circular_reference_direct():
    """Test that ValueError is raised when a direct circular reference is detected."""
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "next": {"$ref": "#/$defs/Node"},
                },
            }
        },
    }

    with pytest.raises(ValueError, match="Circular.*"):
        jsonschema_dereference(schema)


def test_circular_reference_indirect():
    """Test that ValueError is raised when an indirect circular reference is detected (A -> B -> A)."""
    schema = {
        "type": "object",
        "properties": {"person": {"$ref": "#/$defs/Person"}},
        "$defs": {
            "Person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "company": {"$ref": "#/$defs/Company"},
                },
            },
            "Company": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ceo": {"$ref": "#/$defs/Person"},
                },
            },
        },
    }

    with pytest.raises(ValueError, match="Circular.*"):
        jsonschema_dereference(schema)


def test_no_references():
    """Test that schemas without $ref are returned unchanged."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
    }

    result = jsonschema_dereference(schema)

    assert result == schema


def test_escaped_json_pointer_tokens():
    """Test that JSON Pointer escaped tokens (~0 for ~, ~1 for /) are handled correctly."""
    schema = {
        "type": "object",
        "properties": {"field": {"$ref": "#/$defs/tilde~0slash~1name"}},
        "$defs": {
            "tilde~slash/name": {
                "type": "string",
                "description": "Definition with special chars",
            }
        },
    }

    result = jsonschema_dereference(schema)

    assert result["properties"]["field"]["type"] == "string"
    assert (
        result["properties"]["field"]["description"] == "Definition with special chars"
    )


def test_deeply_nested_references():
    """Test resolving deeply nested chain of references."""
    schema = {
        "type": "object",
        "properties": {"level1": {"$ref": "#/$defs/Level1"}},
        "$defs": {
            "Level1": {
                "type": "object",
                "properties": {"level2": {"$ref": "#/$defs/Level2"}},
            },
            "Level2": {
                "type": "object",
                "properties": {"level3": {"$ref": "#/$defs/Level3"}},
            },
            "Level3": {"type": "string", "pattern": "^[a-z]+$"},
        },
    }

    result = jsonschema_dereference(schema)

    assert result["properties"]["level1"]["type"] == "object"
    assert result["properties"]["level1"]["properties"]["level2"]["type"] == "object"
    assert (
        result["properties"]["level1"]["properties"]["level2"]["properties"]["level3"][
            "type"
        ]
        == "string"
    )
    assert (
        result["properties"]["level1"]["properties"]["level2"]["properties"]["level3"][
            "pattern"
        ]
        == "^[a-z]+$"
    )


def test_reference_with_additional_properties():
    """Test that $ref properties are excluded when dereferencing, and other properties are preserved."""
    schema = {
        "type": "object",
        "properties": {
            "payment": {
                "$ref": "#/$defs/PaymentMethod",
                "description": "This should not appear",
            }
        },
        "$defs": {"PaymentMethod": {"type": "string", "enum": ["card", "cash"]}},
    }

    result = jsonschema_dereference(schema)

    # The $ref replacement should not include the description from the original node
    assert result["properties"]["payment"]["type"] == "string"
    assert result["properties"]["payment"]["enum"] == ["card", "cash"]
    assert "description" not in result["properties"]["payment"]


def test_original_schema_unchanged():
    """Test that the original schema is not modified (deep copy behavior)."""
    original_schema = {
        "type": "object",
        "properties": {"payment": {"$ref": "#/$defs/PaymentMethod"}},
        "$defs": {"PaymentMethod": {"type": "string"}},
    }

    schema_copy = copy.deepcopy(original_schema)
    deref_schema = jsonschema_dereference(schema_copy)

    assert original_schema == schema_copy
    assert deref_schema != original_schema
