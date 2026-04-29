# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Convert a JSON Schema object into the ``inputType`` structure used by
BotDesigner bot-definition YAML.

Raises ``SchemaConversionError`` with a human-readable message when the
input schema contains constructs that cannot be represented.
"""

from __future__ import annotations

from typing import Any

_MAX_DEPTH = 50


_STRING_FORMAT_MAP: dict[str, str] = {
    "date-time": "DateTime",
    # Use "bd-file" (custom format) to explicitly opt in to File type.
    "bd-file": "File",
}

_JSONSCHEMA_COMBINERS = ("oneOf", "anyOf", "allOf", "not")


class SchemaConversionError(Exception):
    """Raised when a JSON Schema cannot be represented as an inputType."""


def _path_str(path: list[str]) -> str:
    return ".".join(path) if path else "<root>"


def _fail(msg: str, path: list[str]) -> None:
    raise SchemaConversionError(f"{_path_str(path)}: {msg}")


def _unwrap_nullable(node: dict[str, Any]) -> dict[str, Any] | None:
    """If node contains a {anyOf|oneOf: [X, {"type": "null"}]}, discard
    anyOf/oneOf and its nullable annotation, merge the non-nullable inner branch
    with the parent node.

    If node does not contain that shape, return None.

    A pydantic `Optional[X] = None` translates to this construct, however
    BotDesigner connectors do not support nullable, and the correct
    translation is to discard the nullable annotation, relying on `isRequired: False`
    to allow the `default is None` path.
    """
    for kw in ("anyOf", "oneOf"):
        # check if anyOf|oneOf: [NonNullType, null]
        branches = node.get(kw)
        if not isinstance(branches, list) or len(branches) != 2:
            continue
        null_count = sum(
            1 for b in branches if isinstance(b, dict) and b.get("type") == "null"
        )
        if null_count != 1:
            continue
        # extract the non-null branch
        non_null = next(
            b for b in branches if not (isinstance(b, dict) and b.get("type") == "null")
        )
        if not isinstance(non_null, dict):
            return None

        # merge non-null branch with other keys from node and return it
        merged = dict(non_null)
        for k, v in node.items():
            # if both oneOf and anyOf are present, let the other one through:
            # it will fail later, which is the correct behavior
            if k == kw:
                continue
            merged[k] = v
        return merged
    return None


class ConnectorSchemaTransformer:
    """Walk a JSON Schema tree and convert it into BotDesigner ``inputType``.

    Uses a visitor pattern: ``visit(node)`` resolves the JSON Schema ``type``
    and dispatches to the matching ``visit_<type>`` method. Each method returns
    a type string (``"String"``, ``"Number"``, ...) or a nested
    ``{"properties": {...}}`` dict for objects.

    ``visit_property`` is called once per property inside an object, with
    extra context (``is_required``, ``order``).
    """

    def visit(
        self,
        node: dict[str, Any],
        path: list[str],
        depth: int,
    ) -> Any:
        """Resolve the node's type and call the appropriate visitor method."""
        unwrapped = _unwrap_nullable(node)
        if unwrapped is not None:
            node = unwrapped
        for kw in _JSONSCHEMA_COMBINERS:
            if kw in node:
                _fail(
                    f"'{kw}' is not supported, inputType has no union / "
                    "composition types",
                    path,
                )
        return self._resolve_and_visit_type(node, path, depth)

    def _resolve_and_visit_type(
        self,
        node: dict[str, Any],
        path: list[str],
        depth: int,
    ) -> str | dict:
        """Determine a single type string from a JSON Schema node, and visit it"""
        t: str | list | None = node.get("type")

        if t is None:
            t = "any"
        elif isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            if len(non_null) != 1:
                _fail(
                    f"multi-type arrays are not supported (got {t!r}); "
                    f"inputType requires a single concrete type",
                    path,
                )
            t = non_null[0]
        assert isinstance(t, str)

        match t:
            case "string":
                return self.visit_string(node, path, depth)
            case "number":
                return self.visit_number(node, path, depth)
            case "integer":
                return self.visit_integer(node, path, depth)
            case "boolean":
                return self.visit_boolean(node, path, depth)
            case "any":
                return self.visit_any(node, path, depth)
            case "null":
                return self.visit_null(node, path, depth)
            case "object":
                return self.visit_object(node, path, depth)
            case "array":
                return self.visit_array(node, path, depth)
            case _:
                _fail(f"'{t}' type is not valid", path)

        assert False, "unreachable"

    def visit_string(self, node: dict[str, Any], path: list[str], depth: int) -> str:
        fmt = node.get("format")
        if fmt and fmt in _STRING_FORMAT_MAP:
            return _STRING_FORMAT_MAP[fmt]
        return "String"

    def visit_number(self, node: dict[str, Any], path: list[str], depth: int) -> str:
        return "Number"

    def visit_integer(self, node: dict[str, Any], path: list[str], depth: int) -> str:
        return "Number"

    def visit_boolean(self, node: dict[str, Any], path: list[str], depth: int) -> str:
        return "Boolean"

    def visit_any(self, node: dict[str, Any], path: list[str], depth: int) -> str:
        return "Any"

    def visit_null(self, node: dict[str, Any], path: list[str], depth: int) -> str:
        _fail("'null' type is not supported in inputType", path)

    def visit_object(
        self, node: dict[str, Any], path: list[str], depth: int
    ) -> str | dict:
        if "properties" not in node:
            return "Any"

        if depth >= _MAX_DEPTH:
            _fail(
                f"nesting depth exceeds {_MAX_DEPTH}, a circular"
                " reference would cause this",
                path,
            )

        required_set = set(node.get("required", []))
        props: dict[str, Any] = node.get("properties", {})
        out: dict[str, Any] = {}

        for order, (name, prop) in enumerate(props.items()):
            child_path = [*path, name]
            out[name] = self.visit_property(
                name,
                prop,
                is_required=name in required_set,
                order=order,
                path=child_path,
                depth=depth + 1,
            )

        return {"kind": "Record", "properties": out}

    def visit_array(self, node: dict[str, Any], path: list[str], depth: int) -> dict:
        """Convert a JSON Schema array into an inputType ``kind: Table``.

        The reverse conversion (DataTypeSchemaHelper.ToJsonNode) turns
        ``TableDataType`` back into ``{"type": "array", "items": …}``
        using ``table.GetElementType(true)`` for the item schema.
        """
        items = node.get("items")
        if items is None:
            return {"kind": "Table"}

        item_type = items.get("type")

        # Primitive-element arrays -> Table(Value: <type>)
        if item_type in ("string", "number", "integer", "boolean"):
            inner = self.visit(items, [*path, "items"], depth + 1)
            return {"kind": "Table", "properties": {"Value": inner}}

        # Object-element arrays -> Table with item properties
        if item_type == "object" and "properties" in items:
            record = self.visit_object(items, [*path, "items"], depth)
            return {"kind": "Table", "properties": record["properties"]}

        # Untyped items
        return {"kind": "Table"}

    def visit_property(
        self,
        name: str,
        node: dict[str, Any],
        *,
        is_required: bool,
        order: int,
        path: list[str],
        depth: int,
    ) -> str | dict:
        """Convert one JSON Schema property into short or long form.

        Short form:  just a type string (``"String"``)
        Long form:   a dict with ``type``, ``description``, ``isRequired``, etc.
        """
        effective = _unwrap_nullable(node) or node
        typ = self.visit(effective, path, depth)

        description = effective.get("description")
        enum_values = effective.get("enum")
        has_metadata = description or is_required or enum_values or order > 0

        # Nested structure (dict) — wrap under "type:" key
        if isinstance(typ, dict):
            if has_metadata:
                entry: dict[str, Any] = {}
                if description:
                    entry["displayName"] = name
                    entry["description"] = description
                if is_required:
                    entry["isRequired"] = True
                entry["order"] = order
                if enum_values:
                    _validate_enum(enum_values, path)
                    entry["enumValues"] = list(enum_values)
                entry["type"] = typ
                return entry
            return {"type": typ}

        # Short form: plain type string
        if not has_metadata:
            return typ

        # Long form
        entry = {"type": typ}
        if description:
            entry["displayName"] = name
            entry["description"] = description
        if is_required:
            entry["isRequired"] = True
        entry["order"] = order
        if enum_values:
            _validate_enum(enum_values, path)
            entry["enumValues"] = list(enum_values)
        return entry


def _validate_enum(values: list, path: list[str]) -> None:
    if not all(isinstance(v, (str, int, float, bool)) for v in values):
        _fail(
            "enum values must be scalars (string / number / bool); "
            "complex enum entries are not supported",
            path,
        )


def jsonschema_to_connector_schema(schema: dict[str, Any]) -> dict:
    """Convert a JSON Schema into an ``inputType`` dict for BotDesigner
    connector definition.

    schema must be a JSON Schema object, .e.g. ``{"type": "object", "properties": ...}``.

    if schema is empty, returns {} (no schema)

    If the schema contains constructs that have no inputType equivalent
        (union types, null, multi-type arrays, excessive nesting, ...),
        it raises SchemaConversionError
    """
    if not schema:
        return {}

    if not isinstance(schema, dict):
        raise SchemaConversionError(
            f"expected a JSON Schema dict, got {type(schema).__name__}"
        )

    top_type = schema.get("type", "object")
    if top_type != "object":
        raise SchemaConversionError(
            f"top-level schema must be type 'object' for inputType, "
            f"got {top_type!r}"
        )

    if "properties" not in schema:
        return {}

    transformer = ConnectorSchemaTransformer()
    return transformer.visit(schema, [], depth=0)
