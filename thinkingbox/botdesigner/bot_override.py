# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import datetime
import string
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from thinkingbox.botdesigner.connector_schema import jsonschema_to_connector_schema
from thinkingbox.common.chat_types import ToolDef


class TemplateError(Exception):
    """Raised when template expansion fails, with a path to the failing node."""

    def __init__(self, message: str, path: str):
        self.path = path
        super().__init__(f"at '{path}': {message}" if path else message)


TEMPLATE_FILE = Path(__file__).resolve().parent / "bot_template.yaml"

ICON = (  # 160x160 all-white RGBA PNG base64 encoded
    "iVBORw0KGgoAAAANSUhEUgAAAKAAAACgCAYAAACLz2ctAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNS"
    + "R0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAGpSURBVHic7dIxAQAACIAw+5fWGBxuCTiYhdDUAfxmQFIG"
    + ("JGVAUgYkZUBSBiRlQFIG" * 26)
    + "JHUe9odTRhkNVgAAAABJRU5ErkJggg=="
)

RESERVED_TOOL_NAMES = {
    "UniversalSearchTool",
    "register",
}


class TplConnector(BaseModel):
    index: int
    name: str
    description: str
    input_type: dict


class TplMcpConnector(BaseModel):
    index: int
    name: str
    description: str


class ObjFormatter(string.Formatter):
    def __init__(self):
        super().__init__()
        self.saved_value = None
        self._saved = False

    def format_field(self, value, format_spec):
        assert format_spec == ""
        assert not self._saved
        self.saved_value = value
        self._saved = True
        return ""


def resolve_variable(name: str, variables: dict[str, Any]):
    # resolve a variable using python's formatter syntax, e.g. varname[key] ...
    fmt = ObjFormatter()
    fmt.vformat("{" + name + "}", (), variables)
    return fmt.saved_value


class YAMLTemplate:
    """Simple YAML template engine.

    Expands a dict tree using five directives. All errors raise
    ``TemplateError`` whose ``.path`` attribute points to the failing node
    (e.g. ``"entity.configuration.aISettings.~update"``).

    Directives
    ----------
    ~variables (root-level only)
        Defines default variables as a mapping on the root object::

            "~variables":
              ts: "2026-01-01"
              info:
                created: {"~value": "ts"}

        Entries are resolved in order, so later entries can ``~value``-reference
        earlier ones.  Caller-supplied variables (via ``apply()``) override
        these defaults.  The ``~variables`` key is stripped from the output.

    ~relative-date
        ``{"~relative-date": -1800}`` is replaced by a ``datetime.datetime``
        equal to ``now(UTC) + timedelta(seconds=value)``.  The argument must
        be a number (int or float).  Must be the only key in the dict.

    ~value
        ``{"~value": "varname"}`` is replaced by the variable's value.
        Bracket access is supported: ``{"~value": "connector[name]"}``.
        Must be the only key in the dict.  Raises on undefined variables.

    ~format
        ``{"~format": "prefix_{var[key]:03d}"}`` is replaced by a Python
        format string expanded with the current variables (via
        ``str.format_map``).  Must be the only key in the dict.

    ~unpack (list items only)
        Replicates its containing dict once per element of a list variable::

            - "~unpack": "item : items"
              name: {"~value": "item[name]"}
              id:   {"~format": "id-{index:03d}"}

        Each iteration adds ``index`` (int) and the loop variable to the
        variable scope.  Other keys in the dict are treated as the per-item
        template.

    ~update
        Merges a variable mapping into the containing dict::

            settings:
              default_key: true
              "~update": "overrides"

        Acts as an extension point: if the variable is not defined the
        directive is a silent no-op.  Raises if the variable exists but is
        not a mapping.
    """

    def __init__(self, base: dict[str, Any]):
        self.base = base

    def _expand_variables(
        self, var_defs: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Expand ~variables defaults, then layer caller overrides on top."""
        resolved: dict[str, Any] = {}
        for key, value in var_defs.items():
            resolved[key] = self._resolve_node(
                value, resolved, path=f"~variables.{key}"
            )
        resolved.update(overrides)
        return resolved

    def apply(self, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        variables = variables or {}
        obj: dict[str, Any] = copy.deepcopy(self.base)
        var_defs = obj.pop("~variables", None)
        if var_defs is not None:
            if not isinstance(var_defs, dict):
                raise TemplateError("~variables must be a mapping", path="~variables")
            variables = self._expand_variables(var_defs, variables)
        else:
            variables = variables.copy()
        self._apply_internal(obj, variables)
        return obj

    def _apply_internal(self, obj: dict[str, Any], variables: dict[str, Any]):
        self._process_dict(obj, variables, path="")

    def _resolve_node(self, node: Any, variables: dict[str, Any], path: str) -> Any:
        """Resolve a single node, returning the expanded value."""
        if isinstance(node, dict):
            if "~value" in node:
                if len(node) != 1:
                    extra = sorted(set(node) - {"~value"})
                    raise TemplateError(
                        f"~value must be the only key, found extra: {extra}",
                        path=path,
                    )
                var_name = node["~value"]
                if not isinstance(var_name, str):
                    raise TemplateError(
                        f"~value argument must be a string, "
                        f"got {type(var_name).__name__}",
                        path=path,
                    )
                try:
                    return resolve_variable(var_name, variables)
                except (KeyError, IndexError, AttributeError) as e:
                    raise TemplateError(
                        f"undefined variable '{var_name}'", path=path
                    ) from e

            if "~relative-date" in node:
                if len(node) != 1:
                    extra = sorted(set(node) - {"~relative-date"})
                    raise TemplateError(
                        f"~relative-date must be the only key, found extra: {extra}",
                        path=path,
                    )
                seconds = node["~relative-date"]
                if not isinstance(seconds, (int, float)):
                    raise TemplateError(
                        f"~relative-date argument must be a number, "
                        f"got {type(seconds).__name__}",
                        path=path,
                    )
                return datetime.datetime.now(
                    datetime.timezone.utc
                ) + datetime.timedelta(seconds=seconds)

            if "~format" in node:
                if len(node) != 1:
                    extra = sorted(set(node) - {"~format"})
                    raise TemplateError(
                        f"~format must be the only key, found extra: {extra}",
                        path=path,
                    )
                tpl = node["~format"]
                if not isinstance(tpl, str):
                    raise TemplateError(
                        f"~format argument must be a string, "
                        f"got {type(tpl).__name__}",
                        path=path,
                    )
                try:
                    return tpl.format_map(variables)
                except (KeyError, IndexError, ValueError) as e:
                    raise TemplateError(
                        f"format error in '{tpl}': {e}", path=path
                    ) from e

            # Regular dict — process in place
            self._process_dict(node, variables, path)
            return node

        if isinstance(node, list):
            self._process_list(node, variables, path)
            return node

        return node

    def _process_dict(self, d: dict[str, Any], variables: dict[str, Any], path: str):
        """Expand directives in a dict in-place."""
        # ~update: optionally merge a variable dict into this dict.
        # If the variable is not defined, this is a no-op (extension point).
        if "~update" in d:
            var_name = d.pop("~update")
            update_path = f"{path}.~update" if path else "~update"
            if not isinstance(var_name, str):
                raise TemplateError(
                    f"~update argument must be a string, "
                    f"got {type(var_name).__name__}",
                    path=update_path,
                )
            root_name = var_name.split("[", 1)[0]
            if root_name in variables:
                try:
                    update_val = resolve_variable(var_name, variables)
                except (KeyError, IndexError, AttributeError) as e:
                    raise TemplateError(
                        f"failed to resolve ~update variable '{var_name}': {e}",
                        path=update_path,
                    ) from e
                if not isinstance(update_val, dict):
                    raise TemplateError(
                        f"~update variable '{var_name}' must be a mapping, "
                        f"got {type(update_val).__name__}",
                        path=update_path,
                    )
                d.update(update_val)

        for key in list(d.keys()):
            child_path = f"{path}.{key}" if path else key
            d[key] = self._resolve_node(d[key], variables, path=child_path)

    def _process_list(self, lst: list, variables: dict[str, Any], path: str):
        """Expand directives in a list in-place, handling ~unpack."""
        new_items: list[Any] = []
        for idx, item in enumerate(lst):
            item_path = f"{path}[{idx}]"
            if isinstance(item, dict) and "~unpack" in item:
                unpack_spec = item.pop("~unpack")
                if not isinstance(unpack_spec, str):
                    raise TemplateError(
                        f"~unpack argument must be a string, "
                        f"got {type(unpack_spec).__name__}",
                        path=item_path,
                    )
                parts = unpack_spec.split(":")
                if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                    raise TemplateError(
                        f"invalid ~unpack syntax '{unpack_spec}', "
                        f"expected 'loop_var : list_var'",
                        path=item_path,
                    )
                loop_var = parts[0].strip()
                list_var_name = parts[1].strip()
                try:
                    list_val = resolve_variable(list_var_name, variables)
                except (KeyError, IndexError, AttributeError) as e:
                    raise TemplateError(
                        f"undefined variable '{list_var_name}' in ~unpack",
                        path=item_path,
                    ) from e
                if not isinstance(list_val, (list, tuple)):
                    raise TemplateError(
                        f"~unpack variable '{list_var_name}' must be a "
                        f"sequence, got {type(list_val).__name__}",
                        path=item_path,
                    )
                for i, elem in enumerate(list_val):
                    iter_vars = {**variables, "index": i, loop_var: elem}
                    item_copy = copy.deepcopy(item)
                    iter_path = f"{item_path}[{loop_var}={i}]"
                    self._process_dict(item_copy, iter_vars, path=iter_path)
                    new_items.append(item_copy)
            else:
                resolved = self._resolve_node(item, variables, path=item_path)
                new_items.append(resolved)
        lst.clear()
        lst.extend(new_items)


def load_template_file(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = TEMPLATE_FILE
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BotOverrideTemplate:
    def __init__(self, base: dict[str, Any], variables: dict[str, Any] | None = None):
        self.tpl = YAMLTemplate(base=base)
        self.variables: dict[str, Any] = {
            "icon": ICON,
            "agent_name": "Assistant",
        }
        if variables:
            self.variables.update(variables)

    def render(
        self,
        bot_instructions: str,
        recognizer_kind: str,
        add_mcp_connector: bool = True,
        connector_tools: list[ToolDef] | None = None,
        variables: dict[str, dict[str, Any]] | None = None,
    ):
        have_mcp_connectors: list[bool] = []
        mcp_connectors: list[TplMcpConnector] = []
        if add_mcp_connector:
            mcp_connectors.append(
                TplMcpConnector(
                    index=1,
                    name="mcp",
                    description="ThinkingBox MCP Connector",
                )
            )
            have_mcp_connectors.append(True)

        have_connectors: list[bool] = []
        connectors: list[TplConnector] = []
        for i, tool in enumerate(connector_tools or [], start=1):
            if tool.name in RESERVED_TOOL_NAMES:
                raise ValueError(
                    f"Tool name '{tool.name}' is reserved and cannot be used for connectors"
                )
            connectors.append(
                TplConnector(
                    index=i,
                    name=tool.name,
                    description=tool.description,
                    input_type=jsonschema_to_connector_schema(tool.input_schema),
                )
            )
        if connectors:
            have_connectors.append(True)

        # 1. Defaults from constructor
        tpl_variables = self.variables.copy()

        # 2. From render call
        if variables:
            tpl_variables.update(variables)

        # 3. Tool overrides, these are always generated and cannot be changed by
        # user config
        tool_override_variables = {
            "bot_instructions": bot_instructions,
            "have_connectors": have_connectors,
            "connectors": to_jsonable_python(connectors),
            "have_mcp_connectors": have_mcp_connectors,
            "mcp_connectors": to_jsonable_python(mcp_connectors),
            "recognizer_kind": recognizer_kind,
        }
        tpl_variables.update(tool_override_variables)

        return self.tpl.apply(variables=tpl_variables)
