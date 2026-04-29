# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime
from typing import Any


class YAMLRenderer:
    """
    The YAML parser in BotDesigner does not fully support YAML datetime syntax

    pyyaml produces: 2026-03-09 02:39:55+00:00
    BotDesigner can only parse: 2025-12-02T20:18:51.0000000Z

    This custom YAML renderer produces YAML compatible with BotDesigner
    """

    def __init__(self, indent: int = 2) -> None:
        self.indent: int = indent

    def render(self, data: Any) -> str:
        lines: list[str] = []
        self._render_value(data, lines, 0)
        # Strip trailing blank lines
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + "\n"

    def _prefix(self, level: int) -> str:
        return " " * (self.indent * level)

    def _render_scalar(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return str(value)
        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=datetime.timezone.utc)
            else:
                value = value.astimezone(datetime.timezone.utc)
            return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond:06d}0Z"
        s: str = str(value)
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"{s}"'

    def _render_value(self, value: Any, lines: list[str], level: int) -> None:
        if isinstance(value, dict):
            self._render_mapping(value, lines, level)
        elif isinstance(value, list):
            self._render_sequence(value, lines, level)
        else:
            lines.append(self._prefix(level) + self._render_scalar(value))

    def _render_mapping(
        self, mapping: dict[str, Any], lines: list[str], level: int
    ) -> None:
        if not mapping:
            lines.append(self._prefix(level) + "{}")
            return
        items: list[tuple[str, Any]] = list(mapping.items())
        for i, (key, val) in enumerate(items):
            prefix: str = self._prefix(level)
            if isinstance(val, dict) and val:
                lines.append(f"{prefix}{key}:")
                self._render_mapping(val, lines, level + 1)
                if i < len(items) - 1:
                    lines.append("")
            elif isinstance(val, list) and val:
                lines.append(f"{prefix}{key}:")
                self._render_sequence(val, lines, level + 1)
                if i < len(items) - 1:
                    lines.append("")
            elif isinstance(val, dict) and not val:
                lines.append(f"{prefix}{key}: {{}}")
            elif isinstance(val, list) and not val:
                lines.append(f"{prefix}{key}: []")
            else:
                lines.append(f"{prefix}{key}: {self._render_scalar(val)}")

    def _render_sequence(self, seq: list[Any], lines: list[str], level: int) -> None:
        for i, item in enumerate(seq):
            prefix: str = self._prefix(level)
            if isinstance(item, dict) and item:
                entries: list[tuple[str, Any]] = list(item.items())
                first_key: str
                first_val: Any
                first_key, first_val = entries[0]
                if isinstance(first_val, dict) and first_val:
                    lines.append(f"{prefix}- {first_key}:")
                    self._render_mapping(first_val, lines, level + 2)
                    if len(entries) > 1:
                        lines.append("")
                elif isinstance(first_val, list) and first_val:
                    lines.append(f"{prefix}- {first_key}:")
                    self._render_sequence(first_val, lines, level + 2)
                    if len(entries) > 1:
                        lines.append("")
                elif isinstance(first_val, dict) and not first_val:
                    lines.append(f"{prefix}- {first_key}: {{}}")
                elif isinstance(first_val, list) and not first_val:
                    lines.append(f"{prefix}- {first_key}: []")
                else:
                    lines.append(
                        f"{prefix}- {first_key}: {self._render_scalar(first_val)}"
                    )
                for j, (key, val) in enumerate(entries[1:], 1):
                    self._render_mapping_entry(
                        key, val, lines, level + 1, is_last=(j == len(entries) - 1)
                    )
                if i < len(seq) - 1:
                    lines.append("")
            elif isinstance(item, list) and item:
                lines.append(f"{prefix}-")
                self._render_sequence(item, lines, level + 1)
            elif isinstance(item, list) and not item:
                lines.append(f"{prefix}- []")
            else:
                lines.append(f"{prefix}- {self._render_scalar(item)}")

    def _render_mapping_entry(
        self, key: str, val: Any, lines: list[str], level: int, is_last: bool = False
    ) -> None:
        prefix: str = self._prefix(level)
        if isinstance(val, dict) and val:
            lines.append(f"{prefix}{key}:")
            self._render_mapping(val, lines, level + 1)
            if not is_last:
                lines.append("")
        elif isinstance(val, list) and val:
            lines.append(f"{prefix}{key}:")
            self._render_sequence(val, lines, level + 1)
            if not is_last:
                lines.append("")
        elif isinstance(val, dict) and not val:
            lines.append(f"{prefix}{key}: {{}}")
        elif isinstance(val, list) and not val:
            lines.append(f"{prefix}{key}: []")
        else:
            lines.append(f"{prefix}{key}: {self._render_scalar(val)}")
