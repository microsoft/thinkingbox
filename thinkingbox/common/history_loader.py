# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# See docs/history_and_metadata.md for usage and format details.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter

from thinkingbox.common.chat_types import MessageT


class HistoryRef(BaseModel):
    """Parsed history: reference. Stored in TestCase.history after parsing the docstring."""

    model_config = ConfigDict(frozen=True)

    key: str
    start: int | str  # integer list index or message_id string
    end: str  # message_id (exclusive) or "" meaning "to end of list"

    @classmethod
    def parse(cls, value: str) -> "HistoryRef":
        """Parse a 'key:start:end' string. rsplit from the right preserves colons in the key."""
        parts = value.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"history ref must have exactly 3 colon-separated parts "
                f"'key:start:end', got: {value!r}"
            )
        key, start_str, end_str = parts
        if not key:
            raise ValueError(f"history ref key must not be empty: {value!r}")
        try:
            start: int | str = int(start_str)
        except ValueError:
            start = start_str  # treat as message_id string
        return cls(key=key, start=start, end=end_str)


@dataclass
class MetaFile:
    """Parsed .meta.yaml file. Only $history: is consumed by the framework.

    Other top-level entries are valid (datasets may include documentation or
    analytics metadata alongside $history:) but are not read by the framework.
    """

    history_raw: dict[str, list[dict]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "MetaFile":
        """Load and parse a .meta.yaml file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping at the root of {path}")

        history_section = raw.get("$history") or {}
        if not isinstance(history_section, dict):
            raise ValueError(
                f"$history in {path} must be a mapping, got {type(history_section).__name__}"
            )

        history_raw: dict[str, list[dict]] = {}
        for group_key, messages in history_section.items():
            if str(group_key).startswith("#"):
                continue
            if messages is None:
                continue
            if not isinstance(messages, list):
                raise ValueError(f"$history[{group_key!r}] in {path} must be a list")
            history_raw[group_key] = messages

        return cls(history_raw=history_raw)

    @classmethod
    def load_for_test_file(cls, test_file: Path) -> "MetaFile | None":
        """Return MetaFile if <stem>.meta.yaml exists alongside test_file, else None."""
        meta_path = test_file.with_suffix("").with_suffix(".meta.yaml")
        if not meta_path.is_file():
            return None
        return cls.load(meta_path)


class HistoryLoader:
    """Resolves a HistoryRef to a slice of raw message dicts from a MetaFile."""

    def __init__(self, meta_file: MetaFile | None):
        self._meta = meta_file

    def resolve_raw(self, ref: HistoryRef) -> list[dict]:
        """Return the raw message dicts for the given HistoryRef."""
        if self._meta is None:
            raise KeyError(f"No .meta.yaml found; cannot resolve history ref {ref!r}")
        messages = self._meta.history_raw.get(ref.key)
        if messages is None:
            raise KeyError(
                f"History group {ref.key!r} not found in $history. "
                f"Available keys: {list(self._meta.history_raw)}"
            )
        return _slice_messages(messages, ref.start, ref.end)


def _slice_messages(messages: list[dict], start: int | str, end: str) -> list[dict]:
    """Slice a message list by integer index or message_id boundaries."""
    if isinstance(start, int):
        start_idx = start
        if start_idx < 0 or start_idx > len(messages):
            raise ValueError(
                f"Start index {start_idx} is out of range "
                f"for message list of length {len(messages)}"
            )
    else:
        start_idx = _find_message_id_index(messages, start)

    if end == "":
        end_idx = len(messages)
    else:
        end_idx = _find_message_id_index(messages, end)
        if end_idx <= start_idx:
            raise ValueError(
                f"End message_id {end!r} resolves to index {end_idx} which is not "
                f"after start index {start_idx}"
            )

    return messages[start_idx:end_idx]


def _find_message_id_index(messages: list[dict], message_id: str) -> int:
    """Return the index of the first message with the given message_id."""
    for i, msg in enumerate(messages):
        if msg.get("message_id") == message_id:
            return i
    available = [m.get("message_id") for m in messages if "message_id" in m]
    raise ValueError(
        f"message_id {message_id!r} not found in message list. "
        f"Available ids: {available}"
    )


_message_adapter = TypeAdapter(list[MessageT])


def parse_messages(raw_list: list[dict]) -> list[MessageT]:
    """Parse a list of raw YAML dicts into MessageT objects."""
    return _message_adapter.validate_python(raw_list)
