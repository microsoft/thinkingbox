# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime

import yaml

from thinkingbox.botdesigner.utils import YAMLRenderer


def _roundtrip(data: object) -> None:
    rendered = YAMLRenderer().render(data)
    assert yaml.safe_load(rendered) == data


def test_scalars():
    _roundtrip(
        {
            "str": "hello",
            "int": 42,
            "float": 3.14,
            "bool_t": True,
            "bool_f": False,
            "none": None,
        }
    )


def test_nested_dicts():
    _roundtrip({"a": {"b": {"c": 1}}, "d": {"e": 2, "f": "x"}})


def test_list_of_scalars():
    _roundtrip({"items": [1, 2, "three", True, None]})


def test_list_of_dicts():
    _roundtrip({"rows": [{"name": "a", "val": 1}, {"name": "b", "val": 2}]})


def test_empty_containers():
    _roundtrip({"empty_dict": {}, "empty_list": [], "normal": 1})


def test_nested_empty_containers():
    _roundtrip({"a": [{"x": [], "y": {}}], "b": {"c": {}, "d": []}})


def test_string_escaping():
    _roundtrip(
        {
            "tab": "a\tb",
            "newline": "line1\nline2",
            "quote": 'say "hi"',
            "backslash": "a\\b",
        }
    )


def test_deeply_nested():
    _roundtrip({"l1": {"l2": {"l3": [{"l4": {"l5": "deep"}}]}}})


def test_datetime():
    dt = datetime.datetime(2026, 3, 9, 2, 39, 55, tzinfo=datetime.timezone.utc)
    rendered = YAMLRenderer().render({"ts": dt})
    # BotDesigner format: ISO with 7-digit fractional seconds and Z suffix
    assert "ts: 2026-03-09T02:39:55.0000000Z\n" == rendered
    # yaml.safe_load parses the ISO string back as a datetime object
    loaded = yaml.safe_load(rendered)
    assert loaded["ts"] == dt


def test_mixed_complex():
    _roundtrip(
        {
            "users": [
                {"name": "alice", "tags": ["admin", "active"], "meta": {"age": 30}},
                {"name": "bob", "tags": [], "meta": {}},
            ],
            "config": {"nested": {"flag": True, "items": [1, 2, 3]}},
            "empty": [],
        }
    )
