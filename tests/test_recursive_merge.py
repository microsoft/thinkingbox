# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from thinkingbox.common.recursive_merge import (
    _recursive_merge,
    recursive_merge,
)


def test_merge_simple_dicts():
    a = {"x": 1, "y": 2}
    b = {"y": 3, "z": 4}
    assert recursive_merge(a, b) == {"x": 1, "y": 3, "z": 4}
    assert a == {"x": 1, "y": 2}  # not mutated
    assert b == {"y": 3, "z": 4}


def test_merge_nested_dicts():
    a = {"a": {"x": 1, "y": {"k": 1}}, "b": 2}
    b = {"a": {"y": {"m": 2}, "z": 3}, "c": 4}
    assert recursive_merge(a, b) == {
        "a": {"x": 1, "y": {"k": 1, "m": 2}, "z": 3},
        "b": 2,
        "c": 4,
    }


def test_merge_lists():
    assert _recursive_merge([1, 2], [3, 4]) == [1, 2, 3, 4]


def test_merge_nested_lists():
    a = {"a": [1, 2]}
    b = {"a": [3]}
    assert recursive_merge(a, b) == {"a": [1, 2, 3]}


def test_merge_nested_lists_of_dict():
    a = {
        "files": [
            {"name": "a"},
            {"name": "b"},
        ],
    }
    b = {
        "files": [
            {"name": "c"},
        ],
    }
    assert recursive_merge(a, b) == {
        "files": [
            {"name": "a"},
            {"name": "b"},
            {"name": "c"},
        ]
    }


def test_type_change_overwrites():
    a = {"a": {"x": 1}}
    b = {"a": 5}
    assert recursive_merge(a, b) == {"a": 5}


def test_different_types_root():
    assert _recursive_merge({"a": 1}, ["not", "a", "dict"]) == ["not", "a", "dict"]
