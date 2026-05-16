# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from copy import deepcopy
from typing import Any


def _recursive_merge(a: Any, b: Any) -> Any:
    """
    Recursively merge two JSON-serializable objects (dicts, lists, or primitives).

    Rules:
    - If both a and b are dict:
        * Start with a shallow/deep copy of a (so original isn't mutated).
        * Add keys from b.
        * On key collision:
            - If both values are dict: recurse.
            - If both values are list: recurse (concatenate).
            - Otherwise: b's value replaces a's.
    - If both are list:
        * Return a new list that is a concatenation of a then b (non-destructive).
    - Otherwise:
        * Return a deep copy of b (b takes precedence).

    Parameters
    ----------
    a : Any
        First JSON-serializable object.
    b : Any
        Second JSON-serializable object whose values take precedence.

    Returns
    -------
    Any
        A new merged JSON-serializable object.

    Examples
    --------
    >>> _recursive_merge({"x": 1, "y": {"a": 1}}, {"y": {"b": 2}, "z": 3})
    {'x': 1, 'y': {'a': 1, 'b': 2}, 'z': 3}

    >>> _recursive_merge([1, 2], [3, 4])
    [1, 2, 3, 4]

    >>> _recursive_merge({"a": [1]}, {"a": [2, 3]})
    {'a': [1, 2, 3]}

    >>> _recursive_merge({"a": {"b": 1}}, {"a": 5})
    {'a': 5}
    """

    # Dict merge behavior
    if isinstance(a, dict) and isinstance(b, dict):
        # Start with a copy of a so we don't mutate caller data.
        result = {}
        # Copy keys from a
        for k, v in a.items():
            result[k] = deepcopy(v)
        # Merge / override with keys from b
        for k, v_b in b.items():
            if k in result:
                v_a = result[k]
                if isinstance(v_a, dict) and isinstance(v_b, dict):
                    result[k] = _recursive_merge(v_a, v_b)
                elif isinstance(v_a, list) and isinstance(v_b, list):
                    result[k] = _recursive_merge(v_a, v_b)  # list branch concat
                else:
                    result[k] = deepcopy(v_b)
            else:
                result[k] = deepcopy(v_b)
        return result

    # List merge behavior
    if isinstance(a, list) and isinstance(b, list):
        # Return concatenation (new list)
        return list(a) + list(b)

    # Fallback: types differ or non-mergeable primitives -> b takes precedence
    return deepcopy(b)


def recursive_merge(a: dict, b: dict) -> dict:
    """Recursively merge two dictionaries"""
    assert isinstance(a, dict), f"a must be a dict, found {type(a)}"
    assert isinstance(b, dict), f"b must be a dict, found {type(b)}"
    out = _recursive_merge(a, b)
    assert isinstance(out, dict), f"unexpected merged type {type(out)}"
    return out
