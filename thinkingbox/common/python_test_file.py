# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
Parse a python file to a list of TestFunction objects that can
be converted to TestCase.

=== Example ===
from thinkingbox.common import TestContext, Judge

# string expressions starting with ! are interpreted as yaml dictionaries
# global_config = ...
"""!
scenario: retail_banking
"""

# function parameters and global imports are ignored, however they can be added
# to get hints from editor / LSP
def banking(x: TestContext, judge: Judge):
    # local_config = ...
    """!
    query: What's the balance of my savings account?
    user_context: null
    """

    # function body
    assert judge.text_yesno(x.response, "Does the message confirm the balance is 5000.00?")

    # TestFunction.name is the function name
    # TestFunction.config is {**global_config, **local_config}
    # TestFunction.test_code is the body of the function as a string

=== End ===

'''

import ast
import io
import textwrap
from typing import Any, Iterator, NamedTuple

import yaml

from thinkingbox.common.config_types import TEST_CASE_FN_NAME
from thinkingbox.common.history_loader import HistoryRef

CONFIG_FIELDS = (
    "scenario",
    "query",
    "user_context",
    "max_user_sim_turns",
    "max_agent_sim_turns",
    "bot_instructions",
    "history",
    "init",
    "format_query",
    "tags",
)


class TestFunction(NamedTuple):
    name: str
    config: dict[str, Any]
    test_code: str


def is_constant_string_expr(node: ast.Expr) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def get_yaml_obj(expr_node: ast.Expr) -> dict | None:
    assert isinstance(
        expr_node.value, ast.Constant
    ), f"expr_node.value should be Constant, found {expr_node.value}"
    value: str = expr_node.value.value
    if not value.startswith("!"):
        return None
    value = textwrap.dedent(value[1:])
    obj = yaml.safe_load(io.StringIO(value))
    return obj


def get_test_case_yaml_obj(
    fn_node: ast.FunctionDef, check: bool = False
) -> dict | None:
    if len(fn_node.body) == 0:
        return None
    node = fn_node.body[0]
    if not is_constant_string_expr(node):
        return None
    obj = get_yaml_obj(node)
    if obj is not None and check:
        for key in obj:
            if key not in CONFIG_FIELDS:
                raise ValueError(f"YAML config contains invalid key: {key!r}")
    return obj


def iter_test_functions(
    code: str,
    check_config: bool = False,
    only_names: set[str] | None = None,
) -> Iterator[TestFunction]:
    tree = ast.parse(code)
    code_lines = code.split("\n")
    global_yaml_obj: dict[str, Any] = {}

    # find line ranges for the test case functions, and for each
    # test case, preserve all the non-testcase code plus the
    # the function for that single test case

    # (name: str) -> (lineno: int, end_lineno: int)
    test_cases: dict[str, tuple[dict, int, int]] = {}

    last_end_lineno = -1
    for node in tree.body:
        # simplify the logic here by assuming two statements cannot overlap
        # on the same line (e.g. `a=1; fn()`)
        if node.lineno <= last_end_lineno:
            raise ValueError(
                f"At line {node.lineno}: multiple statements overlap or are on the"
                + " same line in the global scope"
            )
        last_end_lineno = node.end_lineno
        if is_constant_string_expr(node):
            obj = get_yaml_obj(node)
            if obj is not None:
                global_yaml_obj.update(obj)
        elif isinstance(node, ast.FunctionDef):
            obj = get_test_case_yaml_obj(node, check=check_config)
            if obj is not None:
                # this is a test case
                test_case_name = node.name
                merged_obj = {**global_yaml_obj, **obj}
                if "tags" in global_yaml_obj and "tags" in obj:
                    if isinstance(global_yaml_obj["tags"], list) and isinstance(
                        obj["tags"], list
                    ):
                        merged_obj["tags"] = global_yaml_obj["tags"] + obj["tags"]
                if "history" in merged_obj and isinstance(merged_obj["history"], str):
                    merged_obj["history"] = HistoryRef.parse(merged_obj["history"])
                test_cases[test_case_name] = (
                    merged_obj,
                    node.lineno,
                    node.end_lineno,
                )

    # filter to only requested names before expensive code reconstruction,
    # but use all test case ranges for blanking to preserve isolation
    selected_test_cases = test_cases
    if only_names is not None:
        selected_test_cases = {k: v for k, v in test_cases.items() if k in only_names}

    # code without any test case function (blank ALL test cases, not just selected)
    code_lines_without_test_cases = code_lines.copy()
    for _, (_1, lineno, end_lineno) in test_cases.items():
        for k in range(lineno - 1, end_lineno):
            code_lines_without_test_cases[k] = ""

    # for each selected test case, re-add only the related code
    for name, (obj, lineno, end_lineno) in selected_test_cases.items():
        code_assign_test_fn = f"{TEST_CASE_FN_NAME} = {name}"
        test_code_lines = (
            code_lines_without_test_cases[: lineno - 1]
            + code_lines[lineno - 1 : end_lineno]
            + code_lines_without_test_cases[end_lineno:]
            + [code_assign_test_fn, ""]
        )
        # TODO we can shrink "\n+" (or not, to preserve lineno)
        yield TestFunction(
            name=name,
            config=obj,
            test_code="\n".join(test_code_lines),
        )
