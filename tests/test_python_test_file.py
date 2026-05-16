# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for parsing Python test files with YAML configuration."""

from thinkingbox.common.python_test_file import iter_test_functions


def test_merge_tags_from_global():
    """Test that tags from global YAML are merged with test case tags."""
    code = '''
"""!
scenario: test_scenario
tags: [domain:something, global-label]
"""

def my_case(x, judge):
    """!
    query: test query
    tags: [skip, test-label]
    """
    pass
'''
    fns = list(iter_test_functions(code, check_config=True))
    assert len(fns) == 1
    fn = fns[0]

    # Tags should be concatenated, not replaced
    expected_tags = ["domain:something", "global-label", "skip", "test-label"]
    assert fn.config["tags"] == expected_tags


def test_merge_tags_with_domain_override():
    """Test that domain in test case overrides global domain."""
    code = '''
"""!
scenario: test_scenario
tags: [domain:something, global-label]
"""

def my_case(x, judge):
    """!
    query: test query
    tags: [domain:other, test-label]
    """
    pass
'''
    fns = list(iter_test_functions(code, check_config=True))
    assert len(fns) == 1
    fn = fns[0]

    # Both domains should be in the list (distribute_tags will handle the override later)
    expected_tags = ["domain:something", "global-label", "domain:other", "test-label"]
    assert fn.config["tags"] == expected_tags


def test_global_tags_only():
    """Test that global tags are preserved when test case has no tags."""
    code = '''
"""!
scenario: test_scenario
tags: [domain:something, global-label]
"""

def my_case(x, judge):
    """!
    query: test query
    """
    pass
'''
    fns = list(iter_test_functions(code, check_config=True))
    assert len(fns) == 1
    fn = fns[0]

    # Should inherit global tags
    expected_tags = ["domain:something", "global-label"]
    assert fn.config["tags"] == expected_tags


def test_testcase_tags_only():
    """Test that test case tags work when there are no global tags."""
    code = '''
"""!
scenario: test_scenario
"""

def my_case(x, judge):
    """!
    query: test query
    tags: [test-label, skip]
    """
    pass
'''
    fns = list(iter_test_functions(code, check_config=True))
    assert len(fns) == 1
    fn = fns[0]

    # Should only have test case tags
    expected_tags = ["test-label", "skip"]
    assert fn.config["tags"] == expected_tags


def test_no_tags():
    """Test that test case works with no tags at all."""
    code = '''
"""!
scenario: test_scenario
"""

def my_case(x, judge):
    """!
    query: test query
    """
    pass
'''
    fns = list(iter_test_functions(code, check_config=True))
    assert len(fns) == 1
    fn = fns[0]

    # Should have no tags key or empty tags
    assert "tags" not in fn.config or fn.config["tags"] == []


def test_multiple_test_cases_different_tags():
    """Test that multiple test cases each get their own merged tags."""
    code = '''
"""!
scenario: test_scenario
tags: [global-label]
"""

def test_one(x, judge):
    """!
    query: test query 1
    tags: [tag-one]
    """
    pass

def test_two(x, judge):
    """!
    query: test query 2
    tags: [tag-two]
    """
    pass
'''
    fns = list(iter_test_functions(code, check_config=True))
    assert len(fns) == 2

    # Each should have global + their own tags
    fn_one = next(fn for fn in fns if fn.name == "test_one")
    assert fn_one.config["tags"] == ["global-label", "tag-one"]

    fn_two = next(fn for fn in fns if fn.name == "test_two")
    assert fn_two.config["tags"] == ["global-label", "tag-two"]
