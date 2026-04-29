# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from thinkingbox.common import Judge, TestContext

"""!
scenario: mcs_defaults
"""


def test_mcs_defaults_difficult(x: TestContext, judge: Judge):
    """!
    query: |
        What are your store hours and where are you located?
    """
    assert judge.no_repeat(x.response, x.tool_direct_responses)


def test_mcs_defaults_easy(x: TestContext, judge: Judge):
    """!
    query: |
        What are your store hours?
    """
    assert judge.no_repeat(x.response, x.tool_direct_responses)


def test_mcs_defaults_medium(x: TestContext, judge: Judge):
    """!
    query: |
        What are your store location and when are you open?
    """
    assert judge.no_repeat(x.response, x.tool_direct_responses)
