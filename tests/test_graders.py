# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from thinkingbox.common.graders import match_score


def test_match_score_empty_expected_dict_is_trivially_matched():
    # An empty expected dict has no required keys: partial match must be 1.0,
    # not a ZeroDivisionError from dividing by len(expected) == 0.
    assert match_score({}, {}, partial=True) == 1.0
    assert match_score({"a": 1}, {}, partial=True) == 1.0
