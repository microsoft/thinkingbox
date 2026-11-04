# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from thinkingbox.common.graders import average_reward


def test_weighted_average_reward_with_zero_total_weight_returns_zero():
    # Regression: a weighted average whose common weights sum to zero must not
    # raise ZeroDivisionError. This mirrors the existing degenerate-case
    # convention in average_reward (empty rewards / no common keys -> 0.0).
    assert average_reward({"match_score": 5.0}, {"match_score": 0.0}) == 0.0
    assert average_reward({"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 0.0}) == 0.0
    # Weights that cancel to a zero sum are also a zero normalizer.
    assert average_reward({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": -1.0}) == 0.0


def test_weighted_average_reward_still_computes_normal_case():
    # A non-zero total weight is unaffected by the guard.
    assert average_reward({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 1.0}) == 0.5


def test_existing_average_reward_guards_unchanged():
    assert average_reward({}) == 0.0
    assert average_reward({"a": 1.0}, {"b": 2.0}) == 0.0
    assert average_reward({"a": 1.0, "b": 3.0}) == 2.0
