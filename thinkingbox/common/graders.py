# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common grading functions for evaluating AI outputs."""

from typing import Any


def average_reward(
    rewards: dict[str, float], weights: dict[str, float] | None = None
) -> float:
    """Compute the average reward. If weights are provided, compute the weighted average.
    The dictionary keywords can be organized as reward types, e.g. match_score, compliance_score,
    response_score etc or by names of different tests. This can be extended as needed, as long as
    the keys in rewards and weights match.

    Args:
        rewards: A dictionary of reward names to their float values.
        weights: An optional dictionary of reward names to their float weights.
    Returns:
        The average (or weighted average) reward as a float.

    """
    if not rewards:
        return 0.0

    if weights:
        # Only use rewards that have corresponding weights
        common_keys = set(rewards.keys()) & set(weights.keys())
        if not common_keys:
            return 0.0

        weighted_sum = sum(rewards[key] * weights[key] for key in common_keys)
        weight_sum = sum(weights[key] for key in common_keys)

        return weighted_sum / weight_sum

    # Simple average of all rewards
    return sum(rewards.values()) / len(rewards)


def match_score(
    result: Any,
    expected: Any,
    partial: bool = False,
    is_critical: bool = False,
) -> float:
    """
    Compute the match score between the result and expected dictionaries.
    If is_critical is True, a mismatch will raise an AssertionError.
    """
    if type(result) is not type(expected):
        raise ValueError(
            f"Cannot compare different types: {type(result)} vs {type(expected)}"
        )

    if partial and isinstance(expected, dict):
        score = sum(
            1 for k in expected if k in result and result[k] == expected[k]
        ) / len(expected)
        if is_critical and score < 1.0:
            raise AssertionError(f"Critical partial match failed. Score: {score}")
        return score
    elif partial and not isinstance(expected, dict):
        raise ValueError("Partial matching is only supported for dictionaries.")

    match = result == expected
    if is_critical and not match:
        raise AssertionError(
            f"Critical exact match failed. Expected: {expected}, Got: {result}"
        )

    return 1.0 if match else 0.0
