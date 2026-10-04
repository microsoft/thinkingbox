# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from thinkingbox.common.eval_utils import prob_A_gt_B, prob_lift_gt_eps


@pytest.mark.parametrize("k,n", [(10, 10), (200, 200), (95, 100), (0, 100)])
def test_prob_A_gt_B_identical_data_is_one_half(k, n):
    assert prob_A_gt_B(k, n, k, n) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize(
    "kA,nA,kB,nB",
    [
        (1000, 1000, 990, 1000),
        (5000, 5000, 4990, 5000),
        (100, 100, 95, 100),
        (1, 1, 1736, 20000),
        (50, 100, 40, 100),
    ],
)
def test_prob_A_gt_B_matches_sampling(kA, nA, kB, nB):
    p = prob_A_gt_B(kA, nA, kB, nB)
    assert p + prob_A_gt_B(kB, nB, kA, nA) == pytest.approx(1.0, abs=1e-5)
    mc, se = prob_lift_gt_eps(kA, nA, kB, nB, eps=0.0, ns=2_000_000, seed=0)
    assert p == pytest.approx(mc, abs=max(5 * se, 1e-4))


# References computed independently with mpmath (40 digits): P(A > B) as the
# integral of f_B(x) * (1 - F_A(x)) dx, substituting 1 - x = exp(s).
@pytest.mark.parametrize(
    "kA,nA,kB,nB,expected",
    [
        (0, 3, 10**6, 10**6, 9.85157415712e-22),
        (0, 2, 10**7, 10**7, 1.21152886771e-18),
    ],
)
def test_prob_A_gt_B_tiny_when_swapped(kA, nA, kB, nB, expected):
    # B is the narrower posterior, so the quadrature runs over B's quantiles;
    # 1 - P(B > A) would cancel to 0 here. abs=0 overrides approx's 1e-12 default.
    assert prob_A_gt_B(kA, nA, kB, nB) == pytest.approx(expected, rel=2e-2, abs=0)
