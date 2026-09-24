# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
from scipy.special import betainc, betaincinv


def beta_post_params(
    k: int,
    n: int,
    a0: float = 0.5,
    b0: float = 0.5,
) -> tuple[float, float]:
    if not (0 <= k <= n):
        raise ValueError("Require 0 <= k <= n")
    if a0 <= 0 or b0 <= 0:
        raise ValueError("Prior must have a0>0, b0>0")
    return a0 + k, b0 + (n - k)


def prob_in_zone(
    k: int,
    n: int,
    lower: float = 0.0625,
    upper: float = 0.9375,
    a0: float = 0.5,
    b0: float = 0.5,
) -> float:
    """P(lower < p < upper | data) Uses betainc (Beta CDF)"""
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("Invalid bounds. Must have 0.0 <= lower < upper <= 1.0.")

    a, b = beta_post_params(k, n, a0, b0)
    cdf_lower = betainc(a, b, lower)
    cdf_upper = betainc(a, b, upper)

    return float(cdf_upper - cdf_lower)


def cred_int(
    k: int,
    n: int,
    level: float = 0.95,
    a0: float = 0.5,
    b0: float = 0.5,
) -> tuple[float, float]:
    a, b = beta_post_params(k, n, a0, b0)
    lo = (1 - level) / 2
    hi = 1 - lo

    low = betaincinv(a, b, lo)
    high = betaincinv(a, b, hi)

    return low, high


def prob_A_gt_B(
    kA: int, nA: int, kB: int, nB: int, a0: float = 0.5, b0: float = 0.5
) -> float:
    """
    Exact P(p_A > p_B) = INTEGRAL_0^1 f_A(x) * F_B(x) dx
                       = INTEGRAL_0^1 F_B(F_A^-1(u)) du   (u = F_A(x)),
    implemented with betaincinv() and betainc()

    Integrating over quantiles keeps the integrand bounded in [0, 1].
    Integrating f_A directly on a fixed grid misses its peak when the
    posterior is narrow or piled up at 0 or 1 (e.g. 1000/1000 passes).
    We integrate over the quantiles of the narrower posterior, so the
    integrand is smooth, and use P(A > B) = 1 - P(B > A) when that is B.
    """

    aA, bA = beta_post_params(kA, nA, a0, b0)
    aB, bB = beta_post_params(kB, nB, a0, b0)
    swap = aB + bB > aA + bA
    if swap:
        aA, bA, aB, bB = aB, bB, aA, bA

    x, w = np.polynomial.legendre.leggauss(200)  # 200-pt GL; adjust if needed
    # convert frm [-1,1] to [0,1]
    u = (x + 1.0) / 2.0
    wu = w / 2.0
    integrand = betainc(aB, bB, betaincinv(aA, bA, u))
    p = float(np.sum(wu * integrand))
    return 1.0 - p if swap else p


# Sampling to estimate P(A-B > epsilon)
def prob_lift_gt_eps(
    kA: int,
    nA: int,
    kB: int,
    nB: int,
    eps: float = 0.0,
    a0: float = 0.5,
    b0: float = 0.5,
    ns: int = 200_000,
    seed: int | None = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aA, bA = beta_post_params(kA, nA, a0, b0)
    aB, bB = beta_post_params(kB, nB, a0, b0)
    pA = rng.beta(aA, bA, size=ns)
    pB = rng.beta(aB, bB, size=ns)
    hit = (pA - pB) > eps
    p = float(hit.mean())
    se = float(np.sqrt(p * (1 - p) / ns))
    return p, se
