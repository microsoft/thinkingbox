import math
from collections.abc import Iterable

import numpy as np
from scipy.stats import binom


def _validate_counts(n: int, successes: int, k: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and n")


def pass_rate_and_se(n: int, successes: int) -> tuple[float, float]:
    """Return the observed pass rate and its plug-in binomial standard error."""
    _validate_counts(n, successes, 1)
    pass_rate = successes / n
    return pass_rate, math.sqrt(pass_rate * (1.0 - pass_rate) / n)


def pass_at_k(n: int, successes: int, k: int) -> float:
    """Return the unbiased pass@k estimator."""
    _validate_counts(n, successes, k)
    if n - successes < k:
        return 1.0
    return 1.0 - math.comb(n - successes, k) / math.comb(n, k)


def pass_at_k_exact_se(n: int, successes: int, k: int) -> float:
    """Return the exact finite-sum plug-in binomial SE of pass@k."""
    _validate_counts(n, successes, k)
    p_hat = successes / n
    possible_successes = np.arange(n + 1)
    estimates = np.array(
        [pass_at_k(n, int(count), k) for count in possible_successes],
        dtype=float,
    )
    probabilities = binom.pmf(possible_successes, n, p_hat)
    expected = float(np.dot(estimates, probabilities))
    variance = float(np.dot(estimates**2, probabilities) - expected**2)
    return math.sqrt(max(variance, 0.0))


def pass_power_k(n: int, successes: int, k: int) -> float:
    """Return the CLI's plug-in pass^k estimator."""
    _validate_counts(n, successes, k)
    return (successes / n) ** k


def pass_power_k_exact_se(n: int, successes: int, k: int) -> float:
    """Return the exact Binomial-model SE of the plug-in pass^k estimator."""
    _validate_counts(n, successes, k)
    p_hat = successes / n
    values = np.array([(count / n) ** k for count in range(n + 1)], dtype=float)
    probabilities = np.array(
        [
            math.comb(n, count) * p_hat**count * (1.0 - p_hat) ** (n - count)
            for count in range(n + 1)
        ],
        dtype=float,
    )
    mean = float(np.dot(values, probabilities))
    second_moment = float(np.dot(values**2, probabilities))
    return math.sqrt(max(0.0, second_moment - mean**2))


def get_hierarchical_stats(
    mean_vals: np.ndarray, se_vals: np.ndarray, n: int
) -> tuple[float, float, int]:
    """Return the mean and SE from within-item noise and between-item variation."""
    mean_val = float(mean_vals.mean())
    within_var_mean = float(np.sum(se_vals**2)) / (n**2)
    between_var_mean = float(np.var(mean_vals, ddof=1)) / n if n > 1 else 0.0
    return mean_val, math.sqrt(within_var_mean + between_var_mean), n


def mean_and_se_of_mean(
    estimates: Iterable[tuple[float, float]],
) -> tuple[float, float, int]:
    """Calculate a hierarchical mean and SE from estimate/SE pairs."""
    values = list(estimates)
    if not values:
        return float("nan"), float("nan"), 0

    mean_vals = np.array([estimate for estimate, _ in values], dtype=float)
    se_vals = np.array([se for _, se in values], dtype=float)
    if not np.isfinite(mean_vals).all() or not np.isfinite(se_vals).all():
        raise ValueError("Metric estimates and standard errors must be finite")

    return get_hierarchical_stats(mean_vals, se_vals, len(values))
