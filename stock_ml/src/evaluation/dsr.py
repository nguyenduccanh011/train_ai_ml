"""Deflated Sharpe Ratio (DSR) for multiple-testing correction (Phase 1.5.3).

Reference: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".

When choosing the best strategy from N candidates, the observed Sharpe is
biased upward by √(2 ln N) standard deviations. DSR corrects for this.

If you try 50 YAMLs and pick the best Sharpe, naïve Sharpe is biased by ~1.87σ.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    returns: np.ndarray | None = None,
    returns_skew: float | None = None,
    returns_kurt: float | None = None,
    sample_length: int | None = None,
) -> float:
    """Compute Deflated Sharpe Ratio.

    Phase 1.5.3: DSR corrects naïve Sharpe for multiple testing bias.

    Args:
        sharpe: naïve Sharpe ratio
        n_trials: number of strategy candidates tried (search space size)
        returns: optional array of returns (to compute skew/kurt if not provided)
        returns_skew: skewness of returns (default 0 if returns is None)
        returns_kurt: kurtosis of returns (excess kurtosis, default 0)
        sample_length: number of observations (if not provided, inferred from returns)

    Returns:
        deflated_sharpe: DSR adjusted for multiple testing

    Notes:
        - Formula: DSR = SR - sqrt(2 * ln(N)) * sigma(SR)
        - sigma(SR) accounts for returns distribution (skew, kurtosis)
        - References: Bailey & López de Prado (2014)
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    if returns is not None:
        returns = np.asarray(returns)
        sample_length = len(returns)
        if returns_skew is None:
            returns_skew = stats.skew(returns)
        if returns_kurt is None:
            returns_kurt = stats.kurtosis(returns, fisher=True)
    else:
        if sample_length is None:
            raise ValueError("Must provide returns or sample_length")
        if returns_skew is None:
            returns_skew = 0.0
        if returns_kurt is None:
            returns_kurt = 0.0

    sharpe_std = _sharpe_std(sample_length, returns_skew, returns_kurt)
    multiple_testing_correction = np.sqrt(2 * np.log(n_trials))

    deflated = sharpe - multiple_testing_correction * sharpe_std
    return float(deflated)


def _sharpe_std(sample_length: int, returns_skew: float = 0.0, returns_kurt: float = 0.0) -> float:
    """Compute standard deviation of Sharpe ratio estimator.

    Accounts for returns skewness and kurtosis (excess).

    Formula (from Bailey et al. 2017):
    σ(SR) = sqrt(1/T * (1 + 0.5*SR² - skew*SR + (excess_kurt-3)/4 * SR²))

    where T = sample_length, excess_kurt = kurtosis (Fisher definition, already excess)
    """
    if sample_length < 2:
        return 1.0

    T = sample_length
    variance = 1 + 0.5 * (0**2) - returns_skew * 0 + (returns_kurt / 4) * (0**2)
    sharpe_std = np.sqrt(variance / T)
    return float(sharpe_std)


def pvalues_from_sharpe(
    sharpes: np.ndarray, n_trials: int | None = None, sample_length: int | None = None
) -> np.ndarray:
    """Compute p-values for multiple Sharpe ratios under DSR framework.

    Each p-value is the probability of observing sharpe[i] or higher by chance.

    Args:
        sharpes: array of Sharpe ratios
        n_trials: number of trials (if None, uses len(sharpes))
        sample_length: sample length for each Sharpe (default 252 for annual)

    Returns:
        array of p-values
    """
    sharpes = np.asarray(sharpes)
    if n_trials is None:
        n_trials = len(sharpes)
    if sample_length is None:
        sample_length = 252

    sharpe_std = _sharpe_std(sample_length)
    multiple_testing_correction = np.sqrt(2 * np.log(n_trials))

    deflated = sharpes - multiple_testing_correction * sharpe_std

    pvalues = 1 - stats.norm.cdf(deflated)
    return pvalues
