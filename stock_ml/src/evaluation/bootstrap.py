"""Bootstrap confidence intervals for backtest metrics (Phase 1.5.2).

Reference: de Prado, *Advances in Financial ML*, Ch. 2.

Block bootstrap used to account for autocorrelation in returns.
Block size ≈ average holding period of trades.
"""

from __future__ import annotations

import numpy as np


def bootstrap_metric(
    returns: np.ndarray | list,
    metric_fn,
    n_iter: int = 1000,
    ci: float = 0.95,
    block_size: int | None = None,
    random_state: int | None = None,
) -> dict:
    """Compute bootstrap confidence interval for a metric.

    Phase 1.5.2: Bootstrap CI for research-grade backtesting.

    Args:
        returns: array of returns (daily, trade P&L, etc.)
        metric_fn: callable(returns) -> scalar metric (e.g., sharpe_ratio)
        n_iter: number of bootstrap iterations (default 1000)
        ci: confidence level (0.95 = 95% CI)
        block_size: block size for block bootstrap (default: auto = sqrt(len(returns)))
                    None → i.i.d. bootstrap (resample with replacement, no blocking)
        random_state: seed for reproducibility

    Returns:
        dict with keys: point_est, ci_lower, ci_upper, std, all_estimates
    """
    returns = np.asarray(returns)
    n = len(returns)

    if n < 2:
        raise ValueError("Need at least 2 observations")

    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    if random_state is not None:
        np.random.seed(random_state)

    point_est = metric_fn(returns)
    estimates = [point_est]

    for _ in range(n_iter):
        if block_size == 1:
            indices = np.random.choice(n, size=n, replace=True)
            sample = returns[indices]
        else:
            n_blocks = (n + block_size - 1) // block_size
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            indices = []
            for bid in block_indices:
                start = bid * block_size
                end = min(start + block_size, n)
                indices.extend(range(start, end))
            sample = returns[np.array(indices[:n])]

        estimates.append(metric_fn(sample))

    estimates = np.array(estimates)
    std = np.std(estimates, ddof=1)
    alpha = 1 - ci
    ci_lower = np.percentile(estimates, 100 * alpha / 2)
    ci_upper = np.percentile(estimates, 100 * (1 - alpha / 2))

    return {
        "point_estimate": float(point_est),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std": float(std),
        "ci": ci,
        "n_iter": n_iter,
        "block_size": block_size,
        "all_estimates": estimates.tolist(),
    }


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio (assuming daily returns)."""
    excess = returns - risk_free_rate / 252
    if len(excess) < 2 or np.std(excess) == 0:
        return 0.0
    return np.sqrt(252) * np.mean(excess) / np.std(excess, ddof=1)


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sortino ratio (downside deviation)."""
    excess = returns - risk_free_rate / 252
    downside = excess[excess < 0]
    if len(excess) < 2 or len(downside) == 0:
        return 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return np.sqrt(252) * np.mean(excess) / downside_std


def max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown (as negative value, e.g., -0.25 = 25% loss)."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown))


def win_rate(returns: np.ndarray) -> float:
    """Compute win rate (fraction of positive returns)."""
    return float(np.mean(returns > 0))
