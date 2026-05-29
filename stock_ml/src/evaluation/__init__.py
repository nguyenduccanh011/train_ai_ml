"""Research-grade evaluation metrics (Phase 1.5).

- bootstrap: Confidence intervals for backtest metrics
- dsr: Deflated Sharpe Ratio (multiple-testing correction)
- pbo: Probability of Backtest Overfitting (Combinatorially Symmetric CV)
"""

from src.evaluation.bootstrap import (
    bootstrap_metric,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from src.evaluation.dsr import deflated_sharpe, pvalues_from_sharpe
from src.evaluation.pbo import pbo

__all__ = [
    "bootstrap_metric",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "win_rate",
    "deflated_sharpe",
    "pvalues_from_sharpe",
    "pbo",
]
