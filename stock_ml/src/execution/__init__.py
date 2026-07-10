"""Execution tier: TargetWeightFrame + OHLCV -> trades + equity curve.

Time-stepped, capital-aware, leakage-safe (orders decided at close t fill at open t+1).
"""

from stock_ml.src.execution.executor import PortfolioExecutor, run_portfolio_backtest

__all__ = ["PortfolioExecutor", "run_portfolio_backtest"]
