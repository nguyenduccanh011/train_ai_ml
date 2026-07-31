"""Stage-2 portfolio overlay — single source of truth for backtest AND serving.

Public surface: run_portfolio(base, signals, ctx=..., C=...) with a
PortfolioContext (DuckDBContext for backtest/replay; serving injects its own).
See docs/refactor/PORTFOLIO_LAYER_UNIFICATION.md.
"""

from stock_ml.portfolio.api import run_portfolio
from stock_ml.portfolio.constants import PortfolioConstants
from stock_ml.portfolio.context import DuckDBContext, PortfolioContext

__all__ = ["run_portfolio", "PortfolioConstants", "PortfolioContext", "DuckDBContext"]
