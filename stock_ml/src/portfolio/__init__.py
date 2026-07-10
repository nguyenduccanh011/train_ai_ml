"""Portfolio construction tier: AlphaFrame -> TargetWeightFrame.

Turns continuous alpha scores into a desired book of position weights per date,
applying a pluggable sizing policy plus regime gating and size scaling.
"""

from stock_ml.src.portfolio.base import PortfolioConstructor, PortfolioContext
from stock_ml.src.portfolio.factory import build_portfolio_constructor

__all__ = ["PortfolioConstructor", "PortfolioContext", "build_portfolio_constructor"]
