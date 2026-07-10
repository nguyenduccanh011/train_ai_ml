"""Portfolio constructor protocol + shared context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class PortfolioContext:
    """Cross-cutting inputs shared by every sizing policy.

    Attributes:
        direction: "long" (long-only), "short" (short-only), "long_short" or
            "market_neutral" (both sides allowed).
        max_gross: cap on gross exposure per date (sum of |weight|).
        max_per_name: cap on a single name's |weight| (applied before gross renorm).
        regime_signal: optional [date, symbol, gate] frame; gate==0 forces weight 0.
        size_signal: optional [date, symbol, size_score] frame; multiplies weights.
        ohlcv: optional price history (needed by volatility-targeting).
    """

    direction: str = "long"
    max_gross: float = 1.0
    max_per_name: float = 1.0
    regime_signal: pd.DataFrame | None = None
    size_signal: pd.DataFrame | None = None
    ohlcv: pd.DataFrame | None = None


class PortfolioConstructor(Protocol):
    """Maps an AlphaFrame to a TargetWeightFrame.

    Input  : [symbol, date, score] (+ optional side_hint)
    Output : [date, symbol, target_weight, side, rank, gated, score]
    """

    def build(self, alpha: pd.DataFrame, ctx: PortfolioContext) -> pd.DataFrame: ...
