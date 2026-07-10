"""Capital-aware portfolio state for the time-stepped executor."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PortfolioPosition:
    """An open, weighted position (shares are signed: <0 == short)."""

    symbol: str
    entry_date: pd.Timestamp
    entry_signal_date: pd.Timestamp
    entry_price: float  # effective fill after slippage
    shares: float  # signed; notional = shares * price
    entry_weight: float
    entry_idx: int  # calendar index at fill (for hold-bar accounting)

    @property
    def side(self) -> int:
        return 1 if self.shares >= 0 else -1


@dataclass
class PortfolioState:
    """NAV/cash plus the open book."""

    cash: float
    positions: dict[str, PortfolioPosition] = field(default_factory=dict)

    def market_value(self, prices: dict[str, float]) -> float:
        """Σ shares * current price over open positions (skips names without a price)."""
        total = 0.0
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is not None:
                total += pos.shares * px
        return total

    def nav(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(abs(p.shares * prices.get(p.symbol, 0.0)) for p in self.positions.values())

    def net_exposure(self, prices: dict[str, float]) -> float:
        return sum(p.shares * prices.get(p.symbol, 0.0) for p in self.positions.values())
