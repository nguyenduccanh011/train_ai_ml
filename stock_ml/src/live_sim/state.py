"""Live simulator state machine — positions, trades, portfolio."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    """An open position."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float  # effective fill after slippage
    entry_signal_date: pd.Timestamp  # T-1 date that generated the signal
    hold_bars: int = 0


@dataclass
class ClosedTrade:
    """A closed position (trade)."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    holding_days: int
    pnl_pct: float  # net of all costs
    exit_reason: str  # 'signal' | 'max_hold' | 'hard_stop' | 'end_of_sim'
    entry_signal_date: pd.Timestamp
    signal_hash: str  # hash of FrozenSignalSet that generated entry signal


@dataclass
class SimState:
    """Current state of the simulation."""

    current_date: pd.Timestamp | None = None
    open_positions: dict[str, Position] = field(default_factory=dict)  # symbol → Position
    closed_trades: list[ClosedTrade] = field(default_factory=list)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.open_positions

    def open_position(self, pos: Position) -> None:
        if self.has_position(pos.symbol):
            raise ValueError(f"Position already open for {pos.symbol}")
        self.open_positions[pos.symbol] = pos

    def close_position(self, symbol: str) -> Position:
        if not self.has_position(symbol):
            raise ValueError(f"No open position for {symbol}")
        return self.open_positions.pop(symbol)

    def advance_holds(self) -> None:
        """Increment hold_bars for all open positions."""
        for pos in self.open_positions.values():
            pos.hold_bars += 1

    def n_open(self) -> int:
        return len(self.open_positions)

    def n_closed(self) -> int:
        return len(self.closed_trades)

    def realized_pnl(self) -> float:
        return sum(t.pnl_pct for t in self.closed_trades)
