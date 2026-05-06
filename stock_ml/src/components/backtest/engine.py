from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.backtest.defaults import DEFAULT_TRADING_COST
from src.components.base import Trade

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.components.base import Action


def _detect_date_column(df: pd.DataFrame) -> str:
    if "timestamp" in df.columns:
        return "timestamp"
    if "date" in df.columns:
        return "date"
    raise KeyError("df_test must contain a 'timestamp' or 'date' column")


def _format_date(value: object) -> str:
    return str(value)[:10]


class SimpleLongBacktester:
    """Long-only backtester pairing enter_long with the next exit action.

    Mirrors `compare_rule_vs_model.backtest_rule` PnL semantics:
        pnl_pct = (exit/entry - 1) * 100 - (commission + tax) * 2 * 100

    Round-trip costs (commission + tax) are charged twice (entry + exit).
    Trailing positions without a matching exit are dropped (no auto-close).
    """

    def __init__(
        self,
        commission: float = DEFAULT_TRADING_COST["commission"],
        tax: float = DEFAULT_TRADING_COST["tax"],
    ) -> None:
        self.commission = commission
        self.tax = tax

    def run(
        self,
        actions: Iterable[Action],
        df_test: pd.DataFrame,
        *,
        initial_cash: float = 100.0,
        fee_pct: float | None = None,
    ) -> list[Trade]:
        """Convert action stream to closed trades."""
        del initial_cash  # currently unused; kept for Protocol compatibility
        commission = self.commission
        tax = self.tax if fee_pct is None else fee_pct

        date_col = _detect_date_column(df_test)
        dates = df_test[date_col].to_numpy()
        close = df_test["close"].to_numpy()
        symbol_series = df_test["symbol"] if "symbol" in df_test.columns else None
        symbol = str(symbol_series.iloc[0]) if symbol_series is not None else ""

        trades: list[Trade] = []
        entry_idx: int | None = None
        entry_reason: str = ""

        for action in actions:
            i = action.bar_idx
            if action.type == "enter_long":
                if entry_idx is not None:
                    continue  # already in position
                entry_idx = i
                entry_reason = action.reason
            elif action.type == "exit":
                if entry_idx is None:
                    continue
                entry_price = float(close[entry_idx])
                exit_price = float(close[i])
                pnl_pct = (exit_price / entry_price - 1.0) * 100.0
                pnl_pct -= (commission + tax) * 2.0 * 100.0
                trades.append(
                    Trade(
                        entry_date=_format_date(dates[entry_idx]),
                        exit_date=_format_date(dates[i]),
                        entry_price=round(entry_price, 2),
                        exit_price=round(exit_price, 2),
                        pnl_pct=round(pnl_pct, 2),
                        holding_days=i - entry_idx,
                        entry_reason=entry_reason,
                        exit_reason=action.reason,
                        symbol=symbol,
                    )
                )
                entry_idx = None
                entry_reason = ""
            # action.type == "hold" → no-op

        return trades
