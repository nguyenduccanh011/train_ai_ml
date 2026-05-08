from __future__ import annotations

from typing import Any

import pandas as pd

from src.components.base import Trade

_DATE_COLUMNS = ("date", "timestamp")
_PRICE_COLUMN = "close"
_TRADE_FIELDS = frozenset(
    {
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "pnl_pct",
        "holding_days",
        "entry_reason",
        "exit_reason",
        "symbol",
        "entry_symbol",
    }
)


class LegacyBacktestAdapter:
    def convert(
        self,
        result: dict[str, Any],
        *,
        df_test: pd.DataFrame | None = None,
        symbol: str = "",
    ) -> list[Trade]:
        close_values = df_test[_PRICE_COLUMN].to_numpy() if df_test is not None else None
        date_values = self._date_values(df_test)
        trades = result.get("trades") or ()
        return [
            self._convert_trade(
                trade,
                close_values=close_values,
                date_values=date_values,
                symbol=symbol,
            )
            for trade in trades
        ]

    def _convert_trade(
        self,
        trade: dict[str, Any],
        *,
        close_values: Any,
        date_values: Any,
        symbol: str,
    ) -> Trade:
        entry_day = int(trade["entry_day"])
        exit_day = int(trade["exit_day"])
        entry_price = self._price_at(close_values, entry_day, trade.get("entry_price"))
        exit_price = self._price_at(close_values, exit_day, trade.get("exit_price"))

        return Trade(
            entry_date=trade.get("entry_date", self._date_at(date_values, entry_day)),
            exit_date=trade.get("exit_date", self._date_at(date_values, exit_day)),
            entry_price=round(float(entry_price), 2),
            exit_price=round(float(exit_price), 2),
            pnl_pct=float(trade["pnl_pct"]),
            holding_days=int(trade.get("holding_days", exit_day - entry_day)),
            entry_reason=str(trade.get("entry_reason", "legacy_entry")),
            exit_reason=str(trade.get("exit_reason", "legacy_exit")),
            symbol=str(trade.get("symbol", trade.get("entry_symbol", symbol))),
            metadata={k: v for k, v in trade.items() if k not in _TRADE_FIELDS},
        )

    def _date_values(self, df_test: pd.DataFrame | None) -> Any:
        if df_test is None:
            return None
        for date_col in _DATE_COLUMNS:
            if date_col in df_test.columns:
                return df_test[date_col].to_numpy()
        return None

    def _price_at(self, close_values: Any, idx: int, fallback: Any) -> float:
        if fallback is not None:
            return float(fallback)
        if close_values is None:
            return float("nan")
        return float(close_values[idx])

    def _date_at(self, date_values: Any, idx: int) -> str:
        if date_values is None:
            return ""
        ts = pd.Timestamp(date_values[idx])
        if ts.time() == pd.Timestamp(ts.date()).time():
            return ts.date().isoformat()
        return ts.isoformat()
