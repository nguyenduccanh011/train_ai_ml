from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

_REQUIRED_COLS = ("macd_hist", "ma20", "close", "open")


def _read_bar(ctx: BarContext) -> tuple[float, float, float, float] | None:
    row = ctx.df_test.iloc[ctx.bar_idx]
    try:
        macd_hist = float(row["macd_hist"])
        ma20 = float(row["ma20"])
        close = float(row["close"])
        open_ = float(row["open"])
    except (KeyError, ValueError, TypeError):
        return None
    if any(v != v for v in (macd_hist, ma20, close, open_)):
        return None
    return macd_hist, ma20, close, open_


class RuleSignalEntry:
    """Long entry on MACD_hist > 0 AND Close > MA20 AND Close > Open."""

    name: str = "rule_signal_entry"
    layer: FusionLayer = "entry"
    priority: int = 0

    def apply(self, ctx: BarContext) -> FusionResult:
        bar = _read_bar(ctx)
        if bar is None:
            return FusionResult(action="pass", reason="")
        macd_hist, ma20, close, open_ = bar
        if macd_hist > 0 and close > ma20 and close > open_:
            return FusionResult(
                action="enter",
                reason="rule_buy",
                metadata={"counter": "n_rule_signal_entry"},
            )
        return FusionResult(action="pass", reason="")


class RuleSignalExit:
    """Long exit on MACD_hist < 0 AND Close < MA20 AND Close < Open."""

    name: str = "rule_signal_exit"
    layer: FusionLayer = "exit_override"
    priority: int = 0

    def apply(self, ctx: BarContext) -> FusionResult:
        bar = _read_bar(ctx)
        if bar is None:
            return FusionResult(action="pass", reason="")
        macd_hist, ma20, close, open_ = bar
        if macd_hist < 0 and close < ma20 and close < open_:
            return FusionResult(
                action="exit",
                reason="rule_sell",
                metadata={"counter": "n_rule_signal_exit"},
            )
        return FusionResult(action="pass", reason="")
