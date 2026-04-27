from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

PROFIT_LOCK_THRESHOLD = 0.12
PROFIT_LOCK_MIN = 0.06


class ProfitLock:
    """max_profit>=0.12 + cum_ret<0.06 + trend!=strong → exit 'profit_lock'."""

    name: str = "profit_lock"
    layer: FusionLayer = "exit_override"
    priority: int = 28

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        max_profit = float(ts.get("max_profit", 0.0))
        cum_ret = float(ts.get("cum_ret", 0.0))
        trend = ts.get("trend", "weak")
        if max_profit >= PROFIT_LOCK_THRESHOLD and cum_ret < PROFIT_LOCK_MIN and trend != "strong":
            return FusionResult(action="exit", reason="profit_lock")
        return FusionResult(action="pass", reason="")
