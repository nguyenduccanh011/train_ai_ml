from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

ZOMBIE_BARS = 14


class ZombieExit:
    """hold>=14 + cum_ret<0.01 + trend!=strong → exit 'zombie_exit'."""

    name: str = "zombie_exit"
    layer: FusionLayer = "exit_override"
    priority: int = 29

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        hold_days = int(ts.get("hold_days", 0))
        cum_ret = float(ts.get("cum_ret", 0.0))
        trend = ts.get("trend", "weak")
        if hold_days >= ZOMBIE_BARS and cum_ret < 0.01 and trend != "strong":
            return FusionResult(action="exit", reason="zombie_exit")
        return FusionResult(action="pass", reason="")
