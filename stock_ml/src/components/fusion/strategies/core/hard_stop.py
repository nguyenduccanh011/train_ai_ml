from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

HARD_STOP = 0.08


class HardStopExit:
    """cum_ret <= -HARD_STOP → exit with reason 'hard_stop'."""

    name: str = "hard_stop"
    layer: FusionLayer = "exit_override"
    priority: int = 0

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        cum_ret = float(ts.get("cum_ret", 0.0))
        if cum_ret <= -HARD_STOP:
            return FusionResult(action="exit", reason="hard_stop")
        return FusionResult(action="pass", reason="")
