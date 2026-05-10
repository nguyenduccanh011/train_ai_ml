from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class AtrStopLoss:
    """ATR-based stop loss: cum_ret <= -atr_stop → exit 'stop_loss'.

    atr_stop = clip(ATR_MULT * atr14[i] / close[i], 0.025, 0.06).
    """

    name: str = "atr_stop"
    layer: FusionLayer = "exit_override"
    priority: int = 10

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        cum_ret = float(ts.get("cum_ret", 0.0))
        atr_stop = float(ts.get("atr_stop", 0.04))
        if cum_ret <= -atr_stop:
            return FusionResult(action="exit", reason="stop_loss")
        return FusionResult(action="pass", reason="")
