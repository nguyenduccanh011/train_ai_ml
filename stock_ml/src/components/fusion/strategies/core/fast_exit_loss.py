from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class FastExitLossLegacy:
    """Legacy v19_3 fast_exit_loss.

    Trigger: (price_cur_ret<-0.05 and hold>3) OR
             (price_cur_ret<-0.03 and hold>2 and macd_hist<0 and close<ema8).
    """

    name: str = "fast_exit_loss"
    layer: FusionLayer = "exit_override"
    priority: int = 9

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        ind = ctx.config.get("indicators")
        if not ts or not ind:
            return FusionResult(action="pass", reason="")
        i = ctx.bar_idx
        price_cur_ret = float(ts.get("price_cur_ret", 0.0))
        hold_days = int(ts.get("hold_days", 0))
        macd_hist = ind["macd_hist"][i]
        close = ind["close"][i]
        ema8 = ind["ema8"][i]

        cond_a = price_cur_ret < -0.05 and hold_days > 3
        cond_b = (
            price_cur_ret < -0.03
            and hold_days > 2
            and macd_hist < 0
            and not np.isnan(ema8)
            and close < ema8
        )
        if cond_a or cond_b:
            return FusionResult(
                action="exit",
                reason="fast_exit_loss",
                metadata={"counter": "n_fast_exit_loss"},
            )
        return FusionResult(action="pass", reason="")
