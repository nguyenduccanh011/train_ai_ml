from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class PeakProtectEma8Streak:
    """Mod_b second peak protect: max_profit>=0.15 + 2 days below ema8 + cur<0.75*max."""

    name: str = "peak_protect_ema"
    layer: FusionLayer = "exit_override"
    priority: int = 24

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        ind = ctx.config.get("indicators")
        mods = ctx.config.get("mods", {})
        position = ctx.position
        if not ts or not ind or position is None or not mods.get("b", True):
            return FusionResult(action="pass", reason="")
        price_max_profit = float(ts.get("price_max_profit", 0.0))
        price_cur_ret = float(ts.get("price_cur_ret", 0.0))
        if price_max_profit < 0.15:
            # Reset streak when below trigger? Legacy only resets when >= 0.15 branch active.
            return FusionResult(action="pass", reason="")

        i = ctx.bar_idx
        close = ind["close"][i]
        ema8 = ind["ema8"][i]

        state = position.strategy_state
        below_now = (not np.isnan(ema8)) and close < ema8
        if below_now:
            state["consecutive_below_ema8"] = int(state.get("consecutive_below_ema8", 0)) + 1
        else:
            state["consecutive_below_ema8"] = 0

        if state["consecutive_below_ema8"] >= 2 and price_cur_ret < price_max_profit * 0.75:
            return FusionResult(
                action="exit",
                reason="peak_protect_ema",
                metadata={"counter": "n_peak_protect"},
            )
        return FusionResult(action="pass", reason="")
