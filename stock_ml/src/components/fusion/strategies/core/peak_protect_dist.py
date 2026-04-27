from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class PeakProtectDist:
    """Mod_b first peak protect: max_profit>=0.20 + below sma10 + heavy vol + bearish."""

    name: str = "peak_protect_dist"
    layer: FusionLayer = "exit_override"
    priority: int = 23

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        ind = ctx.config.get("indicators")
        mods = ctx.config.get("mods", {})
        if not ts or not ind or not mods.get("b", True):
            return FusionResult(action="pass", reason="")
        price_max_profit = float(ts.get("price_max_profit", 0.0))
        if price_max_profit < 0.20:
            return FusionResult(action="pass", reason="")
        i = ctx.bar_idx
        close = ind["close"][i]
        opn = ind["open"][i]
        sma10 = ind["sma10"][i]
        avg_vol20 = ind["avg_vol20"][i]
        volume = ind["volume"][i]
        price_below_sma10 = not np.isnan(sma10) and close < sma10
        heavy_vol = not np.isnan(avg_vol20) and volume > 1.5 * avg_vol20
        bearish_candle = close < opn
        if price_below_sma10 and heavy_vol and bearish_candle:
            return FusionResult(
                action="exit",
                reason="peak_protect_dist",
                metadata={"counter": "n_peak_protect"},
            )
        return FusionResult(action="pass", reason="")
