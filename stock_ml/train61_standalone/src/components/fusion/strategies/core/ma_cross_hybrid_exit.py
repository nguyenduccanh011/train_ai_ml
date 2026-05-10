from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class MaCrossHybridExit:
    """Hybrid exit gating for strong-uptrend + profit positions.

    Trigger preconditions: strong + cum_ret>0.05 + max_profit>0.08.
    Exit when (macd bearish cross AND below sma20) OR (below sma20 AND cum<0.5*max).

    NOTE: When preconditions are met but the inner test fails, legacy still
    SKIPS the next-tier adaptive_trailing for this bar. The driver enforces
    that ordering by short-circuiting in the exit-cascade orchestration.
    """

    name: str = "ma_cross_hybrid_exit"
    layer: FusionLayer = "exit_override"
    priority: int = 26

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        ind = ctx.config.get("indicators")
        if not ts or not ind:
            return FusionResult(action="pass", reason="")
        cum_ret = float(ts.get("cum_ret", 0.0))
        max_profit = float(ts.get("max_profit", 0.0))
        trend = ts.get("trend", "weak")
        if not (trend == "strong" and cum_ret > 0.05 and max_profit > 0.08):
            return FusionResult(action="pass", reason="")

        i = ctx.bar_idx
        macd_hist = ind["macd_hist"]
        close = ind["close"][i]
        sma20 = ind["sma20"][i]
        macd_bearish = (i > 0) and (macd_hist[i] < 0 and macd_hist[i - 1] >= 0)
        price_below_ma20 = (not np.isnan(sma20)) and close < sma20

        if (macd_bearish and price_below_ma20) or (price_below_ma20 and cum_ret < max_profit * 0.5):
            return FusionResult(action="exit", reason="hybrid_exit")
        # Preconditions met but no exit → block downstream trailing.
        return FusionResult(
            action="pass",
            reason="",
            metadata={"hybrid_block_trailing": True},
        )
