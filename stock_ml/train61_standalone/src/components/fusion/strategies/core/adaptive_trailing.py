from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

STRONG_TREND_TRAIL_MULT = 0.45


class AdaptiveTrailing:
    """Tiered trailing stop based on max_profit and trend strength.

    Trail tier:
        max_profit>0.25 → 0.18
        max_profit>0.15 → 0.25
        max_profit>0.08 → 0.40
        else (max_profit>0.03) → 0.65
    Trend multiplier:
        strong → ×0.45 (STRONG_TREND_TRAIL_MULT)
        moderate → ×0.7
    Trigger when giveback = 1 - cum_ret/max_profit >= trail_pct.

    Skipped if MaCrossHybridExit set hybrid_block_trailing in this bar
    (driver handles short-circuit by checking trade_state['hybrid_block_trailing']).
    """

    name: str = "adaptive_trailing"
    layer: FusionLayer = "exit_override"
    priority: int = 27

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        if ts.get("hybrid_block_trailing"):
            return FusionResult(action="pass", reason="")
        max_profit = float(ts.get("max_profit", 0.0))
        if max_profit <= 0.03:
            return FusionResult(action="pass", reason="")
        cum_ret = float(ts.get("cum_ret", 0.0))
        trend = ts.get("trend", "weak")

        if max_profit > 0.25:
            trail_pct = 0.18
        elif max_profit > 0.15:
            trail_pct = 0.25
        elif max_profit > 0.08:
            trail_pct = 0.40
        else:
            trail_pct = 0.65

        if trend == "strong":
            trail_pct *= STRONG_TREND_TRAIL_MULT
        elif trend == "moderate":
            trail_pct *= 0.7

        giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
        if giveback >= trail_pct:
            return FusionResult(action="exit", reason="trailing_stop")
        return FusionResult(action="pass", reason="")
