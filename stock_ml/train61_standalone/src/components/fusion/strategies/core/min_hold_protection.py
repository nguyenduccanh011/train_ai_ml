from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

MIN_HOLD = 6

# Exit reasons that BYPASS the min-hold protection (forceful exits).
_BYPASS_REASONS: frozenset[str] = frozenset(
    {
        "stop_loss",
        "hard_stop",
        "hybrid_exit",
        "peak_protect_dist",
        "peak_protect_ema",
        "fast_loss_cut",
        "signal_hard_cap",
        "fast_exit_loss",
        "exit_model",
    }
)


class MinHoldProtection:
    """Re-instate position when a soft-exit fires before MIN_HOLD bars.

    Driver-orchestrated: this strategy is consulted AFTER the exit cascade
    settles on a `pending_exit_reason`. It returns `keep_position` if the
    exit reason is "soft" and hold_days<6 and cum_ret>-atr_stop.
    """

    name: str = "min_hold_protection"
    layer: FusionLayer = "hold"
    priority: int = 99  # high priority → runs late in hold layer

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        params = ctx.config.get("params", {}) or {}
        pending = ts.get("pending_exit_reason")
        if not pending or pending in _BYPASS_REASONS:
            return FusionResult(action="pass", reason="")
        hold_days = int(ts.get("hold_days", 0))
        min_hold_bars = int(params.get("patch_min_hold_bars", MIN_HOLD))
        if hold_days >= min_hold_bars:
            return FusionResult(action="pass", reason="")
        loss_bypass = params.get("patch_min_hold_loss_bypass_pct")
        if loss_bypass is not None:
            try:
                loss_bypass = float(loss_bypass)
            except (TypeError, ValueError):
                loss_bypass = None
        if loss_bypass is not None:
            price_cur_ret = float(ts.get("price_cur_ret", ts.get("cum_ret", 0.0)))
            if price_cur_ret <= loss_bypass:
                return FusionResult(action="pass", reason="")
        cum_ret = float(ts.get("cum_ret", 0.0))
        atr_stop = float(ts.get("atr_stop", 0.04))
        if cum_ret > -atr_stop:
            return FusionResult(action="keep_position", reason="min_hold")
        return FusionResult(action="pass", reason="")
