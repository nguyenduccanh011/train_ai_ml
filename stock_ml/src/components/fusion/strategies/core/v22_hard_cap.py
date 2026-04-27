from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class V22HardCap:
    name: str = "v22_hard_cap"
    layer: FusionLayer = "exit_override"
    priority: int = 7

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")

        price_cur_ret = float(ts.get("price_cur_ret", 0.0))
        cap = _cap(ctx, ts)
        if price_cur_ret <= -cap:
            return FusionResult(
                action="exit",
                reason="signal_hard_cap",
                metadata={"counter": "n_signal_hard_cap"},
            )
        return FusionResult(action="pass", reason="")


def _cap(ctx: BarContext, ts: dict[str, Any]) -> float:
    params = ctx.config.get("params", {})
    adaptive = bool(
        params.get("v22_adaptive_hard_cap", ctx.config.get("v22_adaptive_hard_cap", True))
    )
    if not adaptive:
        return 0.12

    atr_ratio_now = _atr_ratio_now(ctx, ts)
    profile = _profile(ctx)
    if profile == "high_beta":
        floor = float(
            params.get("v22_hard_cap_floor_hb", ctx.config.get("v22_hard_cap_floor_hb", 0.15))
        )
        mult = float(
            params.get("v22_hard_cap_mult_hb", ctx.config.get("v22_hard_cap_mult_hb", 3.0))
        )
    else:
        floor = float(params.get("v22_hard_cap_floor", ctx.config.get("v22_hard_cap_floor", 0.12)))
        mult = float(
            params.get("v22_hard_cap_mult_std", ctx.config.get("v22_hard_cap_mult_std", 2.5))
        )
    return max(floor, mult * atr_ratio_now)


def _atr_ratio_now(ctx: BarContext, ts: dict[str, Any]) -> float:
    raw = ts.get("atr_ratio_now")
    if isinstance(raw, int | float):
        return float(raw)

    ind = ctx.config.get("indicators")
    if not ind:
        return 0.03
    i = ctx.bar_idx
    atr14 = ind.get("atr14")
    close = ind.get("close")
    if atr14 is None or close is None:
        return 0.03
    atr = atr14[i]
    price = close[i]
    if price > 0 and not np.isnan(atr):
        return float(atr / price)
    return 0.03


def _profile(ctx: BarContext) -> str:
    if ctx.symbol_profile:
        return ctx.symbol_profile
    regime_cfg = ctx.config.get("regime_cfg", {})
    profile = regime_cfg.get("profile") if isinstance(regime_cfg, dict) else None
    return str(profile or ctx.config.get("profile") or "balanced")
