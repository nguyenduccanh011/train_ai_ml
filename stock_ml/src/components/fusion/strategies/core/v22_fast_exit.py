from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class V22FastExit:
    name: str = "v22_fast_exit"
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
        ft = _threshold(ctx)
        mt = ft + 0.02

        cond_a = price_cur_ret < ft and hold_days > 3
        cond_b = (
            price_cur_ret < mt
            and hold_days > 2
            and ind["macd_hist"][i] < 0
            and not np.isnan(ind["ema8"][i])
            and ind["close"][i] < ind["ema8"][i]
        )
        if not (cond_a or cond_b):
            return FusionResult(action="pass", reason="")

        if _trend_healthy(ctx, ts, ind) and not _vol_selling(ctx, ind):
            return FusionResult(
                action="pass",
                reason="",
                metadata={"counter": "n_fast_exit_saved"},
            )
        return FusionResult(
            action="exit",
            reason="fast_exit_loss",
            metadata={"counter": "n_fast_exit_loss"},
        )


def _threshold(ctx: BarContext) -> float:
    params = ctx.config.get("params", {})
    if _profile(ctx) == "high_beta":
        return float(
            params.get(
                "v22_fast_exit_threshold_hb",
                ctx.config.get("v22_fast_exit_threshold_hb", -0.07),
            )
        )
    return float(
        params.get(
            "v22_fast_exit_threshold_std",
            ctx.config.get("v22_fast_exit_threshold_std", -0.05),
        )
    )


def _trend_healthy(ctx: BarContext, ts: dict[str, Any], ind: dict[str, Any]) -> bool:
    params = ctx.config.get("params", {})
    skip_strong = bool(
        params.get("v22_fast_exit_skip_strong", ctx.config.get("v22_fast_exit_skip_strong", True))
    )
    i = ctx.bar_idx
    strong_uptrend = bool(ts.get("strong_uptrend", ts.get("trend") == "strong"))
    return bool(
        skip_strong
        and strong_uptrend
        and ind["macd_line"][i] > 0
        and not np.isnan(ind["sma20"][i])
        and ind["close"][i] > ind["sma20"][i] * 0.97
    )


def _vol_selling(ctx: BarContext, ind: dict[str, Any]) -> bool:
    params = ctx.config.get("params", {})
    vol_confirm = bool(
        params.get("v22_fast_exit_vol_confirm", ctx.config.get("v22_fast_exit_vol_confirm", True))
    )
    i = ctx.bar_idx
    opn = ind.get("open", ind.get("opn"))
    return bool(
        vol_confirm
        and opn is not None
        and not np.isnan(ind["avg_vol20"][i])
        and ind["volume"][i] > 1.3 * ind["avg_vol20"][i]
        and ind["close"][i] < opn[i]
    )


def _profile(ctx: BarContext) -> str:
    if ctx.symbol_profile:
        return ctx.symbol_profile
    regime_cfg = ctx.config.get("regime_cfg", {})
    profile = regime_cfg.get("profile") if isinstance(regime_cfg, dict) else None
    return str(profile or ctx.config.get("profile") or "balanced")
