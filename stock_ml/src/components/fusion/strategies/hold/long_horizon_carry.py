from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

_ALLOWED_REASONS = {
    "signal",
    "peak_protect_dist",
    "peak_protect_ema",
    "profit_lock",
    "trailing_stop",
}


class LongHorizonCarry:
    name: str = "long_horizon_carry"
    layer: FusionLayer = "hold"
    priority: int = 30

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        ind = ctx.config.get("indicators")
        if not ts or not ind or not _enabled(ctx):
            return FusionResult(action="pass", reason="")

        if ts.get("pending_exit_reason") not in _ALLOWED_REASONS:
            return FusionResult(action="pass", reason="")

        i = ctx.bar_idx
        if not _long_horizon_regime(ts, ind, i):
            return FusionResult(action="pass", reason="")
        if _hard_breakdown(ind, i):
            return FusionResult(action="pass", reason="")
        return FusionResult(
            action="keep_position",
            reason="long_horizon_carry",
            metadata={"counter": "n_long_horizon_carry"},
        )


def _enabled(ctx: BarContext) -> bool:
    params = ctx.config.get("params", {})
    return bool(params.get("patch_long_horizon", ctx.config.get("patch_long_horizon", False)))


def _long_horizon_regime(ts: dict[str, Any], ind: dict[str, Any], i: int) -> bool:
    return bool(
        ind["ret_60d"][i] > 0.30
        and not np.isnan(ind["sma20"][i])
        and not np.isnan(ind["sma50"][i])
        and not np.isnan(ind["sma100"][i])
        and ind["sma20"][i] > ind["sma50"][i] > ind["sma100"][i]
        and ind["days_above_sma50"][i] >= 20
        and float(ts.get("cum_ret", 0.0)) > 0.15
    )


def _hard_breakdown(ind: dict[str, Any], i: int) -> bool:
    if ind["close"][i] < ind["sma50"][i] * 0.97:
        return True
    return bool(
        i >= 3
        and ind["macd_hist"][i] < 0
        and ind["macd_hist"][i - 1] < 0
        and ind["macd_hist"][i - 2] < 0
    )
