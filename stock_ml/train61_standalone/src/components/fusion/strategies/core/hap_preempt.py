from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class HapPreemptExit:
    name: str = "hap_preempt"
    layer: FusionLayer = "exit_override"
    priority: int = 8

    def __init__(
        self,
        trigger: float = 0.05,
        floor: float = -0.03,
        min_hold: int = 0,
        consec_drop: bool = False,
        min_days: int = 2,
    ) -> None:
        self.trigger = trigger
        self.floor = floor
        self.min_hold = min_hold
        self.consec_drop = consec_drop
        self.min_days = min_days

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")

        mods = ctx.config.get("mods", {})
        enabled = bool(mods.get("v32_hap_preempt", True))
        trigger = float(mods.get("v39b_hap_trigger", mods.get("v32_hap_pre_trigger", self.trigger)))
        floor = float(mods.get("v32_hap_pre_floor", self.floor))
        min_hold = int(mods.get("v39b_hap_min_hold", self.min_hold))
        consec_drop = bool(mods.get("v33_hap_consec_drop", self.consec_drop))
        min_days = int(mods.get("v33_hcd_min_days", self.min_days))

        hold_days = int(ts.get("hold_days", 0))
        price_max_profit = float(ts.get("price_max_profit", 0.0))
        price_cur_ret = float(ts.get("price_cur_ret", 0.0))
        consec_below_ema8 = int(ts.get("v33_consec_below_ema8", ts.get("consec_below_ema8", 0)))

        price_ok = enabled and price_max_profit >= trigger and price_cur_ret <= floor
        if min_hold > 0 and hold_days < min_hold:
            price_ok = False
        if price_ok and consec_drop and consec_below_ema8 < min_days:
            price_ok = False

        if price_ok:
            return FusionResult(
                action="exit",
                reason="v32_hap_preempt",
                metadata={"counter": "v32_hap_preempt"},
            )
        return FusionResult(action="pass", reason="")
