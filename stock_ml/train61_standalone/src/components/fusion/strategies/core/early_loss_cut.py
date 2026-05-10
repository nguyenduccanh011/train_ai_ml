from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext


class EarlyLossCutExit:
    name: str = "early_loss_cut"
    layer: FusionLayer = "exit_override"
    priority: int = 3

    def __init__(self, threshold: float = -0.05, max_hold_days: int = 5) -> None:
        self.threshold = threshold
        self.max_hold_days = max_hold_days

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")

        mods = ctx.config.get("mods", {})
        enabled = bool(mods.get("v28_early_loss_cut", True))
        threshold = float(mods.get("v28_early_loss_cut_threshold", self.threshold))
        max_hold_days = int(mods.get("v28_early_loss_cut_days", self.max_hold_days))

        hold_days = int(ts.get("hold_days", 0))
        price_cur_ret = float(ts.get("price_cur_ret", 0.0))
        vshape_entry = bool(ts.get("vshape_entry", False))

        if enabled and hold_days <= max_hold_days and not vshape_entry:
            if price_cur_ret <= threshold:
                return FusionResult(
                    action="exit",
                    reason="v28_early_loss_cut",
                    metadata={"counter": "v28_early_loss_cut"},
                )
        return FusionResult(action="pass", reason="")
