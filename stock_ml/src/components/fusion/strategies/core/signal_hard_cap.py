from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

SIGNAL_HARD_CAP = 0.12


class SignalHardCapExit:
    """price_cur_ret <= -SIGNAL_HARD_CAP → exit with reason 'signal_hard_cap'."""

    name: str = "signal_hard_cap"
    layer: FusionLayer = "exit_override"
    priority: int = 5

    def apply(self, ctx: BarContext) -> FusionResult:
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        price_cur_ret = float(ts.get("price_cur_ret", 0.0))
        if price_cur_ret <= -SIGNAL_HARD_CAP:
            return FusionResult(
                action="exit",
                reason="signal_hard_cap",
                metadata={"counter": "n_signal_hard_cap"},
            )
        return FusionResult(action="pass", reason="")
