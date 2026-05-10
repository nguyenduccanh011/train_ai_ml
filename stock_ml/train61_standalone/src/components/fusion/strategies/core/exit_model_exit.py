from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

EXIT_MODEL_MIN_HOLD = 3


class ExitModelExit:
    name: str = "exit_model"
    layer: FusionLayer = "exit_override"
    priority: int = 1  # after HardStopExit (0), before V22FastExit (9)

    def apply(self, ctx: BarContext) -> FusionResult:
        if ctx.exit_signal is None:
            return FusionResult(action="pass", reason="")
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        hold_days = int(ts.get("hold_days", 0))
        params = ctx.config.get("params", {})
        min_hold = int(params.get("exit_model_min_hold", EXIT_MODEL_MIN_HOLD))
        if hold_days < min_hold:
            return FusionResult(action="pass", reason="")
        if ctx.exit_signal == 1:
            return FusionResult(
                action="exit",
                reason="exit_model",
                metadata={"counter": "exit_model"},
            )
        return FusionResult(action="pass", reason="")
