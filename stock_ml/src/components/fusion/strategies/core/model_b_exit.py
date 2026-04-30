from __future__ import annotations

from typing import TYPE_CHECKING

from src.components.base import FusionResult
from src.components.fusion.base import FusionLayer

if TYPE_CHECKING:
    from src.components.base import BarContext

MODEL_B_MIN_HOLD = 3


class ModelBExit:
    """Exit when Model B (exit model) predicts sell after minimum hold period.

    Reads ctx.exit_signal populated by the runner from y_pred_exit[i-1].
    Returns pass when exit_signal is None (no exit model configured).
    """

    name: str = "model_b_exit"
    layer: FusionLayer = "exit_override"
    priority: int = 1  # after HardStopExit (0), before V22FastExit (9)

    def apply(self, ctx: BarContext) -> FusionResult:
        if ctx.exit_signal is None:
            return FusionResult(action="pass", reason="")
        ts = ctx.config.get("trade_state")
        if not ts:
            return FusionResult(action="pass", reason="")
        hold_days = int(ts.get("hold_days", 0))
        if hold_days < MODEL_B_MIN_HOLD:
            return FusionResult(action="pass", reason="")
        if ctx.exit_signal == 1:
            return FusionResult(
                action="exit",
                reason="model_b_exit",
                metadata={"counter": "model_b_exit"},
            )
        return FusionResult(action="pass", reason="")
