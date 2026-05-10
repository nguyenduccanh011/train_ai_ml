from __future__ import annotations

from typing import Literal, Protocol

from src.components.base import BarContext, FusionResult

FusionLayer = Literal["pre_entry", "entry", "hold", "exit_override"]


class FusionStrategy(Protocol):
    name: str
    layer: FusionLayer
    priority: int

    def apply(self, ctx: BarContext) -> FusionResult:
        """Apply one strategy to current bar context."""
