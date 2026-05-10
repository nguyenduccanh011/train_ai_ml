from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.components.base import Action, BarContext, FusionResult
from src.components.fusion.base import FusionLayer, FusionStrategy

if TYPE_CHECKING:
    from collections.abc import Iterable


_LAYER_ORDER: tuple[FusionLayer, ...] = ("pre_entry", "entry", "hold", "exit_override")


@dataclass(slots=True)
class StackOutcome:
    action: Action
    counters: Counter[str] = field(default_factory=Counter)
    reasons: list[str] = field(default_factory=list)


class FusionStack:
    """Chain fusion strategies layer-by-layer.

    Strategies are grouped by `layer` and sorted by `priority` (lower runs first).
    Layers run in lifecycle order: pre_entry → entry → hold → exit_override.

    Short-circuit semantics per layer:
        pre_entry:      first `skip_entry` -> emit Action(hold) and stop
        entry:          first `enter` -> emit Action(enter_long) and stop
        hold:           first `exit` -> emit Action(exit) immediately
                        first `keep_position` -> block subsequent exit_override
        exit_override:  first terminal `exit` -> emit Action(exit) and stop

    Counter increments accumulate via FusionResult.metadata["counter"]; matching
    legacy `counters[]` keys preserves golden parity for regression tests.
    """

    def __init__(self, strategies: Iterable[FusionStrategy]) -> None:
        groups: dict[FusionLayer, list[FusionStrategy]] = {layer: [] for layer in _LAYER_ORDER}
        for strat in strategies:
            layer = getattr(strat, "layer", None)
            if layer not in groups:
                raise ValueError(f"strategy {strat.name!r} has invalid layer {layer!r}")
            groups[layer].append(strat)
        for layer, items in groups.items():
            items.sort(key=lambda s: getattr(s, "priority", 0))
            groups[layer] = items
        self._groups = groups

    def strategies(self, layer: FusionLayer | None = None) -> list[FusionStrategy]:
        if layer is None:
            return [s for items in self._groups.values() for s in items]
        return list(self._groups[layer])

    def run(self, ctx: BarContext) -> StackOutcome:
        counters: Counter[str] = Counter()
        reasons: list[str] = []
        in_position = ctx.position is not None

        # pre_entry: only when flat and there's an entry signal to consider.
        if not in_position and ctx.entry_signal != 0:
            for strat in self._groups["pre_entry"]:
                res = strat.apply(ctx)
                self._track(res, counters, reasons)
                if res.action == "skip_entry":
                    return StackOutcome(
                        action=Action(
                            bar_idx=ctx.bar_idx, type="hold", reason=res.reason or strat.name
                        ),
                        counters=counters,
                        reasons=reasons,
                    )

        # entry: only when flat.
        if not in_position:
            for strat in self._groups["entry"]:
                res = strat.apply(ctx)
                self._track(res, counters, reasons)
                if res.action == "enter":
                    return StackOutcome(
                        action=Action(
                            bar_idx=ctx.bar_idx,
                            type="enter_long",
                            size=float(res.metadata.get("size", 1.0)),
                            reason=res.reason or strat.name,
                            metadata=dict(res.metadata),
                        ),
                        counters=counters,
                        reasons=reasons,
                    )
            # No entry triggered → hold flat.
            return StackOutcome(
                action=Action(bar_idx=ctx.bar_idx, type="hold", reason="no_entry"),
                counters=counters,
                reasons=reasons,
            )

        # In position: hold layer first.
        keep_position = False
        for strat in self._groups["hold"]:
            res = strat.apply(ctx)
            self._track(res, counters, reasons)
            if res.action == "exit":
                return StackOutcome(
                    action=Action(
                        bar_idx=ctx.bar_idx, type="exit", reason=res.reason or strat.name
                    ),
                    counters=counters,
                    reasons=reasons,
                )
            if res.action == "keep_position":
                keep_position = True

        # exit_override layer (skipped if hold layer pinned position).
        if not keep_position:
            for strat in self._groups["exit_override"]:
                res = strat.apply(ctx)
                self._track(res, counters, reasons)
                if res.action == "exit":
                    return StackOutcome(
                        action=Action(
                            bar_idx=ctx.bar_idx, type="exit", reason=res.reason or strat.name
                        ),
                        counters=counters,
                        reasons=reasons,
                    )

        return StackOutcome(
            action=Action(bar_idx=ctx.bar_idx, type="hold", reason="hold_position"),
            counters=counters,
            reasons=reasons,
        )

    @staticmethod
    def _track(res: FusionResult, counters: Counter[str], reasons: list[str]) -> None:
        if res.reason:
            reasons.append(res.reason)
        key = res.metadata.get("counter")
        if isinstance(key, str) and key:
            counters[key] += 1
        bulk = res.metadata.get("counters")
        if isinstance(bulk, dict):
            for bulk_key, value in bulk.items():
                if isinstance(bulk_key, str) and isinstance(value, int):
                    counters[bulk_key] += value
