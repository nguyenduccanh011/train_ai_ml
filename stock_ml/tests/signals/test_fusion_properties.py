"""Property-based tests for FusionStack.

Properties verified:
P1. FusionStack with no strategies always returns action type "hold" (pass-through).
P2. A single "skip_entry" strategy in pre_entry layer always prevents entry.
P3. A single "enter" strategy in entry layer always produces enter_long when no position.
P4. A single "exit" strategy in exit_override layer always produces exit when in position.
P5. "keep_position" in hold layer blocks all exit_override strategies.
P6. pre_entry skipped when in position.
P7. Priority ordering within a layer: lower priority value wins (executes first).
P8. pre_entry skipped when entry_signal == 0.
P9. Counter accumulation via metadata["counter"].
P10. Entry layer does not fire when already in position.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
from src.components.base import BarContext, FusionResult, Position
from src.components.fusion import (
    FusionStack,
    clear_registry,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class _Strat:
    name: str
    layer: str
    priority: int
    result: FusionResult

    def apply(self, ctx: BarContext) -> FusionResult:
        return self.result


def _build(name: str, layer: str, action: str, priority: int = 0, **extras) -> _Strat:
    result = FusionResult(action=action, reason=name, **extras)
    return _Strat(name=name, layer=layer, priority=priority, result=result)


def _ctx(*, in_position: bool = False, entry_signal: int = 1, bar_idx: int = 5) -> BarContext:
    df = pd.DataFrame({"close": [100.0] * 10, "open": [99.0] * 10})
    pos = (
        Position(
            symbol="X",
            entry_idx=0,
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=100.0,
        )
        if in_position
        else None
    )
    return BarContext(
        bar_idx=bar_idx,
        df_test=df,
        entry_signal=entry_signal,
        entry_proba=None,
        exit_signal=None,
        exit_proba=None,
        position=pos,
        config={},
    )


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_registry()
    yield
    clear_registry()
    import importlib

    import src.components.fusion.strategies as _strats

    importlib.reload(_strats)


# ── Properties ───────────────────────────────────────────────────────────────


def test_p1_empty_stack_returns_hold() -> None:
    """P1: No strategies → always hold."""
    stack = FusionStack([])
    out = stack.run(_ctx(in_position=False))
    assert out.action.type == "hold"
    out2 = stack.run(_ctx(in_position=True))
    assert out2.action.type == "hold"


def test_p2_skip_entry_in_pre_entry_blocks_entry() -> None:
    """P2: skip_entry in pre_entry prevents entry."""
    filter_a = _build("filter_a", "pre_entry", "skip_entry")
    entry_a = _build("entry_a", "entry", "enter")
    stack = FusionStack([filter_a, entry_a])
    out = stack.run(_ctx(in_position=False))
    assert out.action.type == "hold"
    assert out.action.reason == "filter_a"


def test_p3_enter_strategy_fires_when_no_position() -> None:
    """P3: entry-layer 'enter' action fires when not in position."""
    entry_a = _build("entry_a", "entry", "enter")
    stack = FusionStack([entry_a])
    out = stack.run(_ctx(in_position=False))
    assert out.action.type == "enter_long"


def test_p4_exit_override_fires_when_in_position() -> None:
    """P4: exit_override 'exit' fires when in position."""
    stopper = _build("stopper", "exit_override", "exit")
    stack = FusionStack([stopper])
    out = stack.run(_ctx(in_position=True))
    assert out.action.type == "exit"


def test_p5_keep_position_blocks_exit_override() -> None:
    """P5: keep_position in hold blocks all exit_override strategies."""
    trend_carry = _build("trend_carry", "hold", "keep_position")
    stopper = _build("stopper", "exit_override", "exit")
    stack = FusionStack([trend_carry, stopper])
    out = stack.run(_ctx(in_position=True))
    assert out.action.type == "hold"


def test_p6_pre_entry_skipped_when_in_position() -> None:
    """P6: pre_entry only runs when flat."""
    fired = []

    @dataclass
    class TrackingStrat:
        name: str = "pre_entry_track"
        layer: str = "pre_entry"
        priority: int = 0

        def apply(self, ctx: BarContext) -> FusionResult:
            fired.append("pre_entry")
            return FusionResult(action="pass", reason="track")

    stack = FusionStack([TrackingStrat()])
    stack.run(_ctx(in_position=True))
    assert not fired, "pre_entry must not fire when in position"


def test_p7_priority_ordering_within_layer() -> None:
    """P7: Among exit strategies, lower priority value runs first (wins short-circuit)."""
    exit_lo = _build("exit_low", "exit_override", "exit", priority=0)
    exit_hi = _build("exit_high", "exit_override", "exit", priority=10)
    stack = FusionStack([exit_hi, exit_lo])  # intentionally reversed in list
    out = stack.run(_ctx(in_position=True))
    assert out.action.reason == "exit_low"


def test_p8_pre_entry_skipped_when_no_signal() -> None:
    """P8: pre_entry skipped when entry_signal == 0."""
    fired = []

    @dataclass
    class TrackStrat:
        name: str = "pre_entry_nosig"
        layer: str = "pre_entry"
        priority: int = 0

        def apply(self, ctx: BarContext) -> FusionResult:
            fired.append("fired")
            return FusionResult(action="pass", reason="track")

    stack = FusionStack([TrackStrat()])
    stack.run(_ctx(in_position=False, entry_signal=0))
    assert not fired, "pre_entry must not fire when entry_signal == 0"


def test_p9_counter_accumulation() -> None:
    """P9: Strategies that run contribute their counters."""

    @dataclass
    class StratA:
        name: str = "strat_a"
        layer: str = "hold"
        priority: int = 0

        def apply(self, ctx: BarContext) -> FusionResult:
            return FusionResult(action="pass", reason="a", metadata={"counter": "n_a"})

    @dataclass
    class StratB:
        name: str = "strat_b"
        layer: str = "hold"
        priority: int = 1

        def apply(self, ctx: BarContext) -> FusionResult:
            return FusionResult(action="pass", reason="b", metadata={"counter": "n_b"})

    stack = FusionStack([StratA(), StratB()])
    out = stack.run(_ctx(in_position=True))
    assert out.counters["n_a"] >= 1
    assert out.counters["n_b"] >= 1


def test_p10_entry_layer_does_not_fire_when_in_position() -> None:
    """P10: entry layer strategies don't fire when already in position."""
    fired = []

    @dataclass
    class EntryStrat:
        name: str = "entry_guard"
        layer: str = "entry"
        priority: int = 0

        def apply(self, ctx: BarContext) -> FusionResult:
            fired.append("entry")
            return FusionResult(action="enter", reason="entry_guard")

    stack = FusionStack([EntryStrat()])
    out = stack.run(_ctx(in_position=True))
    assert not fired, "Entry layer must not fire when in position"
    assert out.action.type == "hold"
