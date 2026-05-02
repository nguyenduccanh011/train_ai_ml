"""Unit tests for FusionStack chain executor and registry."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest
from src.components.base import BarContext, FusionResult, Position
from src.components.fusion import (
    FusionStack,
    clear_registry,
    get_strategy,
    list_always_on,
    list_strategies,
    register_strategy,
)


@dataclass
class FakeStrategy:
    name: str
    layer: str
    priority: int
    result: FusionResult

    def apply(self, ctx: BarContext) -> FusionResult:
        return self.result


def _ctx(*, in_position: bool, entry_signal: int = 1, bar_idx: int = 0) -> BarContext:
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
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


# ---- Layer dispatch ----------------------------------------------------------


def test_pre_entry_skip_short_circuits_to_hold() -> None:
    skip = FakeStrategy(
        "filter_a", "pre_entry", 0, FusionResult(action="skip_entry", reason="choppy")
    )
    enter = FakeStrategy("ml_only", "entry", 0, FusionResult(action="enter", reason="ml"))
    stack = FusionStack([skip, enter])

    out = stack.run(_ctx(in_position=False))

    assert out.action.type == "hold"
    assert out.action.reason == "choppy"


def test_entry_layer_emits_enter_long_action() -> None:
    enter = FakeStrategy(
        "ml_only",
        "entry",
        0,
        FusionResult(action="enter", reason="ml", metadata={"size": 0.35}),
    )
    stack = FusionStack([enter])

    out = stack.run(_ctx(in_position=False))

    assert out.action.type == "enter_long"
    assert out.action.size == pytest.approx(0.35)
    assert out.action.reason == "ml"


def test_entry_layer_no_trigger_emits_flat_hold() -> None:
    pas = FakeStrategy("ml_only", "entry", 0, FusionResult(action="pass", reason=""))
    stack = FusionStack([pas])

    out = stack.run(_ctx(in_position=False))

    assert out.action.type == "hold"
    assert out.action.reason == "no_entry"


def test_no_pre_entry_when_signal_zero() -> None:
    """pre_entry filters MUST NOT run when entry_signal == 0."""
    called = []

    class Spy:
        name = "spy"
        layer = "pre_entry"
        priority = 0

        def apply(self, ctx: BarContext) -> FusionResult:
            called.append(ctx.bar_idx)
            return FusionResult(action="pass", reason="")

    stack = FusionStack([Spy()])
    out = stack.run(_ctx(in_position=False, entry_signal=0))

    assert called == []
    assert out.action.type == "hold"


# ---- In-position lifecycle ---------------------------------------------------


def test_hold_exit_short_circuits_exit_override() -> None:
    hold_exit = FakeStrategy(
        "ma_cross_exit", "hold", 0, FusionResult(action="exit", reason="ma_cross")
    )
    bad_override = FakeStrategy(
        "should_not_run",
        "exit_override",
        0,
        FusionResult(action="exit", reason="hard_cap"),
    )
    stack = FusionStack([hold_exit, bad_override])

    out = stack.run(_ctx(in_position=True))

    assert out.action.type == "exit"
    assert out.action.reason == "ma_cross"
    # hard_cap should not appear in reasons.
    assert "hard_cap" not in out.reasons


def test_keep_position_blocks_exit_override() -> None:
    keep = FakeStrategy(
        "signal_confirm",
        "hold",
        0,
        FusionResult(action="keep_position", reason="confirm_pending"),
    )
    override = FakeStrategy(
        "hard_cap", "exit_override", 0, FusionResult(action="exit", reason="hard_cap")
    )
    stack = FusionStack([keep, override])

    out = stack.run(_ctx(in_position=True))

    assert out.action.type == "hold"
    assert out.action.reason == "hold_position"


def test_exit_override_priority_order() -> None:
    """Lower priority runs first — first exit wins, counters reflect that."""
    high = FakeStrategy(
        "high_pri",
        "exit_override",
        1,
        FusionResult(action="exit", reason="high", metadata={"counter": "high"}),
    )
    low = FakeStrategy(
        "low_pri",
        "exit_override",
        0,
        FusionResult(action="exit", reason="low", metadata={"counter": "low"}),
    )
    stack = FusionStack([high, low])

    out = stack.run(_ctx(in_position=True))

    assert out.action.reason == "low"
    assert out.counters["low"] == 1
    assert out.counters["high"] == 0


def test_hold_layer_no_exit_falls_through_to_override() -> None:
    keep = FakeStrategy(
        "extend_hold",
        "hold",
        0,
        FusionResult(action="modify_hold", reason="extend"),
    )
    override = FakeStrategy(
        "hard_cap", "exit_override", 0, FusionResult(action="exit", reason="hard_cap")
    )
    stack = FusionStack([keep, override])

    out = stack.run(_ctx(in_position=True))

    assert out.action.type == "exit"
    assert out.action.reason == "hard_cap"


def test_position_state_carry_via_strategy_state() -> None:
    """Strategies can read/write Position.strategy_state across bars."""

    class Counter:
        name = "consec_counter"
        layer = "hold"
        priority = 0

        def apply(self, ctx: BarContext) -> FusionResult:
            assert ctx.position is not None
            n = ctx.position.strategy_state.get("consec", 0) + 1
            ctx.position.strategy_state["consec"] = n
            if n >= 3:
                return FusionResult(action="exit", reason=f"consec={n}")
            return FusionResult(action="pass", reason="")

    stack = FusionStack([Counter()])
    pos = Position(
        symbol="X",
        entry_idx=0,
        entry_date=pd.Timestamp("2024-01-01"),
        entry_price=100.0,
    )

    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    for bar in range(3):
        ctx = BarContext(
            bar_idx=bar,
            df_test=df,
            entry_signal=0,
            entry_proba=None,
            exit_signal=None,
            exit_proba=None,
            position=pos,
            config={},
        )
        out = stack.run(ctx)

    assert pos.strategy_state["consec"] == 3
    assert out.action.type == "exit"
    assert out.action.reason == "consec=3"


# ---- Counters / metadata -----------------------------------------------------


def test_counters_accumulate_across_layers() -> None:
    s1 = FakeStrategy(
        "f1",
        "pre_entry",
        0,
        FusionResult(action="pass", reason="", metadata={"counter": "f1_seen"}),
    )
    s2 = FakeStrategy(
        "e1",
        "entry",
        0,
        FusionResult(action="enter", reason="enter", metadata={"counter": "e1_fired", "size": 1.0}),
    )
    stack = FusionStack([s1, s2])

    out = stack.run(_ctx(in_position=False))

    assert out.counters["f1_seen"] == 1
    assert out.counters["e1_fired"] == 1


# ---- FusionStack constructor -------------------------------------------------


def test_invalid_layer_raises() -> None:
    bad = FakeStrategy("bad", "invalid_layer", 0, FusionResult(action="pass", reason=""))
    with pytest.raises(ValueError, match="invalid layer"):
        FusionStack([bad])


def test_strategies_listing() -> None:
    a = FakeStrategy("a", "pre_entry", 1, FusionResult(action="pass", reason=""))
    b = FakeStrategy("b", "pre_entry", 0, FusionResult(action="pass", reason=""))
    c = FakeStrategy("c", "exit_override", 0, FusionResult(action="pass", reason=""))
    stack = FusionStack([a, b, c])

    pre = stack.strategies("pre_entry")
    assert [s.name for s in pre] == ["b", "a"]  # priority sorted
    assert {s.name for s in stack.strategies()} == {"a", "b", "c"}


# ---- Registry ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_registry()
    yield
    clear_registry()
    # Restore production strategies so subsequent test modules see a populated registry.
    import importlib

    import src.components.fusion.strategies as _strats

    importlib.reload(_strats)


def _make_pass(name: str, layer: str = "pre_entry"):
    def builder(**params):
        return FakeStrategy(name, layer, 0, FusionResult(action="pass", reason=name))

    return builder


def test_register_and_get() -> None:
    register_strategy("foo", "pre_entry", _make_pass("foo"))
    strat = get_strategy("foo")
    assert strat.name == "foo"


def test_duplicate_register_raises() -> None:
    register_strategy("foo", "pre_entry", _make_pass("foo"))
    with pytest.raises(ValueError, match="already registered"):
        register_strategy("foo", "pre_entry", _make_pass("foo"))


def test_replace_register() -> None:
    register_strategy("foo", "pre_entry", _make_pass("foo"))
    register_strategy("foo", "entry", _make_pass("foo2", layer="entry"), replace=True)
    assert get_strategy("foo").layer == "entry"


def test_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError, match="unknown fusion strategy"):
        get_strategy("nonexistent")


def test_always_on_listing() -> None:
    register_strategy("core_a", "pre_entry", _make_pass("core_a"), always_on=True)
    register_strategy("core_b", "exit_override", _make_pass("core_b"), always_on=True)
    register_strategy("opt_a", "pre_entry", _make_pass("opt_a"))

    assert list_always_on() == ["core_a", "core_b"]
    assert list_always_on("pre_entry") == ["core_a"]
    assert set(list_strategies()) == {"core_a", "core_b", "opt_a"}
