"""Unit tests for RuleSignalEntry / RuleSignalExit fusion strategies."""

from __future__ import annotations

import pandas as pd
import pytest
from src.components.base import BarContext
from src.components.fusion.strategies.rule_signal import RuleSignalEntry, RuleSignalExit


def _ctx(
    macd_hist: float,
    ma20: float,
    close: float,
    open_: float,
    *,
    in_position: bool = False,
) -> BarContext:
    df = pd.DataFrame(
        {
            "open": [open_],
            "close": [close],
            "ma20": [ma20],
            "macd_hist": [macd_hist],
        }
    )
    return BarContext(
        bar_idx=0,
        df_test=df,
        entry_signal=0 if in_position else 1,
        entry_proba=None,
        exit_signal=1 if in_position else 0,
        exit_proba=None,
        position=None,
        config={},
    )


class TestRuleSignalEntry:
    def test_buy_when_all_conditions(self) -> None:
        ctx = _ctx(macd_hist=0.5, ma20=99.0, close=101.0, open_=100.0)
        res = RuleSignalEntry().apply(ctx)
        assert res.action == "enter"
        assert res.reason == "rule_buy"
        assert res.metadata["counter"] == "n_rule_signal_entry"

    def test_pass_when_macd_zero(self) -> None:
        # Strict > 0
        ctx = _ctx(macd_hist=0.0, ma20=99.0, close=101.0, open_=100.0)
        assert RuleSignalEntry().apply(ctx).action == "pass"

    def test_pass_when_close_equals_ma20(self) -> None:
        ctx = _ctx(macd_hist=0.5, ma20=101.0, close=101.0, open_=100.0)
        assert RuleSignalEntry().apply(ctx).action == "pass"

    def test_pass_when_close_equals_open(self) -> None:
        ctx = _ctx(macd_hist=0.5, ma20=99.0, close=100.0, open_=100.0)
        assert RuleSignalEntry().apply(ctx).action == "pass"

    def test_pass_when_negative_macd(self) -> None:
        ctx = _ctx(macd_hist=-0.5, ma20=99.0, close=101.0, open_=100.0)
        assert RuleSignalEntry().apply(ctx).action == "pass"

    def test_pass_when_nan_input(self) -> None:
        df = pd.DataFrame(
            {"open": [100.0], "close": [101.0], "ma20": [float("nan")], "macd_hist": [0.5]}
        )
        ctx = BarContext(
            bar_idx=0,
            df_test=df,
            entry_signal=1,
            entry_proba=None,
            exit_signal=0,
            exit_proba=None,
            position=None,
            config={},
        )
        assert RuleSignalEntry().apply(ctx).action == "pass"


class TestRuleSignalExit:
    def test_sell_when_all_conditions(self) -> None:
        ctx = _ctx(macd_hist=-0.5, ma20=101.0, close=99.0, open_=100.0, in_position=True)
        res = RuleSignalExit().apply(ctx)
        assert res.action == "exit"
        assert res.reason == "rule_sell"
        assert res.metadata["counter"] == "n_rule_signal_exit"

    def test_pass_when_macd_zero(self) -> None:
        ctx = _ctx(macd_hist=0.0, ma20=101.0, close=99.0, open_=100.0, in_position=True)
        assert RuleSignalExit().apply(ctx).action == "pass"

    def test_pass_when_close_equals_ma20(self) -> None:
        ctx = _ctx(macd_hist=-0.5, ma20=99.0, close=99.0, open_=100.0, in_position=True)
        assert RuleSignalExit().apply(ctx).action == "pass"

    def test_pass_when_close_equals_open(self) -> None:
        ctx = _ctx(macd_hist=-0.5, ma20=101.0, close=100.0, open_=100.0, in_position=True)
        assert RuleSignalExit().apply(ctx).action == "pass"

    def test_pass_when_positive_macd(self) -> None:
        ctx = _ctx(macd_hist=0.5, ma20=101.0, close=99.0, open_=100.0, in_position=True)
        assert RuleSignalExit().apply(ctx).action == "pass"


class TestRegistry:
    def test_strategies_registered(self) -> None:
        # Importing module triggers registration.
        import src.components.fusion.strategies  # noqa: F401
        from src.components.fusion.registry import get_entry

        e_entry = get_entry("rule_signal_entry")
        assert e_entry.layer == "entry"
        e_exit = get_entry("rule_signal_exit")
        assert e_exit.layer == "exit_override"

    def test_unknown_strategy_raises(self) -> None:
        from src.components.fusion.registry import get_strategy

        with pytest.raises(KeyError, match="unknown fusion strategy"):
            get_strategy("nonexistent")
