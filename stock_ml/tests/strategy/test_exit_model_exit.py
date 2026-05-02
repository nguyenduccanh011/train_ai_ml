from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from src.components.base import BarContext, Position
from src.components.fusion.strategies.core import ExitModelExit


def _make_position() -> Position:
    return Position(
        symbol="TEST",
        entry_idx=1,
        entry_date=pd.Timestamp("2023-01-01"),
        entry_price=100.0,
        size=1.0,
        holding_days=0,
        entry_close=100.0,
        entry_equity=100_000.0,
        max_equity_in_trade=100_000.0,
        max_price_in_trade=100.0,
        metadata={},
        strategy_state={},
    )


def _ctx(
    *,
    exit_signal: int | None,
    hold_days: int,
) -> BarContext:
    ind: dict[str, Any] = {
        "close": np.full(20, 100.0),
        "dates": [pd.Timestamp("2023-01-01")] * 20,
    }
    return BarContext(
        bar_idx=10,
        df_test=pd.DataFrame({"close": [100.0] * 20}),
        entry_signal=1,
        entry_proba=None,
        exit_signal=exit_signal,
        exit_proba=None,
        position=_make_position(),
        config={
            "indicators": ind,
            "mods": {},
            "trend": "neutral",
            "regime_cfg": {},
            "trade_state": {"hold_days": hold_days},
            "entry_state": {},
        },
    )


strat = ExitModelExit()


def test_exit_signal_none_returns_pass():
    ctx = _ctx(exit_signal=None, hold_days=5)
    res = strat.apply(ctx)
    assert res.action == "pass"


def test_hold_days_below_min_returns_pass():
    ctx = _ctx(exit_signal=1, hold_days=2)
    res = strat.apply(ctx)
    assert res.action == "pass"


def test_exit_signal_1_above_min_hold_returns_exit():
    ctx = _ctx(exit_signal=1, hold_days=3)
    res = strat.apply(ctx)
    assert res.action == "exit"
    assert res.reason == "exit_model"


def test_exit_signal_0_above_min_hold_returns_pass():
    ctx = _ctx(exit_signal=0, hold_days=5)
    res = strat.apply(ctx)
    assert res.action == "pass"
