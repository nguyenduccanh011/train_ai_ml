from __future__ import annotations

import pandas as pd
from src.components.base import BarContext, Position
from src.components.fusion.strategies.core import EarlyLossCutExit


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
    hold_days: int,
    price_cur_ret: float,
    vshape_entry: bool = False,
    mods: dict | None = None,
) -> BarContext:
    return BarContext(
        bar_idx=10,
        df_test=pd.DataFrame({"close": [100.0] * 20}),
        entry_signal=1,
        entry_proba=None,
        exit_signal=None,
        exit_proba=None,
        position=_make_position(),
        config={
            "indicators": {},
            "mods": mods or {},
            "trend": "neutral",
            "regime_cfg": {},
            "trade_state": {
                "hold_days": hold_days,
                "price_cur_ret": price_cur_ret,
                "vshape_entry": vshape_entry,
            },
            "entry_state": {},
        },
    )


strat = EarlyLossCutExit()


def test_fires_within_hold_window():
    res = strat.apply(_ctx(hold_days=3, price_cur_ret=-0.06))
    assert res.action == "exit"
    assert res.reason == "v28_early_loss_cut"


def test_no_fire_after_hold_window():
    res = strat.apply(_ctx(hold_days=6, price_cur_ret=-0.06))
    assert res.action == "pass"


def test_no_fire_if_vshape_entry():
    res = strat.apply(_ctx(hold_days=3, price_cur_ret=-0.06, vshape_entry=True))
    assert res.action == "pass"


def test_configurable_threshold_and_days():
    res = strat.apply(
        _ctx(
            hold_days=7,
            price_cur_ret=-0.041,
            mods={
                "v28_early_loss_cut_threshold": -0.04,
                "v28_early_loss_cut_days": 10,
            },
        )
    )
    assert res.action == "exit"
    assert res.reason == "v28_early_loss_cut"
