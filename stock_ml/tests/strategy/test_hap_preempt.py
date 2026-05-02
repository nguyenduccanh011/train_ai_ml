from __future__ import annotations

import pandas as pd
from src.components.base import BarContext, Position
from src.components.fusion.strategies.core import HapPreemptExit


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
    hold_days: int = 5,
    price_max_profit: float = 0.06,
    price_cur_ret: float = -0.04,
    consec_below_ema8: int = 0,
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
                "price_max_profit": price_max_profit,
                "price_cur_ret": price_cur_ret,
                "v33_consec_below_ema8": consec_below_ema8,
            },
            "entry_state": {},
        },
    )


strat = HapPreemptExit()


def test_fires_when_max_profit_then_drops():
    res = strat.apply(_ctx())
    assert res.action == "exit"
    assert res.reason == "v32_hap_preempt"


def test_no_fire_if_max_profit_too_low():
    res = strat.apply(_ctx(price_max_profit=0.04))
    assert res.action == "pass"


def test_no_fire_if_cur_ret_above_floor():
    res = strat.apply(_ctx(price_cur_ret=-0.02))
    assert res.action == "pass"


def test_min_hold_guard():
    res = strat.apply(_ctx(hold_days=3, mods={"v39b_hap_min_hold": 5}))
    assert res.action == "pass"


def test_consec_drop_guard():
    blocked = strat.apply(
        _ctx(consec_below_ema8=1, mods={"v33_hap_consec_drop": True, "v33_hcd_min_days": 2})
    )
    fired = strat.apply(
        _ctx(consec_below_ema8=2, mods={"v33_hap_consec_drop": True, "v33_hcd_min_days": 2})
    )
    assert blocked.action == "pass"
    assert fired.action == "exit"
    assert fired.reason == "v32_hap_preempt"
