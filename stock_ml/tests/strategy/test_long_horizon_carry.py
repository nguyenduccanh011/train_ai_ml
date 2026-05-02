from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from src.components.base import BarContext, Position
from src.components.fusion.strategies.hold import LongHorizonCarry


def _ind(n: int = 20, **overrides: Any) -> dict[str, Any]:
    base = {
        "close": np.full(n, 115.0),
        "sma20": np.full(n, 120.0),
        "sma50": np.full(n, 110.0),
        "sma100": np.full(n, 100.0),
        "ret_60d": np.full(n, 0.31),
        "days_above_sma50": np.full(n, 20),
        "macd_hist": np.full(n, 0.1),
    }
    base.update(overrides)
    return base


def _ctx(
    *,
    trade_state: dict[str, Any],
    indicators: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> BarContext:
    return BarContext(
        bar_idx=10,
        df_test=pd.DataFrame({"close": [115.0] * 20}),
        entry_signal=0,
        entry_proba=None,
        exit_signal=1,
        exit_proba=None,
        position=Position(
            symbol="TEST",
            entry_idx=0,
            entry_date=pd.Timestamp("2020-01-01"),
            entry_price=100.0,
        ),
        config={
            "trade_state": trade_state,
            "indicators": indicators or _ind(),
            "params": params or {"patch_long_horizon": True},
        },
    )


def test_long_horizon_carry_keeps_allowed_exit_in_regime() -> None:
    ctx = _ctx(trade_state={"pending_exit_reason": "profit_lock", "cum_ret": 0.16})

    res = LongHorizonCarry().apply(ctx)

    assert res.action == "keep_position"
    assert res.reason == "long_horizon_carry"
    assert res.metadata["counter"] == "n_long_horizon_carry"


def test_long_horizon_carry_passes_when_disabled() -> None:
    ctx = _ctx(
        trade_state={"pending_exit_reason": "profit_lock", "cum_ret": 0.16},
        params={"patch_long_horizon": False},
    )

    assert LongHorizonCarry().apply(ctx).action == "pass"


def test_long_horizon_carry_passes_for_unprotected_exit_reason() -> None:
    ctx = _ctx(trade_state={"pending_exit_reason": "hard_stop", "cum_ret": 0.16})

    assert LongHorizonCarry().apply(ctx).action == "pass"


def test_long_horizon_carry_passes_outside_regime() -> None:
    ind = _ind(ret_60d=np.full(20, 0.29))
    ctx = _ctx(trade_state={"pending_exit_reason": "profit_lock", "cum_ret": 0.16}, indicators=ind)

    assert LongHorizonCarry().apply(ctx).action == "pass"


def test_long_horizon_carry_passes_on_price_breakdown() -> None:
    ind = _ind(close=np.full(20, 100.0), sma50=np.full(20, 110.0))
    ctx = _ctx(trade_state={"pending_exit_reason": "profit_lock", "cum_ret": 0.16}, indicators=ind)

    assert LongHorizonCarry().apply(ctx).action == "pass"


def test_long_horizon_carry_passes_on_three_negative_macd_bars() -> None:
    macd_hist = np.full(20, 0.1)
    macd_hist[8:11] = -0.1
    ctx = _ctx(
        trade_state={"pending_exit_reason": "profit_lock", "cum_ret": 0.16},
        indicators=_ind(macd_hist=macd_hist),
    )

    assert LongHorizonCarry().apply(ctx).action == "pass"
