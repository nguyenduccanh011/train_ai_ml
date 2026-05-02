from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from src.components.base import BarContext, Position
from src.components.fusion.strategies.core import V22FastExit


def _ind(n: int = 20, **overrides: Any) -> dict[str, Any]:
    base = {
        "close": np.full(n, 95.0),
        "open": np.full(n, 96.0),
        "volume": np.full(n, 1_000_000.0),
        "avg_vol20": np.full(n, 1_000_000.0),
        "ema8": np.full(n, 100.0),
        "sma20": np.full(n, 100.0),
        "macd_line": np.full(n, 1.0),
        "macd_hist": np.full(n, -0.5),
    }
    base.update(overrides)
    return base


def _ctx(
    *,
    trade_state: dict[str, Any],
    indicators: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    profile: str | None = None,
) -> BarContext:
    return BarContext(
        bar_idx=10,
        df_test=pd.DataFrame({"close": [95.0] * 20}),
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
            "params": params or {},
        },
        symbol_profile=profile,
    )


def test_v22_fast_exit_standard_loss_branch() -> None:
    ctx = _ctx(trade_state={"price_cur_ret": -0.06, "hold_days": 4}, profile="balanced")

    res = V22FastExit().apply(ctx)

    assert res.action == "exit"
    assert res.reason == "fast_exit_loss"
    assert res.metadata["counter"] == "n_fast_exit_loss"


def test_v22_fast_exit_high_beta_momentum_branch() -> None:
    ctx = _ctx(trade_state={"price_cur_ret": -0.06, "hold_days": 3}, profile="high_beta")

    assert V22FastExit().apply(ctx).reason == "fast_exit_loss"


def test_v22_fast_exit_saved_by_healthy_strong_trend() -> None:
    ind = _ind(close=np.full(20, 99.0), open=np.full(20, 100.0), volume=np.full(20, 1_000_000.0))
    ctx = _ctx(
        trade_state={"price_cur_ret": -0.06, "hold_days": 4, "strong_uptrend": True},
        indicators=ind,
        profile="balanced",
    )

    res = V22FastExit().apply(ctx)

    assert res.action == "pass"
    assert res.metadata["counter"] == "n_fast_exit_saved"


def test_v22_fast_exit_volume_selling_overrides_strong_trend_save() -> None:
    ind = _ind(close=np.full(20, 99.0), open=np.full(20, 100.0), volume=np.full(20, 1_400_000.0))
    ctx = _ctx(
        trade_state={"price_cur_ret": -0.06, "hold_days": 4, "strong_uptrend": True},
        indicators=ind,
        profile="balanced",
    )

    assert V22FastExit().apply(ctx).reason == "fast_exit_loss"


def test_v22_fast_exit_pass_when_threshold_not_met() -> None:
    ctx = _ctx(trade_state={"price_cur_ret": -0.03, "hold_days": 5}, profile="balanced")

    assert V22FastExit().apply(ctx).action == "pass"
