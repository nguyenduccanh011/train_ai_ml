from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from src.components.base import BarContext, Position
from src.components.fusion.strategies.core import V22HardCap


def _ind(n: int = 20, **overrides: Any) -> dict[str, Any]:
    base = {
        "close": np.full(n, 100.0),
        "atr14": np.full(n, 3.0),
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
        df_test=pd.DataFrame({"close": [100.0] * 20}),
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


def test_v22_standard_profile_uses_floor_when_atr_cap_is_lower() -> None:
    ctx = _ctx(trade_state={"price_cur_ret": -0.12}, profile="balanced")

    res = V22HardCap().apply(ctx)

    assert res.action == "exit"
    assert res.reason == "signal_hard_cap"
    assert res.metadata["counter"] == "n_signal_hard_cap"


def test_v22_high_beta_profile_uses_wider_adaptive_cap() -> None:
    ind = _ind(atr14=np.full(20, 10.0), close=np.full(20, 100.0))
    ctx = _ctx(trade_state={"price_cur_ret": -0.25}, indicators=ind, profile="high_beta")

    assert V22HardCap().apply(ctx).action == "pass"


def test_v22_high_beta_profile_exits_after_adaptive_cap() -> None:
    ind = _ind(atr14=np.full(20, 10.0), close=np.full(20, 100.0))
    ctx = _ctx(trade_state={"price_cur_ret": -0.31}, indicators=ind, profile="high_beta")

    assert V22HardCap().apply(ctx).reason == "signal_hard_cap"


def test_v22_flat_hard_cap_when_adaptive_disabled() -> None:
    ctx = _ctx(
        trade_state={"price_cur_ret": -0.121},
        params={"v22_adaptive_hard_cap": False},
        profile="high_beta",
    )

    assert V22HardCap().apply(ctx).reason == "signal_hard_cap"
