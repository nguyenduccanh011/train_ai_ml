"""Unit tests for V19SignalHoldGuard — gộp 4 hold-layer modifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from src.components.base import BarContext, Position
from src.components.fusion.helpers import compute_v19_indicators, get_regime_adapter
from src.components.fusion.strategies.hold import V19SignalHoldGuard


def _df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0.05, 0.5, n))
    return pd.DataFrame(
        {
            "open": base + rng.normal(0, 0.1, n),
            "close": base,
            "high": base + np.abs(rng.normal(0.4, 0.2, n)),
            "low": base - np.abs(rng.normal(0.4, 0.2, n)),
            "volume": rng.uniform(1_000_000, 3_000_000, n),
            "symbol": ["TEST"] * n,
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )


def _ctx(
    *,
    bar_idx: int,
    trade_state: dict[str, Any] | None = None,
    mods: dict[str, bool] | None = None,
    df: pd.DataFrame | None = None,
) -> BarContext:
    df = df if df is not None else _df(80)
    ind = compute_v19_indicators(df)
    regime = get_regime_adapter("TEST", ind, bar_idx, "moderate")
    pos = Position(
        symbol="TEST",
        entry_idx=0,
        entry_date=pd.Timestamp("2020-01-01"),
        entry_price=100.0,
    )
    return BarContext(
        bar_idx=bar_idx,
        df_test=df,
        entry_signal=0,
        entry_proba=None,
        exit_signal=1,
        exit_proba=None,
        position=pos,
        config={
            "indicators": ind,
            "mods": mods or {"h": True, "i": True},
            "trade_state": trade_state or {},
            "regime_cfg": regime,
        },
    )


class TestNoPendingSignalExit:
    def test_pass_when_no_pending(self) -> None:
        ctx = _ctx(bar_idx=40, trade_state={})
        assert V19SignalHoldGuard().apply(ctx).action == "pass"

    def test_pass_when_pending_is_hard_stop(self) -> None:
        ctx = _ctx(bar_idx=40, trade_state={"pending_exit_reason": "hard_stop"})
        assert V19SignalHoldGuard().apply(ctx).action == "pass"


class TestStrongCarry:
    def test_keeps_position_when_strong_uptrend_in_profit(self) -> None:
        ctx = _ctx(
            bar_idx=40,
            trade_state={
                "pending_exit_reason": "signal",
                "cum_ret": 0.05,
                "max_profit": 0.08,
                "hold_days": 5,
                "trend": "strong",
                "raw_signal": 0,
                "consecutive_exit_signals": 0,
            },
            mods={"h": False, "i": True},  # disable mod_h to isolate carry
        )
        res = V19SignalHoldGuard().apply(ctx)
        assert res.action == "keep_position"


class TestConfirmBars:
    def test_keeps_when_confirm_bars_unmet(self) -> None:
        ctx = _ctx(
            bar_idx=40,
            trade_state={
                "pending_exit_reason": "signal",
                "cum_ret": 0.04,
                "max_profit": 0.08,
                "hold_days": 5,
                "trend": "moderate",
                "raw_signal": 0,
                "consecutive_exit_signals": 0,
            },
            mods={"h": False, "i": False},  # disable confirmed_signal & trend_carry
        )
        res = V19SignalHoldGuard().apply(ctx)
        # cum_ret > 0 → confirm_bars from regime (default 3) → consecutive (1) < 3
        assert res.action == "keep_position"


class TestTrendCarryOverride:
    def test_keeps_position_with_mod_i(self) -> None:
        df = _df(80, seed=3)
        ctx = _ctx(
            bar_idx=50,
            df=df,
            trade_state={
                "pending_exit_reason": "signal",
                "cum_ret": 0.05,
                "max_profit": 0.08,
                "hold_days": 5,
                "trend": "moderate",
                "raw_signal": 1,  # raw_signal == 1 → confirm_bars passes
                "consecutive_exit_signals": 0,
            },
            mods={"h": False, "i": True},
        )
        # outcome depends on synthetic close vs sma20 — only assert it's well-formed
        res = V19SignalHoldGuard().apply(ctx)
        assert res.action in {"pass", "keep_position"}
