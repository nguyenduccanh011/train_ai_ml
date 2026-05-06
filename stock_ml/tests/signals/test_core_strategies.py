"""Unit tests for 11 always-on core fusion strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from src.components.base import BarContext, Position
from src.components.fusion.strategies.core import (
    AdaptiveTrailing,
    AtrStopLoss,
    FastExitLossLegacy,
    HardStopExit,
    MaCrossHybridExit,
    MinHoldProtection,
    PeakProtectDist,
    PeakProtectEma8Streak,
    ProfitLock,
    SignalHardCapExit,
    ZombieExit,
)


def _ind(n: int = 50, **overrides: Any) -> dict[str, Any]:
    base = {
        "n": n,
        "close": np.full(n, 100.0),
        "open": np.full(n, 99.5),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "volume": np.full(n, 1_000_000.0),
        "sma10": np.full(n, 100.0),
        "sma20": np.full(n, 99.0),
        "sma50": np.full(n, 98.0),
        "ema8": np.full(n, 99.5),
        "ema12": np.full(n, 99.0),
        "ema26": np.full(n, 98.0),
        "macd_line": np.full(n, 1.0),
        "macd_signal": np.full(n, 0.5),
        "macd_hist": np.full(n, 0.5),
        "atr14": np.full(n, 2.0),
        "avg_vol20": np.full(n, 1_000_000.0),
        "ret_5d": np.zeros(n),
    }
    base.update(overrides)
    return base


def _ctx(
    *,
    bar_idx: int = 10,
    trade_state: dict[str, Any] | None = None,
    indicators: dict[str, Any] | None = None,
    mods: dict[str, bool] | None = None,
    params: dict[str, Any] | None = None,
    in_position: bool = True,
) -> BarContext:
    pos: Position | None = None
    if in_position:
        pos = Position(
            symbol="TEST",
            entry_idx=0,
            entry_date=pd.Timestamp("2020-01-01"),
            entry_price=100.0,
        )
    df = pd.DataFrame({"close": [100.0] * 50})
    return BarContext(
        bar_idx=bar_idx,
        df_test=df,
        entry_signal=0 if in_position else 1,
        entry_proba=None,
        exit_signal=1 if in_position else 0,
        exit_proba=None,
        position=pos,
        config={
            "trade_state": trade_state or {},
            "indicators": indicators or _ind(),
            "mods": mods or {},
            "params": params or {},
        },
    )


class TestHardStop:
    def test_trigger(self) -> None:
        ctx = _ctx(trade_state={"cum_ret": -0.10})
        res = HardStopExit().apply(ctx)
        assert res.action == "exit"
        assert res.reason == "hard_stop"

    def test_no_trigger(self) -> None:
        ctx = _ctx(trade_state={"cum_ret": -0.05})
        assert HardStopExit().apply(ctx).action == "pass"

    def test_missing_trade_state(self) -> None:
        ctx = _ctx(trade_state={})
        assert HardStopExit().apply(ctx).action == "pass"


class TestSignalHardCap:
    def test_trigger(self) -> None:
        ctx = _ctx(trade_state={"price_cur_ret": -0.13})
        res = SignalHardCapExit().apply(ctx)
        assert res.action == "exit"
        assert res.reason == "signal_hard_cap"
        assert res.metadata["counter"] == "n_signal_hard_cap"

    def test_no_trigger(self) -> None:
        ctx = _ctx(trade_state={"price_cur_ret": -0.11})
        assert SignalHardCapExit().apply(ctx).action == "pass"


class TestFastExitLoss:
    def test_branch_a(self) -> None:
        ctx = _ctx(trade_state={"price_cur_ret": -0.06, "hold_days": 4})
        res = FastExitLossLegacy().apply(ctx)
        assert res.action == "exit"
        assert res.reason == "fast_exit_loss"

    def test_branch_b(self) -> None:
        ind = _ind(macd_hist=np.full(50, -0.5), close=np.full(50, 95.0), ema8=np.full(50, 100.0))
        ctx = _ctx(
            trade_state={"price_cur_ret": -0.04, "hold_days": 3},
            indicators=ind,
        )
        assert FastExitLossLegacy().apply(ctx).action == "exit"

    def test_no_trigger(self) -> None:
        ctx = _ctx(trade_state={"price_cur_ret": -0.02, "hold_days": 5})
        assert FastExitLossLegacy().apply(ctx).action == "pass"


class TestAtrStop:
    def test_trigger(self) -> None:
        ctx = _ctx(trade_state={"cum_ret": -0.05, "atr_stop": 0.04})
        assert AtrStopLoss().apply(ctx).reason == "stop_loss"

    def test_no_trigger(self) -> None:
        ctx = _ctx(trade_state={"cum_ret": -0.03, "atr_stop": 0.04})
        assert AtrStopLoss().apply(ctx).action == "pass"


class TestPeakProtectDist:
    def test_trigger(self) -> None:
        ind = _ind(
            close=np.full(50, 100.0),
            open=np.full(50, 102.0),  # close < open → bearish
            sma10=np.full(50, 105.0),  # close < sma10
            volume=np.full(50, 2_000_000.0),  # >1.5×avg
            avg_vol20=np.full(50, 1_000_000.0),
        )
        ctx = _ctx(trade_state={"price_max_profit": 0.25}, indicators=ind, mods={"b": True})
        res = PeakProtectDist().apply(ctx)
        assert res.action == "exit"
        assert res.reason == "peak_protect_dist"
        assert res.metadata["counter"] == "n_peak_protect"

    def test_disabled_when_mod_b_off(self) -> None:
        ind = _ind(close=np.full(50, 100.0), sma10=np.full(50, 105.0))
        ctx = _ctx(trade_state={"price_max_profit": 0.25}, indicators=ind, mods={"b": False})
        assert PeakProtectDist().apply(ctx).action == "pass"

    def test_below_threshold(self) -> None:
        ctx = _ctx(trade_state={"price_max_profit": 0.10}, mods={"b": True})
        assert PeakProtectDist().apply(ctx).action == "pass"


class TestPeakProtectEma:
    def test_streak_triggers(self) -> None:
        ind = _ind(close=np.full(50, 90.0), ema8=np.full(50, 100.0))  # close < ema8
        pos = Position(
            symbol="T",
            entry_idx=0,
            entry_date=pd.Timestamp("2020-01-01"),
            entry_price=100.0,
        )
        pos.strategy_state["consecutive_below_ema8"] = 1  # already 1 streak
        df = pd.DataFrame({"close": [90.0] * 50})
        ctx = BarContext(
            bar_idx=10,
            df_test=df,
            entry_signal=0,
            entry_proba=None,
            exit_signal=1,
            exit_proba=None,
            position=pos,
            config={
                "trade_state": {"price_max_profit": 0.20, "price_cur_ret": 0.10},
                "indicators": ind,
                "mods": {"b": True},
            },
        )
        res = PeakProtectEma8Streak().apply(ctx)
        assert res.action == "exit"

    def test_resets_when_above_ema(self) -> None:
        ind = _ind(close=np.full(50, 110.0), ema8=np.full(50, 100.0))
        pos = Position(
            symbol="T",
            entry_idx=0,
            entry_date=pd.Timestamp("2020-01-01"),
            entry_price=100.0,
        )
        pos.strategy_state["consecutive_below_ema8"] = 5
        df = pd.DataFrame({"close": [110.0] * 50})
        ctx = BarContext(
            bar_idx=10,
            df_test=df,
            entry_signal=0,
            entry_proba=None,
            exit_signal=1,
            exit_proba=None,
            position=pos,
            config={
                "trade_state": {"price_max_profit": 0.20, "price_cur_ret": 0.18},
                "indicators": ind,
                "mods": {"b": True},
            },
        )
        PeakProtectEma8Streak().apply(ctx)
        assert pos.strategy_state["consecutive_below_ema8"] == 0


class TestMaCrossHybridExit:
    def test_pass_when_preconditions_unmet(self) -> None:
        ctx = _ctx(trade_state={"trend": "moderate", "cum_ret": 0.10, "max_profit": 0.10})
        assert MaCrossHybridExit().apply(ctx).action == "pass"

    def test_exit_on_macd_cross(self) -> None:
        macd_hist = np.zeros(50)
        macd_hist[9] = 0.1
        macd_hist[10] = -0.2
        ind = _ind(macd_hist=macd_hist, close=np.full(50, 90.0), sma20=np.full(50, 100.0))
        ctx = _ctx(
            trade_state={"trend": "strong", "cum_ret": 0.06, "max_profit": 0.10},
            indicators=ind,
        )
        res = MaCrossHybridExit().apply(ctx)
        assert res.action == "exit"
        assert res.reason == "hybrid_exit"

    def test_blocks_trailing_when_inactive(self) -> None:
        ind = _ind(close=np.full(50, 105.0), sma20=np.full(50, 100.0))  # close>sma20
        ctx = _ctx(
            trade_state={"trend": "strong", "cum_ret": 0.06, "max_profit": 0.10},
            indicators=ind,
        )
        res = MaCrossHybridExit().apply(ctx)
        assert res.action == "pass"
        assert res.metadata.get("hybrid_block_trailing") is True


class TestAdaptiveTrailing:
    def test_giveback_triggers(self) -> None:
        ctx = _ctx(
            trade_state={
                "max_profit": 0.10,
                "cum_ret": 0.04,
                "trend": "weak",
            }
        )
        # giveback = 1 - 0.04/0.10 = 0.6, trail_pct=0.40 → trigger
        res = AdaptiveTrailing().apply(ctx)
        assert res.action == "exit"
        assert res.reason == "trailing_stop"

    def test_skip_when_blocked(self) -> None:
        ctx = _ctx(
            trade_state={
                "max_profit": 0.10,
                "cum_ret": 0.04,
                "trend": "weak",
                "hybrid_block_trailing": True,
            }
        )
        assert AdaptiveTrailing().apply(ctx).action == "pass"

    def test_no_trigger_below_threshold(self) -> None:
        ctx = _ctx(trade_state={"max_profit": 0.02, "cum_ret": 0.01, "trend": "weak"})
        assert AdaptiveTrailing().apply(ctx).action == "pass"


class TestProfitLock:
    def test_trigger(self) -> None:
        ctx = _ctx(trade_state={"max_profit": 0.13, "cum_ret": 0.05, "trend": "moderate"})
        assert ProfitLock().apply(ctx).reason == "profit_lock"

    def test_strong_trend_skipped(self) -> None:
        ctx = _ctx(trade_state={"max_profit": 0.13, "cum_ret": 0.05, "trend": "strong"})
        assert ProfitLock().apply(ctx).action == "pass"


class TestZombieExit:
    def test_trigger(self) -> None:
        ctx = _ctx(trade_state={"hold_days": 14, "cum_ret": 0.005, "trend": "weak"})
        assert ZombieExit().apply(ctx).reason == "zombie_exit"

    def test_strong_skipped(self) -> None:
        ctx = _ctx(trade_state={"hold_days": 14, "cum_ret": 0.005, "trend": "strong"})
        assert ZombieExit().apply(ctx).action == "pass"

    def test_short_hold_skipped(self) -> None:
        ctx = _ctx(trade_state={"hold_days": 5, "cum_ret": 0.005, "trend": "weak"})
        assert ZombieExit().apply(ctx).action == "pass"


class TestMinHoldProtection:
    def test_keep_position_for_soft_exit(self) -> None:
        ctx = _ctx(
            trade_state={
                "pending_exit_reason": "signal",
                "hold_days": 3,
                "cum_ret": 0.01,
                "atr_stop": 0.04,
            }
        )
        res = MinHoldProtection().apply(ctx)
        assert res.action == "keep_position"

    def test_bypass_for_hard_exit(self) -> None:
        ctx = _ctx(
            trade_state={
                "pending_exit_reason": "hard_stop",
                "hold_days": 3,
                "cum_ret": 0.01,
                "atr_stop": 0.04,
            }
        )
        assert MinHoldProtection().apply(ctx).action == "pass"

    def test_pass_when_hold_long_enough(self) -> None:
        ctx = _ctx(
            trade_state={
                "pending_exit_reason": "signal",
                "hold_days": 7,
                "cum_ret": 0.01,
                "atr_stop": 0.04,
            }
        )
        assert MinHoldProtection().apply(ctx).action == "pass"

    def test_pass_when_below_atr_stop(self) -> None:
        ctx = _ctx(
            trade_state={
                "pending_exit_reason": "signal",
                "hold_days": 3,
                "cum_ret": -0.05,
                "atr_stop": 0.04,
            }
        )
        assert MinHoldProtection().apply(ctx).action == "pass"

    def test_pass_when_price_loss_hits_patch_bypass(self) -> None:
        ctx = _ctx(
            trade_state={
                "pending_exit_reason": "signal",
                "hold_days": 3,
                "cum_ret": 0.01,
                "price_cur_ret": -0.07,
                "atr_stop": 0.04,
            },
            params={"patch_min_hold_loss_bypass_pct": -0.05},
        )
        assert MinHoldProtection().apply(ctx).action == "pass"

    def test_pass_when_custom_min_hold_bars_reached(self) -> None:
        ctx = _ctx(
            trade_state={
                "pending_exit_reason": "signal",
                "hold_days": 4,
                "cum_ret": 0.01,
                "atr_stop": 0.04,
            },
            params={"patch_min_hold_bars": 4},
        )
        assert MinHoldProtection().apply(ctx).action == "pass"


@pytest.mark.parametrize(
    "cls",
    [
        HardStopExit,
        SignalHardCapExit,
        FastExitLossLegacy,
        AtrStopLoss,
        PeakProtectDist,
        MaCrossHybridExit,
        AdaptiveTrailing,
        ProfitLock,
        ZombieExit,
        MinHoldProtection,
    ],
)
def test_no_trade_state_passes(cls: Any) -> None:
    """Without trade_state, every strategy must return pass (no crash)."""
    df = pd.DataFrame({"close": [100.0] * 50})
    pos = Position(
        symbol="T",
        entry_idx=0,
        entry_date=pd.Timestamp("2020-01-01"),
        entry_price=100.0,
    )
    ctx = BarContext(
        bar_idx=10,
        df_test=df,
        entry_signal=0,
        entry_proba=None,
        exit_signal=1,
        exit_proba=None,
        position=pos,
        config={"indicators": _ind()},
    )
    assert cls().apply(ctx).action == "pass"
