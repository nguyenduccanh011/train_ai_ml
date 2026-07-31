"""Tests for ReversalEntryRegressionTarget (dip-gated penalized rebound, contra-phase)."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_ml.src.targets.registry import build_target
from stock_ml.src.targets.reversal_entry import ReversalEntryRegressionTarget


def _series_df(close: list[float], symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.date_range("2020-01-01", periods=len(close), freq="D"),
            "close": close,
        }
    )


def test_dip_rebound_is_rewarded():
    # A dip below the trailing mean that then rebounds cleanly -> positive label.
    close = [100, 98, 96, 94, 92, 90, 96, 96, 96, 96]  # last bars dip then bounce
    out = ReversalEntryRegressionTarget(horizon=2, penalty=1.0, dip_window=5).apply(
        _series_df(close)
    )
    # bar idx 5 (close 90) is below its trailing mean and rebounds to 96 -> reward > 0
    assert out["target"].iloc[5] > 0


def test_extended_top_positive_is_clipped():
    # Steady uptrend: every bar is above its trailing mean (extended) -> positive base clipped to <=0.
    close = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    out = ReversalEntryRegressionTarget(horizon=2, penalty=1.0, dip_window=3).apply(
        _series_df(close)
    )
    body = out["target"].iloc[3:-2]  # past warmup, before NaN tail
    assert (body <= 1e-9).all()


def test_falling_knife_not_rewarded():
    # In a dip but keeps falling: penalty makes base negative -> low/negative label (no knife buy).
    close = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55]
    out = ReversalEntryRegressionTarget(horizon=2, penalty=1.0, dip_window=4).apply(
        _series_df(close)
    )
    body = out["target"].iloc[4:-2]
    assert (body <= 1e-9).all()


def test_tail_is_nan():
    out = ReversalEntryRegressionTarget(horizon=2, dip_window=3).apply(
        _series_df([10, 9, 11, 10, 12, 11])
    )
    assert out["target"].iloc[-2:].isna().all()


def test_min_fwd_rally_rejects_unconfirmed_dip():
    # A shallow dip that drifts up only +1% but min_fwd_rally requires +5% -> not confirmed,
    # positive label clipped away even though it's in a dip.
    close = [100, 98, 96, 94, 92, 90, 90.9, 90.9, 90.9, 90.9]  # bounce only +1% off 90
    out = ReversalEntryRegressionTarget(
        horizon=3, penalty=0.0, dip_window=5, min_fwd_rally=0.05
    ).apply(_series_df(close))
    assert out["target"].iloc[5] <= 1e-9


def test_min_fwd_rally_keeps_confirmed_bottom():
    # Same dip but a real +8% rally off 90 -> confirmed -> positive reward survives.
    close = [100, 98, 96, 94, 92, 90, 97.2, 97.2, 97.2, 97.2]
    out = ReversalEntryRegressionTarget(
        horizon=3, penalty=0.0, dip_window=5, min_fwd_rally=0.05
    ).apply(_series_df(close))
    assert out["target"].iloc[5] > 0


def test_uptrend_gate_clips_dip_in_downtrend():
    # A long decline that rebounds: with trend_window the bar is below the long MA (downtrend),
    # so the uptrend gate clips the positive reward (no knife-buying in a bear) — whereas without
    # the gate the same dip-rebound would be rewarded.
    close = [100, 95, 90, 85, 80, 75, 82, 82, 82, 82]
    gated = ReversalEntryRegressionTarget(
        horizon=3, penalty=0.0, dip_window=3, trend_window=6
    ).apply(_series_df(close))
    ungated = ReversalEntryRegressionTarget(horizon=3, penalty=0.0, dip_window=3).apply(
        _series_df(close)
    )
    assert gated["target"].iloc[5] <= 1e-9
    assert ungated["target"].iloc[5] > 0


def test_uptrend_gate_keeps_pullback_in_uptrend():
    # Rising trend with a one-bar pullback that rebounds: below short MA but above long MA -> reward.
    close = [80, 82, 84, 86, 88, 90, 88, 94, 94, 94]
    out = ReversalEntryRegressionTarget(horizon=3, penalty=0.0, dip_window=2, trend_window=6).apply(
        _series_df(close)
    )
    assert out["target"].iloc[6] > 0


def test_trend_window_must_exceed_dip_window():
    with pytest.raises(ValueError):
        ReversalEntryRegressionTarget(dip_window=50, trend_window=30)


def test_registry_builds_target():
    t = build_target(
        {"type": "reversal_entry_regression", "horizon": 10, "penalty": 0.5, "dip_window": 30}
    )
    assert isinstance(t, ReversalEntryRegressionTarget)
    assert t.horizon == 10
    assert t.penalty == 0.5
    assert t.dip_window == 30


def test_invalid_params_fail_loud():
    with pytest.raises(ValueError):
        ReversalEntryRegressionTarget(horizon=0)
    with pytest.raises(ValueError):
        ReversalEntryRegressionTarget(penalty=-1.0)
