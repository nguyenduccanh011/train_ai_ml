"""Tests for VelocityExitRegressionTarget (asymmetric-horizon downside-minus-near-upside)."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_ml.src.targets.registry import build_target
from stock_ml.src.targets.risk_exit import RiskExitRegressionTarget
from stock_ml.src.targets.velocity_exit import VelocityExitRegressionTarget


def _series_df(close: list[float], symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.date_range("2020-01-01", periods=len(close), freq="D"),
            "close": close,
        }
    )


def test_reduces_to_risk_exit_when_upside_horizon_equals_horizon():
    close = [100, 110, 95, 120, 90, 130, 100, 100]
    df = _series_df(close)
    vel = VelocityExitRegressionTarget(horizon=3, upside_horizon=3).apply(df)
    risk = RiskExitRegressionTarget(horizon=3).apply(df)
    pd.testing.assert_series_equal(vel["target"], risk["target"], check_names=False)


def test_slow_grind_sells_earlier_than_risk_exit():
    # Gain arrives only at the far end (slow grind): near-upside ~0 over U=1 -> sell-leaning,
    # while risk_exit sees the full upside and holds (lower target).
    close = [100, 100, 100, 130, 130]
    vel = VelocityExitRegressionTarget(horizon=3, upside_horizon=1).apply(_series_df(close))
    risk = RiskExitRegressionTarget(horizon=3).apply(_series_df(close))
    assert vel["target"].iloc[0] > risk["target"].iloc[0]


def test_fast_pop_is_hold():
    # 100 -> 120 then flat: downside -0.20 (never below 100), near-upside 0.20 -> target -0.40 (hold).
    out = VelocityExitRegressionTarget(horizon=2, upside_horizon=1).apply(
        _series_df([100, 120, 120, 120])
    )
    assert out["target"].iloc[0] == pytest.approx(-0.40)


def test_imminent_drop_is_high_sell():
    out = VelocityExitRegressionTarget(horizon=1, upside_horizon=1).apply(
        _series_df([100, 80, 80])
    )
    assert out["target"].iloc[0] == pytest.approx(0.40)


def test_tail_is_nan():
    out = VelocityExitRegressionTarget(horizon=2, upside_horizon=1).apply(
        _series_df([10, 11, 12, 13, 14, 15])
    )
    assert out["target"].iloc[-2:].isna().all()
    assert out["target"].iloc[:-2].notna().all()


def test_registry_builds_target():
    t = build_target({"type": "velocity_exit_regression", "horizon": 20, "upside_horizon": 5})
    assert isinstance(t, VelocityExitRegressionTarget)
    assert t.horizon == 20
    assert t.upside_horizon == 5


def test_invalid_horizon_fails_loud():
    with pytest.raises(ValueError):
        VelocityExitRegressionTarget(horizon=0)


def test_upside_horizon_exceeding_horizon_fails_loud():
    with pytest.raises(ValueError):
        VelocityExitRegressionTarget(horizon=5, upside_horizon=10)
