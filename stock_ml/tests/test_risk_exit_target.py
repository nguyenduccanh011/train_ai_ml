"""Tests for RiskExitRegressionTarget (downside-minus-upside sell-side)."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_ml.src.targets.registry import build_target
from stock_ml.src.targets.risk_exit import RiskExitRegressionTarget


def _series_df(close: list[float], symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.date_range("2020-01-01", periods=len(close), freq="D"),
            "close": close,
        }
    )


def test_imminent_drop_is_high_sell():
    # 100 -> 80: downside 0.20, upside -0.20 -> target 0.40 (sell).
    out = RiskExitRegressionTarget(horizon=1).apply(_series_df([100, 80, 80]))
    assert out["target"].iloc[0] == pytest.approx(0.40)


def test_strong_continuation_is_negative_hold():
    # 100 -> 120: downside -0.20, upside 0.20 -> target -0.40 (hold the run).
    out = RiskExitRegressionTarget(horizon=1).apply(_series_df([100, 120, 120]))
    assert out["target"].iloc[0] == pytest.approx(-0.40)


def test_sideways_is_near_zero():
    out = RiskExitRegressionTarget(horizon=2).apply(_series_df([100, 100, 100, 100]))
    assert out["target"].iloc[0] == pytest.approx(0.0)


def test_tail_is_nan():
    out = RiskExitRegressionTarget(horizon=2).apply(_series_df([10, 11, 12, 13, 14, 15]))
    assert out["target"].iloc[-2:].isna().all()
    assert out["target"].iloc[:-2].notna().all()


def test_per_symbol_isolation():
    df = pd.concat([_series_df([100, 80, 100], "AAA"), _series_df([100, 120, 100], "BBB")])
    out = RiskExitRegressionTarget(horizon=1).apply(df)
    assert out[out["symbol"] == "AAA"]["target"].iloc[0] == pytest.approx(0.40)
    assert out[out["symbol"] == "BBB"]["target"].iloc[0] == pytest.approx(-0.40)


def test_registry_builds_target():
    t = build_target({"type": "risk_exit_regression", "horizon": 7})
    assert isinstance(t, RiskExitRegressionTarget)
    assert t.horizon == 7


def test_invalid_horizon_fails_loud():
    with pytest.raises(ValueError):
        RiskExitRegressionTarget(horizon=0)
