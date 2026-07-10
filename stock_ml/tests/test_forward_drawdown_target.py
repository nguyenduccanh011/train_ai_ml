"""Tests for ForwardDrawdownRegressionTarget (sell-side forward downside)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ml.src.targets.forward_drawdown import ForwardDrawdownRegressionTarget
from stock_ml.src.targets.registry import build_target


def _series_df(close: list[float], symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.date_range("2020-01-01", periods=len(close), freq="D"),
            "close": close,
        }
    )


def test_drawdown_magnitude_of_coming_drop():
    # close 100 then drops to 90 over the next bars -> downside = 0.10 at t0
    close = [100, 95, 90, 92, 94]
    df = _series_df(close)
    out = ForwardDrawdownRegressionTarget(horizon=2).apply(df)
    # t0: min(close[1..2]) = 90 -> 1 - 90/100 = 0.10
    assert out["target"].iloc[0] == pytest.approx(0.10)
    # t1: min(close[2..3]) = 90 -> 1 - 90/95 ~= 0.0526
    assert out["target"].iloc[1] == pytest.approx(1 - 90 / 95)


def test_rising_series_is_non_positive():
    # strictly rising -> no forward drop -> downside <= 0 everywhere it's defined
    close = list(np.linspace(50, 100, 12))
    out = ForwardDrawdownRegressionTarget(horizon=3).apply(_series_df(close))
    defined = out["target"].dropna()
    assert (defined <= 1e-9).all()


def test_tail_is_nan_like_forward_return():
    # last `horizon` rows have no full forward window -> NaN (mirrors forward-return)
    close = [10, 11, 12, 13, 14, 15]
    out = ForwardDrawdownRegressionTarget(horizon=2).apply(_series_df(close))
    assert out["target"].iloc[-2:].isna().all()
    assert out["target"].iloc[:-2].notna().all()


def test_per_symbol_isolation():
    # two symbols must not bleed forward windows across the boundary
    df = pd.concat([_series_df([100, 80, 100], "AAA"), _series_df([100, 100, 100], "BBB")])
    out = ForwardDrawdownRegressionTarget(horizon=1).apply(df)
    aaa = out[out["symbol"] == "AAA"]["target"].to_numpy()
    bbb = out[out["symbol"] == "BBB"]["target"].to_numpy()
    assert aaa[0] == pytest.approx(0.20)  # 1 - 80/100
    assert bbb[0] == pytest.approx(0.0)


def test_registry_builds_target():
    t = build_target({"type": "forward_drawdown_regression", "horizon": 7})
    assert isinstance(t, ForwardDrawdownRegressionTarget)
    assert t.horizon == 7


def test_invalid_horizon_fails_loud():
    with pytest.raises(ValueError):
        ForwardDrawdownRegressionTarget(horizon=0)
