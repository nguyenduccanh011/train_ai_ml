"""Tests for ContinuationEntryRegressionTarget (trend-gated buy-side)."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_ml.src.targets.continuation_entry import ContinuationEntryRegressionTarget
from stock_ml.src.targets.registry import build_target


def _series_df(close: list[float], symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.date_range("2020-01-01", periods=len(close), freq="D"),
            "close": close,
        }
    )


def test_uptrend_clean_rise_is_positive():
    # 11 sits above the 2-bar SMA (10.5) and rises cleanly -> rewarded.
    close = [10, 11, 12, 13, 14]
    out = ContinuationEntryRegressionTarget(horizon=2, penalty=1.0, trend_window=2).apply(
        _series_df(close)
    )
    # t1: fwd_return=13/11-1; fwd_downside=1-12/11 (negative, no drop); base>0, trend_ok -> kept
    assert out["target"].iloc[1] > 0


def test_falling_knife_is_clipped_even_if_it_bounces():
    # Buy at 70 (below the 2-bar SMA 85) then rockets to 105: plain penalized return
    # would reward +1.0, but the trend gate clips a below-trend entry to 0.
    close = [100, 70, 105]
    out = ContinuationEntryRegressionTarget(horizon=1, penalty=1.0, trend_window=2).apply(
        _series_df(close)
    )
    assert out["target"].iloc[1] == pytest.approx(0.0)


def test_penalty_offsets_gain_reached_through_a_dip():
    # In uptrend (12>SMA 11) but the path dips to 9 before reaching 15:
    # base = (15/12-1) - 1*(1-9/12) = 0.25 - 0.25 = 0.
    close = [10, 12, 9, 15]
    out = ContinuationEntryRegressionTarget(horizon=2, penalty=1.0, trend_window=2).apply(
        _series_df(close)
    )
    assert out["target"].iloc[1] == pytest.approx(0.0)


def test_tail_is_nan():
    close = [10, 11, 12, 13, 14, 15]
    out = ContinuationEntryRegressionTarget(horizon=2, trend_window=2).apply(_series_df(close))
    assert out["target"].iloc[-2:].isna().all()
    assert out["target"].iloc[:-2].notna().all()


def test_per_symbol_isolation():
    df = pd.concat([_series_df([10, 11, 12], "AAA"), _series_df([10, 11, 12], "BBB")])
    out = ContinuationEntryRegressionTarget(horizon=1, trend_window=2).apply(df)
    assert out[out["symbol"] == "AAA"]["target"].iloc[-1:].isna().all()
    assert out[out["symbol"] == "BBB"]["target"].iloc[-1:].isna().all()


def test_registry_builds_target():
    t = build_target(
        {
            "type": "continuation_entry_regression",
            "horizon": 5,
            "penalty": 0.5,
            "trend_window": 20,
            "require_rising": True,
        }
    )
    assert isinstance(t, ContinuationEntryRegressionTarget)
    assert t.horizon == 5
    assert t.penalty == 0.5
    assert t.trend_window == 20
    assert t.require_rising is True


def test_invalid_params_fail_loud():
    with pytest.raises(ValueError):
        ContinuationEntryRegressionTarget(horizon=0)
    with pytest.raises(ValueError):
        ContinuationEntryRegressionTarget(penalty=-1)
    with pytest.raises(ValueError):
        ContinuationEntryRegressionTarget(trend_window=0)
