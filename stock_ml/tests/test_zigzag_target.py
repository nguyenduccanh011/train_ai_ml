"""Tests for ZigzagPivotTarget (soft-label proximity to zigzag pivots)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ml.src.targets.registry import build_target
from stock_ml.src.targets.zigzag import ZigzagPivotTarget, _zigzag_pivots


def _series_df(close: list[float], symbol: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.date_range("2020-01-01", periods=len(close), freq="D"),
            "close": close,
        }
    )


def test_zigzag_detects_v_bottom():
    # clean V: down 100->50 then up 50->100 (50% moves, well above 10% pct)
    close = list(np.linspace(100, 50, 11)) + list(np.linspace(55, 100, 10))
    bottoms, peaks = _zigzag_pivots(np.array(close), pct=0.10)
    # the bottom is at index 10 (price 50)
    assert 10 in bottoms
    # no confirmed peak yet on the trailing up-leg (no future reversal)
    assert peaks == []


def test_zigzag_detects_peak_and_bottom_sequence():
    # up to a peak, down to a bottom, up again -> one peak + one bottom confirmed
    close = (
        list(np.linspace(50, 100, 11))  # up, peak at idx 10
        + list(np.linspace(95, 50, 10))  # down, bottom at idx 20
        + list(np.linspace(55, 90, 8))  # up (unconfirmed leg)
    )
    bottoms, peaks = _zigzag_pivots(np.array(close), pct=0.10)
    assert peaks == [10]
    assert bottoms == [20]


def test_below_threshold_moves_ignored():
    # tiny 2% wiggles never reverse by 10% -> no pivots confirmed
    rng = 100 + np.array([0, 1, -1, 2, -2, 1, 0, 1, -1, 0], dtype=float)
    bottoms, peaks = _zigzag_pivots(rng, pct=0.10)
    assert bottoms == []
    assert peaks == []


def test_soft_label_is_one_at_pivot_and_decays():
    close = list(np.linspace(100, 50, 11)) + list(np.linspace(55, 100, 10))
    df = _series_df(close)
    out = ZigzagPivotTarget(direction="bottom", pct=0.10, tau=5.0).apply(df)
    target = out["target"].to_numpy()
    # all labels in [0, 1]
    assert target.min() >= 0.0 and target.max() <= 1.0
    # peak proximity == 1 at the bottom bar (idx 10)
    assert target[10] == pytest.approx(1.0, abs=1e-6)
    # monotonic decay moving away from the bottom
    assert target[9] < target[10]
    assert target[11] < target[10]
    assert target[5] < target[9]


def test_peak_direction_labels_peaks_not_bottoms():
    close = (
        list(np.linspace(50, 100, 11))
        + list(np.linspace(95, 50, 10))
        + list(np.linspace(55, 90, 8))
    )
    df = _series_df(close)
    out = ZigzagPivotTarget(direction="peak", pct=0.10, tau=5.0).apply(df)
    target = out["target"].to_numpy()
    assert target[10] == pytest.approx(1.0, abs=1e-6)  # peak bar
    # the bottom bar (idx 20) should have low peak-proximity
    assert target[20] < 0.2


def test_only_writes_target_column():
    df = _series_df(list(np.linspace(100, 50, 11)) + list(np.linspace(55, 100, 10)))
    feature_cols_before = set(df.columns)
    out = ZigzagPivotTarget(direction="bottom", pct=0.10).apply(df)
    # exactly one new column ('target'); original columns untouched
    assert set(out.columns) - feature_cols_before == {"target"}
    for col in feature_cols_before:
        pd.testing.assert_series_equal(out[col], df[col], check_names=False)


def test_custom_target_col_for_exit_slot():
    df = _series_df(list(np.linspace(50, 100, 11)) + list(np.linspace(95, 50, 10)))
    out = ZigzagPivotTarget(direction="peak", pct=0.10, target_col="exit_target").apply(df)
    assert "exit_target" in out.columns
    assert "target" not in out.columns


def test_per_symbol_isolation():
    # two symbols with different shapes must be labeled independently
    a = _series_df(list(np.linspace(100, 50, 11)) + list(np.linspace(55, 100, 10)), "AAA")
    # BBB: up to peak, down to a bottom at idx 20, then up again so the bottom is confirmed
    b = _series_df(
        list(np.linspace(50, 100, 11))
        + list(np.linspace(95, 50, 10))
        + list(np.linspace(55, 90, 8)),
        "BBB",
    )
    df = pd.concat([a, b], ignore_index=True)
    out = ZigzagPivotTarget(direction="bottom", pct=0.10, tau=5.0).apply(df)
    aa = out[out.symbol == "AAA"]["target"].to_numpy()
    bb = out[out.symbol == "BBB"]["target"].to_numpy()
    assert aa[10] == pytest.approx(1.0, abs=1e-6)  # AAA bottom at local idx 10
    assert bb[20] == pytest.approx(1.0, abs=1e-6)  # BBB bottom at local idx 20


def test_build_target_via_registry():
    t = build_target({"type": "zigzag_pivot", "direction": "bottom", "pct": 0.12, "tau": 4.0})
    assert isinstance(t, ZigzagPivotTarget)
    assert t.direction == "bottom" and t.pct == 0.12 and t.tau == 4.0


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        ZigzagPivotTarget(direction="sideways")
    with pytest.raises(ValueError):
        ZigzagPivotTarget(pct=1.5)
    with pytest.raises(ValueError):
        ZigzagPivotTarget(tau=0)
