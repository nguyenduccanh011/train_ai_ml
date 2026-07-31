"""Causal ZigZag DSL op — leakage-safety (prefix-invariance) + sanity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import dsl_series

ZZ_OUTPUTS = [
    "last_dir",
    "last_leg_return",
    "last_leg_dur",
    "prev_leg_return",
    "prev_leg_dur",
    "bars_since_pivot",
    "return_since_pivot",
    "progress_to_deviation",
    "dist_to_confirm",
    "price_pos_in_swing",
    "max_adverse_since_pivot",
    "n_pivots",
]


def _one_symbol_df(closes: list[float], symbol: str = "AAA") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range("2020-01-01", periods=n)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": c,
            "high": c * 1.001,
            "low": c * 0.999,
            "close": c,
            "volume": 1000.0,
        }
    )


def _wave(n=200, seed=11) -> list[float]:
    rng = np.random.default_rng(seed)
    p = 100.0
    out = []
    for _ in range(n):
        p *= 1.0 + rng.normal(0.0, 0.025)
        out.append(p)
    return out


def test_prefix_invariance_no_lookahead():
    """Each output at bars < k must be identical whether computed on the full
    series or on the truncated prefix series[:k] — i.e. no future leakage."""
    closes = _wave(220)
    full = _one_symbol_df(closes)
    k = 140
    prefix = _one_symbol_df(closes[:k])
    for attr in ZZ_OUTPUTS:
        expr = f"ZigZag($close, 0.06, 3).{attr}"
        s_full = dsl_series(full, expr).reset_index(drop=True).to_numpy()[:k]
        s_pref = dsl_series(prefix, expr).reset_index(drop=True).to_numpy()
        # equal_nan: both NaN in the warmup region counts as equal
        assert np.allclose(s_full, s_pref, rtol=1e-9, atol=1e-9, equal_nan=True), (
            f"{attr}: prefix differs from full → lookahead leak"
        )


def test_progress_in_range_and_confirms_late():
    """progress is a non-negative fraction; the pivot it implies is only known
    after the reversal (a pivot at the extreme bar is NOT flagged at that bar)."""
    # clean down-then-up: bottom at idx 10, confirmed only after +>10% rise
    closes = list(np.linspace(100, 60, 11)) + list(np.linspace(63, 100, 12))
    df = _one_symbol_df(closes)
    prog = dsl_series(df, "ZigZag($close, 0.10).progress_to_deviation").reset_index(drop=True)
    bars_since = dsl_series(df, "ZigZag($close, 0.10).bars_since_pivot").reset_index(drop=True)
    valid = prog.dropna()
    assert (valid >= -1e-9).all()
    # at the true bottom bar (idx 10) the bottom is NOT yet confirmed → bars_since
    # is NaN there (no pivot known as of that bar)
    assert np.isnan(bars_since.iloc[10])
    # once price has risen >10% off the low, a bottom is confirmed and bars_since
    # becomes a positive count
    assert bars_since.dropna().iloc[-1] > 0


def test_leg_dir_sign_after_confirmation():
    """After a confirmed peak then bottom, leg_dir reflects the last pivot type."""
    closes = (
        list(np.linspace(60, 100, 11))  # up → peak at idx 10
        + list(np.linspace(95, 60, 10))  # down → bottom at idx 20
        + list(np.linspace(63, 90, 9))  # up (confirms the bottom)
    )
    df = _one_symbol_df(closes)
    leg = dsl_series(df, "ZigZag($close, 0.10).last_dir").reset_index(drop=True).dropna()
    # last confirmed pivot is the bottom (type -1)
    assert leg.iloc[-1] == -1.0


def test_multi_symbol_isolation():
    """Per-symbol walk: one symbol's pivots must not leak into another's features."""
    a = _one_symbol_df(_wave(160, seed=1), "AAA")
    b = _one_symbol_df(_wave(160, seed=2), "BBB")
    df = pd.concat([a, b], ignore_index=True)
    joint = dsl_series(df, "ZigZag($close, 0.10).return_since_pivot")
    solo_a = dsl_series(a, "ZigZag($close, 0.10).return_since_pivot")
    ja = joint.xs("AAA", level="symbol").to_numpy()
    sa = solo_a.xs("AAA", level="symbol").to_numpy()
    assert np.allclose(ja, sa, rtol=1e-9, atol=1e-9, equal_nan=True)
