"""Shared fixtures/helpers for DSL feature tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stock_ml.src.features.dsl.engine import Engine, EvalContext
from stock_ml.src.features.dsl.parser import parse

# Frozen snapshot of the legacy leading_v2 builder on make_ohlcv(), captured before
# the builders were deleted (Phase 5). The DSL parity gate compares against this.
_GOLDEN_PATH = Path(__file__).parent / "golden" / "leading_v2_golden.parquet"


def make_ohlcv(n_per: int = 160, symbols=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    """Deterministic synthetic OHLCV across several symbols (no internal NaNs)."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=n_per)
    rows = []
    for si, sym in enumerate(symbols):
        price = 100.0 + si * 25.0
        closes = np.empty(n_per)
        for i in range(n_per):
            price *= 1.0 + rng.normal(0.0, 0.02)
            closes[i] = price
        high = closes * (1.0 + np.abs(rng.normal(0.0, 0.012, n_per)))
        low = closes * (1.0 - np.abs(rng.normal(0.0, 0.012, n_per)))
        open_ = closes * (1.0 + rng.normal(0.0, 0.006, n_per))
        vol = rng.integers(100_000, 1_000_000, n_per).astype(float)
        for i in range(n_per):
            rows.append((sym, dates[i], open_[i], high[i], low[i], closes[i], vol[i]))
    return pd.DataFrame(
        rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"]
    )


def dsl_series(df: pd.DataFrame, expr: str, features=None) -> pd.Series:
    """Evaluate an expression, return result indexed by (symbol, date)."""
    ctx = EvalContext.from_df(df.copy(), features)
    val = Engine().eval(parse(expr), ctx)
    out = ctx.df[["symbol", "date"]].copy()
    out["v"] = np.asarray(val, dtype="float64")
    return out.set_index(["symbol", "date"])["v"]


def golden_frame() -> pd.DataFrame:
    """The frozen leading_v2 golden, indexed by (symbol, date)."""
    return pd.read_parquet(_GOLDEN_PATH).set_index(["symbol", "date"])


def golden_series(col: str) -> pd.Series:
    """One leading_v2 golden column indexed by (symbol, date)."""
    return golden_frame()[col]


def assert_close(a: pd.Series, b: pd.Series, tol: float = 1e-9) -> None:
    a2, b2 = a.align(b, join="inner")
    assert len(a2) > 0, "no overlapping rows to compare"
    av = a2.to_numpy(dtype="float64")
    bv = b2.to_numpy(dtype="float64")
    ok = np.allclose(av, bv, rtol=tol, atol=tol, equal_nan=True)
    if not ok:
        diff = np.nanmax(np.abs(av - bv))
        raise AssertionError(f"series differ; max abs diff={diff:g}")
