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
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])


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
    """DSL vs legacy-builder parity.

    Where the DSL emits a value it must equal the golden to ``tol`` (strict). The one accepted
    divergence: the DSL leaves the leading warmup as NaN while the legacy builder fabricated a
    neutral fill there (e.g. RSI/MFI -> 50.0). That is an intentional honesty improvement — the
    training pipeline drops NaN-feature rows anyway (subsumed by the longer sma_200 warmup), so the
    fabricated warmup never reached the model. We still guard it tightly: a DSL NaN that the golden
    fills is allowed ONLY as a contiguous leading run per symbol; a DSL NaN mid-series (a real gap)
    or a DSL value where the golden is NaN both fail.
    """
    a2, b2 = a.align(b, join="inner")
    assert len(a2) > 0, "no overlapping rows to compare"
    av = a2.to_numpy(dtype="float64")
    bv = b2.to_numpy(dtype="float64")
    na, nb = np.isnan(av), np.isnan(bv)

    both = ~na & ~nb
    assert both.any(), "no finite overlap to compare"
    if not np.allclose(av[both], bv[both], rtol=tol, atol=tol):
        diff = np.nanmax(np.abs(av[both] - bv[both]))
        raise AssertionError(f"finite values differ; max abs diff={diff:g}")

    # DSL value where the golden withheld one -> DSL fabricating; never allowed.
    dsl_extra = ~na & nb
    assert not dsl_extra.any(), (
        f"DSL emits {int(dsl_extra.sum())} value(s) where the legacy builder is NaN"
    )

    # DSL NaN where the golden has a value: allowed only as each symbol's leading warmup run.
    dsl_only_nan = na & ~nb
    if dsl_only_nan.any():
        syms = a2.index.get_level_values(0).to_numpy()
        for sym in pd.unique(syms):
            m = dsl_only_nan[syms == sym]
            if m.any():
                last = int(np.max(np.nonzero(m)[0]))
                assert m[: last + 1].all(), (
                    f"{sym}: DSL NaN appears mid-series (not leading warmup) — a real gap, not a fill diff"
                )
