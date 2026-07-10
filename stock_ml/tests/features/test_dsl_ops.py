"""Operator parity tests: DSL eval == legacy leading_v2 builder (tol 1e-9)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml.src.features.dsl.engine import EvalContext

from ._helpers import assert_close, dsl_series, golden_series, make_ohlcv

DF = make_ohlcv()


def _v2(col: str) -> pd.Series:
    return golden_series(col)


def test_ret_5d():
    assert_close(dsl_series(DF, "$close / Ref($close, 5) - 1"), _v2("ret_5d"))


def test_sma_20_ratio():
    assert_close(dsl_series(DF, "$close / Mean($close, 20) - 1"), _v2("sma_20_ratio"))


def test_ema_10_ratio():
    assert_close(dsl_series(DF, "$close / EMA($close, 10) - 1"), _v2("ema_10_ratio"))


def test_rsi_14_matches_builder():
    assert_close(dsl_series(DF, "RSI($close, 14)"), _v2("rsi_14"))


def test_rsi_7():
    assert_close(dsl_series(DF, "RSI($close, 7)"), _v2("rsi_7"))


def test_macd_hist():
    assert_close(dsl_series(DF, "MACD($close, 12, 26, 9).hist"), _v2("macd_hist"))


def test_macd_line():
    assert_close(dsl_series(DF, "MACD($close, 12, 26, 9).line"), _v2("macd_line"))


def test_atr_14_ratio():
    assert_close(dsl_series(DF, "ATR($high, $low, $close, 14) / $close"), _v2("atr_14_ratio"))


def test_adx_14():
    assert_close(dsl_series(DF, "ADX($high, $low, $close, 14).adx"), _v2("adx_14"))


def test_plus_di():
    assert_close(dsl_series(DF, "ADX($high, $low, $close, 14).plus_di"), _v2("plus_di_14"))


def test_mfi_14():
    assert_close(dsl_series(DF, "MFI($high, $low, $close, $volume, 14)"), _v2("mfi_14"))


def test_bollinger_pct():
    assert_close(dsl_series(DF, "Bollinger($close, 20, 2).pct"), _v2("bb_pct_20"))


def test_bollinger_width():
    assert_close(dsl_series(DF, "Bollinger($close, 20, 2).width"), _v2("bb_width_20"))


def test_roc_10():
    assert_close(dsl_series(DF, "ROC($close, 10)"), _v2("roc_10"))


def test_realized_vol_10():
    assert_close(dsl_series(DF, "Std(Pct($close, 1), 10)"), _v2("realized_vol_10"))


def test_dist_52w_high():
    assert_close(dsl_series(DF, "$close / Max($close, 252) - 1"), _v2("dist_52w_high"))


def test_elementwise_max_min_pairwise():
    # high_low_pct = (high - low) / close  — sanity of elementwise + division
    got = dsl_series(DF, "($high - $low) / $close")
    assert_close(got, _v2("high_low_pct"))


def test_comparison_returns_float():
    got = dsl_series(DF, "$close > Mean($close, 20)")
    vals = got.dropna().unique()
    assert set(np.unique(vals)).issubset({0.0, 1.0})


def test_csrank_is_leakage_safe_and_correct():
    # CSRank over a per-symbol return; warmup rows must stay NaN (no bfill).
    ctx = EvalContext.from_df(DF.copy())
    base = ctx.df.copy()
    base["ret20"] = base.groupby("symbol")["close"].transform(lambda x: x / x.shift(20) - 1)
    expected = base.groupby("date")["ret20"].rank(pct=True)
    expected.index = pd.MultiIndex.from_frame(base[["symbol", "date"]])

    got = dsl_series(DF, "CSRank($close / Ref($close, 20) - 1)")
    assert got.isna().any(), "warmup rows should be NaN (no forward/backward fill)"
    assert_close(got, expected)


def test_csgroupmedian_by_sector():
    df = DF.copy()
    df["sector"] = df["symbol"].map({"AAA": "X", "BBB": "X", "CCC": "Y"})
    ret = df.groupby("symbol")["close"].transform(lambda x: x / x.shift(20) - 1)
    df["ret20"] = ret
    expected = df.groupby(["date", "sector"])["ret20"].transform("median")
    expected.index = pd.MultiIndex.from_frame(df[["symbol", "date"]])

    feats = {}
    ctx = EvalContext.from_df(df.copy())
    ctx_ret = ctx.df.groupby("symbol")["close"].transform(lambda x: x / x.shift(20) - 1)
    feats["ret20"] = ctx_ret
    got = dsl_series(df, "CSGroupMedian(#ret20, by=$sector)", features=feats)
    assert_close(got, expected)
