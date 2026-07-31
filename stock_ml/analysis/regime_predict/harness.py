"""Shared harness for the regime-robust predict campaign.

Loads the champion prediction_history (raw entry/exit head scores per symbol/date)
and joins forward price outcomes + observable regime state from market.duckdb.

Every diagnostic in this campaign imports `load()` so all agents share ONE
validated data-build (no per-agent re-derivation / divergent bugs).

Usage:
    from harness import load, ic_by_year, ic_overall
    df = load()                      # merged frame, one row per (symbol,date)
    print(ic_by_year(df, 'score', 'fwd10'))
"""
import os
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import spearmanr

ROOT = "f:/PROJECTS/train_ai_ml"
PRED = f"{ROOT}/bundles/bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf/prediction_history.parquet"
MARKET = f"{ROOT}/market_data/market.duckdb"

HORIZONS = [3, 5, 10, 20, 40]


def load():
    pred = pd.read_parquet(PRED)
    pred["date"] = pd.to_datetime(pred["date"])
    syms = sorted(pred["symbol"].unique())
    con = duckdb.connect(MARKET, read_only=True)
    inlist = ",".join(repr(s) for s in syms)
    px = con.execute(
        f"""select symbol,date,open,high,low,close,volume
            from ohlcv where timeframe='1D' and symbol in ({inlist})
            and date>='2019-01-01' order by symbol,date"""
    ).fetchdf()
    con.close()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = px.groupby("symbol", group_keys=False)
    px["ret1"] = g["close"].pct_change()
    # forward horizon returns
    for h in HORIZONS:
        px[f"fwd{h}"] = g["close"].shift(-h) / px["close"] - 1.0
    # forward path quality over 10d: MFE (max favorable) / MAE (max adverse)
    fwd_hi = g["high"].apply(lambda s: s.shift(-1).rolling(10).max())
    fwd_lo = g["low"].apply(lambda s: s.shift(-1).rolling(10).min())
    px["mfe10"] = fwd_hi / px["close"] - 1.0
    px["mae10"] = fwd_lo / px["close"] - 1.0
    px["pathq10"] = px["mfe10"] + px["mae10"]  # net path skew: +ve = upside-dominant

    # per-symbol trend/vol state (causal, uses only past)
    px["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    px["ma50"] = g["close"].transform(lambda s: s.rolling(50).mean())
    px["dist_ma20"] = px["close"] / px["ma20"] - 1.0
    px["above_ma50"] = (px["close"] > px["ma50"]).astype(float)
    px["rvol20"] = g["ret1"].transform(lambda s: s.rolling(20).std())

    # equal-weight market index + observable regime state
    mkt = px.groupby("date")["ret1"].mean().rename("mkt_ret1").reset_index()
    mkt["mkt_lvl"] = (1.0 + mkt["mkt_ret1"].fillna(0)).cumprod()
    mkt["mkt_ma50"] = mkt["mkt_lvl"].rolling(50).mean()
    mkt["mkt_ma20"] = mkt["mkt_lvl"].rolling(20).mean()
    mkt["mkt_above_ma50"] = (mkt["mkt_lvl"] > mkt["mkt_ma50"]).astype(float)
    mkt["mkt_ret20"] = mkt["mkt_lvl"] / mkt["mkt_lvl"].shift(20) - 1.0
    mkt["mkt_rvol20"] = mkt["mkt_ret1"].rolling(20).std()
    # market-wide breadth: fraction of universe above own MA50
    breadth = px.groupby("date")["above_ma50"].mean().rename("breadth").reset_index()
    mkt = mkt.merge(breadth, on="date", how="left")
    # cross-sectional demeaned forward returns (market-neutral)
    for h in HORIZONS:
        px[f"fwd{h}_xs"] = px[f"fwd{h}"] - px.groupby("date")[f"fwd{h}"].transform("mean")

    pred["ens"] = pred[["score", "score2", "score3", "score4", "score5"]].mean(axis=1)
    keep = (
        ["symbol", "date", "fwd3", "fwd5", "fwd10", "fwd20", "fwd40",
         "fwd3_xs", "fwd5_xs", "fwd10_xs", "fwd20_xs", "fwd40_xs",
         "mfe10", "mae10", "pathq10", "dist_ma20", "above_ma50", "rvol20"]
    )
    m = pred.merge(px[keep], on=["symbol", "date"], how="inner")
    m = m.merge(
        mkt[["date", "mkt_above_ma50", "mkt_ret20", "mkt_rvol20", "breadth", "mkt_ret1"]],
        on="date", how="left",
    )
    m["year"] = m["date"].dt.year
    return m


def ic_overall(df, col, tgt):
    g = df.dropna(subset=[col, tgt])
    if len(g) < 50:
        return None
    return round(float(spearmanr(g[col], g[tgt]).correlation), 4)


def ic_by_year(df, col, tgt):
    out = {}
    for y, g in df.groupby("year"):
        g = g.dropna(subset=[col, tgt])
        if len(g) > 50:
            out[int(y)] = round(float(spearmanr(g[col], g[tgt]).correlation), 3)
    return out


def flips(byyear):
    """True if IC changes sign across years (regime-fragile)."""
    vals = [v for v in byyear.values()]
    return any(v > 0.03 for v in vals) and any(v < -0.03 for v in vals)
