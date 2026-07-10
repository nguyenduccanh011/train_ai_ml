# -*- coding: utf-8 -*-
"""TWOSIDED HEAD SCREEN — step 1: position-bar dataset for gb_x08 s42.

One row per (trade x bar) from entry+1 while position open, PLUS a 20-bar
post-exit extension (in_pos=False, labels NaN) used only for the decision-value
simulation (can the head say "keep holding" past the real exit?).

Labels (ex-post, close-based, per task spec):
  rem_mfe = max(close[t+1 .. W]) / close_t - 1,  W = min(exit_bar + 20, t + 40)
  rem_mae = min(close[t+1 .. W]) / close_t - 1
  label_end_date recorded for purge-aware fold splits.

Features at bar t (causal, <= t):
  A) the 19 exit_vol_downpress features, exact DSL port from
     stock_ml/src/features/catalog.py + dsl/ops.py semantics:
     ATR = Wilder ewm(alpha=1/14, min_periods=14); Bollinger width/pct n=20 k=2;
     TsRank = rolling(n, min_periods=n).rank(); CSRank = per-date pct rank
     (within the 61-symbol traded universe -- caveat: live system ranks within
     its panel; same universe here); market_* on the EW return index built from
     the same universe (market.py construction).
  B) position state: gain, peak (high-based), giveback, age, bars_since_peak.
  C) gate state (the EXACT force-gate inputs, replicated em02-parity):
     leg12/leg6 zigzag dir, bma20p2, market nonbull (EW<MA35 persist2),
     breadth pct_above_ma50 (full duckdb) + lowbreadth flag, drop5 zscore +
     mkt_drop flag, SNR20, and the 3 composite force flags.
"""
import json

import duckdb
import numpy as np
import pandas as pd
import psycopg2

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
POST_EXT = 20      # post-exit extension bars (simulation only)
LBL_EXIT_PAD = 20  # label window: exit_bar + 20
LBL_CAP = 40       # ... capped at t + 40

# ------------------------------------------------------------------ load
tr = pd.read_csv(EM + "/gbx08_enriched2.csv",
                 parse_dates=["entry_date", "exit_date", "entry_signal_date"])
print("trades:", len(tr), "pnl", round(tr.pnl_pct.sum(), 3))

pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                      user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()
print("universe:", len(uni_syms))

duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, open, high, low, close, volume FROM ohlcv "
    "WHERE timeframe='1D' AND symbol IN ({}) ORDER BY symbol, date"
    .format(",".join(f"'{s}'" for s in uni_syms))).df()
alls = duck.execute(
    "SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' ORDER BY symbol, date").df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
alls["date"] = pd.to_datetime(alls["date"])
bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)

# ------------------------------------------------------------------ market series (em02 parity)
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
W = 20
snr_series = rets.mean(axis=1).rolling(W).sum() / (rets.rolling(W).sum().std(axis=1) + 1e-9)

mret = rets.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
lvl = (1.0 + mret.fillna(0.0)).cumprod()
ma35 = lvl.rolling(35, min_periods=35).mean()
nonbull = ((lvl < ma35).rolling(2, min_periods=2).sum() >= 2).where(ma35.notna(), True)

pall = alls.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
ma50a = pall.rolling(50, min_periods=50).mean()
ind = (pall > ma50a)
breadth = ind.sum(axis=1) / ind.notna().sum(axis=1).clip(lower=1)

drop5 = mret.rolling(5).sum()
mu = drop5.rolling(252, min_periods=60).mean()
sd = drop5.rolling(252, min_periods=60).std()
drop5_z = (drop5 - mu) / sd.replace(0, np.nan)
mkt_drop = (drop5_z < -1.75).fillna(False)

# market_* features on EW index built from THIS universe (market.py construction:
# mean of raw per-symbol pct_change, no clip -- use unclipped mean for parity)
mret_raw = rets.mean(axis=1)
mkt_close = (1.0 + mret_raw.fillna(0.0)).cumprod() * 100.0
market_trend = (mkt_close > mkt_close.rolling(200, min_periods=200).mean()).astype(float)
mvol = mkt_close.pct_change().rolling(20, min_periods=20).std()
mvol_q90 = mvol.rolling(200, min_periods=200).quantile(0.9)
market_vol_regime = (mvol > mvol_q90).astype(float)

MKT = pd.DataFrame({
    "snr20": snr_series, "nonbull_mkt": nonbull.astype(float),
    "breadth_ma50": breadth, "lowbreadth_mkt": (breadth < 0.25).astype(float),
    "drop5_z": drop5_z, "mkt_drop": mkt_drop.astype(float),
    "market_trend": market_trend, "market_volatility_regime": market_vol_regime,
})


def causal_leg(close, pct):
    n = len(close)
    leg = np.zeros(n, dtype=np.int8)
    if n == 0:
        return leg
    direction, ext = 0, close[0]
    for i in range(1, n):
        p = close[i]
        if direction >= 0 and p > ext:
            ext = p; direction = 1
        elif direction <= 0 and p < ext:
            ext = p; direction = -1
        elif direction == 1 and p <= ext * (1.0 - pct):
            direction = -1; ext = p
        elif direction == -1 and p >= ext * (1.0 + pct):
            direction = 1; ext = p
        leg[i] = direction
    return leg


# ------------------------------------------------------------------ per-symbol features
def sym_features(g):
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"].astype(float)
    f = pd.DataFrame(index=g.index)
    pc = c.shift(1)
    ret1 = c.pct_change()
    # ATR (Wilder, dsl/ops.py _atr)
    tr_ = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr_.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    f["atr_14_ratio"] = atr14 / c
    rv10 = ret1.rolling(10, min_periods=10).std()
    f["realized_vol_10"] = rv10
    f["vol_percentile_60"] = rv10.rolling(60, min_periods=60).rank() / 60.0
    mid = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std()
    rng = 4.0 * std20
    f["bb_width_20"] = rng / mid.replace(0.0, np.nan)
    f["bb_pct_20"] = (c - (mid - 2.0 * std20)) / rng.replace(0.0, np.nan)
    f["high_low_pct_5d"] = (h.rolling(5, min_periods=5).max()
                            - l.rolling(5, min_periods=5).min()) / c
    ma5 = c.rolling(5, min_periods=5).mean()
    ma5_slope = ma5.diff(3) / ma5
    f["ma5_accel"] = ma5_slope.diff(3)
    f["dist_63d_high"] = c / c.rolling(63, min_periods=63).max() - 1
    f["dist_52w_high"] = c / c.rolling(252, min_periods=252).max() - 1
    f["sma_20_ratio"] = c / mid - 1
    v20 = v.rolling(20, min_periods=20).mean()
    f["dist_day_25"] = (((c <= pc * 0.998) * 1.0) * ((v > v.shift(1)) * 1.0)
                        ).rolling(25, min_periods=25).sum()
    f["dist_day_vol20_25"] = (((c < pc) * 1.0) * ((v > v20) * 1.0)
                              ).rolling(25, min_periods=25).sum()
    vr20 = v / (v20 + (v20 <= 0) * 1.0)
    f["down_vol_intensity_5"] = (((c.diff(1) < 0) * 1.0) * vr20
                                 ).rolling(5, min_periods=5).sum()
    f["down_vol_count_10"] = (((c < pc * 0.98) * 1.0) * ((v > v20 * 1.3) * 1.0)
                              ).rolling(10, min_periods=10).sum()
    upv = ((c > pc) * v).rolling(20, min_periods=20).sum()
    dnv = ((c <= pc) * v).rolling(20, min_periods=20).sum()
    f["updown_vol_20"] = upv / (dnv + (dnv <= 0) * 1.0)
    f["ret_20d"] = c / c.shift(20) - 1   # helper for momentum_rank
    return f


parts = []
for s, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    f = sym_features(g)
    f["symbol"], f["date"] = s, g["date"].values
    parts.append(f)
feat = pd.concat(parts, ignore_index=True)
feat["momentum_rank"] = feat.groupby("date")["ret_20d"].rank(pct=True)
feat["volatility_rank"] = feat.groupby("date")["realized_vol_10"].rank(pct=True)
feat = feat.drop(columns=["ret_20d"])
feat = feat.merge(MKT.reset_index().rename(columns={"index": "date"}), on="date", how="left")
print("feature panel:", feat.shape)

# ------------------------------------------------------------------ per-symbol arrays
SYM = {}
for sym, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    c = g["close"].to_numpy(float)
    cs = pd.Series(c)
    ma20 = cs.rolling(20, min_periods=20).mean().to_numpy()
    below20 = c < ma20
    below20[np.isnan(ma20)] = False
    bma20p2 = (pd.Series(below20).rolling(2, min_periods=2).sum().to_numpy() >= 2)
    SYM[sym] = dict(dates=pd.DatetimeIndex(g["date"]), close=c,
                    high=g["high"].to_numpy(float),
                    leg12=causal_leg(c, 0.12), leg6=causal_leg(c, 0.06), bma20p2=bma20p2)

FEAT19 = ["atr_14_ratio", "realized_vol_10", "vol_percentile_60", "bb_width_20",
          "volatility_rank", "high_low_pct_5d", "ma5_accel", "dist_63d_high",
          "dist_52w_high", "sma_20_ratio", "bb_pct_20", "market_volatility_regime",
          "market_trend", "momentum_rank", "dist_day_25", "dist_day_vol20_25",
          "down_vol_intensity_5", "down_vol_count_10", "updown_vol_20"]
feat_idx = feat.set_index(["symbol", "date"]).sort_index()

rows = []
skipped = 0
for ti, t in tr.iterrows():
    s = SYM.get(t.symbol)
    if s is None:
        skipped += 1; continue
    di = s["dates"]
    ei = di.get_indexer([t.entry_date])[0]
    xi = di.get_indexer([t.exit_date])[0]
    if ei < 0 or xi < 0:
        skipped += 1; continue
    n = len(di)
    c = s["close"]; h = s["high"]
    ep = float(t.entry_price)
    dec_last = xi - 1 if t.exit_reason != "open" else xi   # last actionable bar
    hi_cum = np.maximum.accumulate(h[ei:])                  # peak from entry
    end_bar = min(xi + POST_EXT, n - 1)
    for j in range(ei + 1, end_bar + 1):
        in_pos = j <= dec_last
        we = min(xi + LBL_EXIT_PAD, j + LBL_CAP, n - 1)
        if in_pos and we > j:
            seg = c[j + 1: we + 1]
            rem_mfe = seg.max() / c[j] - 1.0
            rem_mae = seg.min() / c[j] - 1.0
            lbl_end = di[we]
        else:
            rem_mfe = rem_mae = np.nan
            lbl_end = pd.NaT
        pk = hi_cum[j - ei] / ep - 1.0
        gain = c[j] / ep - 1.0
        peak_bar = int(np.argmax(h[ei:j + 1]))
        rows.append((ti, t.symbol, di[j], in_pos, j - ei, j - xi,
                     rem_mfe, rem_mae, lbl_end,
                     gain, pk, pk - gain, (j - ei) - peak_bar,
                     int(s["leg12"][j]), int(s["leg6"][j]), float(s["bma20p2"][j])))
cols = ["trade_id", "symbol", "date", "in_pos", "age", "bars_vs_exit",
        "rem_mfe", "rem_mae", "label_end_date",
        "gain", "peak", "giveback", "bars_since_peak",
        "leg12", "leg6", "bma20p2"]
D = pd.DataFrame(rows, columns=cols)
print("bars:", len(D), "in_pos:", int(D.in_pos.sum()), "skipped trades:", skipped)

D = D.merge(feat_idx.reset_index(), on=["symbol", "date"], how="left")
D["f_dl12"] = (D.leg12 == -1).astype(float)
D["f_nb"] = ((D.bma20p2 > 0) & (D.nonbull_mkt > 0)).astype(float)
D["f_lb"] = ((D.leg6 == -1) & (D.lowbreadth_mkt > 0)).astype(float)

# trade meta for later steps
meta = tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price",
           "pnl_pct", "exit_reason", "label", "rallied", "year_exit", "holding_days"]].copy()
meta["trade_id"] = meta.index
meta.to_parquet(EM + "/ts_trades_meta.parquet", index=False)
D.to_parquet(EM + "/ts_bars.parquet", index=False)
print("saved ts_bars.parquet / ts_trades_meta.parquet")

# ---- sanity
ip = D[D.in_pos]
print("\nsanity:")
print("  rem_mfe>=0 frac (should be ~1, max includes any up-bar):",
      round(float((ip.rem_mfe >= -1e-12).mean()), 4))
print("  rem_mfe>=rem_mae all:", bool((ip.rem_mfe >= ip.rem_mae - 1e-12).all()))
print("  label NaN in_pos:", int(ip.rem_mfe.isna().sum()))
print("  feature NaN rate (in_pos, 19f):",
      round(float(ip[FEAT19].isna().mean().mean()), 4))
print("  parity f_dl12 at decision bar vs enriched2:")
dec = D[(D.bars_vs_exit == -1)].set_index("trade_id")
tr2 = tr.copy()
tr2["trade_id"] = tr2.index
j = tr2.set_index("trade_id").join(dec[["f_dl12", "f_nb", "f_lb"]], rsuffix="_new")
j = j.dropna(subset=["f_dl12_new"])
for ccol in ["f_dl12", "f_nb", "f_lb"]:
    a = j[ccol].astype(bool)
    b = j[ccol + "_new"] > 0
    print(f"    {ccol}: match {(a == b).mean():.4f} (n={len(j)})")
