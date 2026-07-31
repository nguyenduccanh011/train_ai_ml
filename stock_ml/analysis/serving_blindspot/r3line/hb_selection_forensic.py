# -*- coding: utf-8 -*-
"""SELECTION-ERROR forensic (user): (A) big-LOSERS the champion SELECTED (false positive), (B) big-WINNERS
left OUTSIDE (conv-skipped or pullback-expired). Characterize entry-time features (incl NEW ones not in cs4:
extension, ATR-vol, gap, 63hi-proximity, liquidity, mkt-regime) to find a SEPARABLE signature -> a filter or
guard-rail lever. Two questions: (1) are selected big-losers separable from big-winners? (2) among conv-skipped,
are the winners separable from losers (=miscalibrated skip)? Read-only. AUC = rank separability."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
CHAMP = "template/x2_struct_to_k16preempt_cssize-69338138"

con = psycopg2.connect(**PG)
tr = pd.read_sql("SELECT symbol,entry_date,entry_signal_date,pnl_pct,exit_reason FROM run_trades WHERE run_id=%s "
                 "AND exit_reason<>'preempt' AND entry_signal_date IS NOT NULL", con, params=(CHAMP,))
sk = pd.read_sql("SELECT symbol,signal_date,pnl_pct,conv FROM run_skipped WHERE run_id=%s", con, params=(CHAMP,))
sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(CHAMP,))
con.close()
tr["d"] = pd.to_datetime(tr["entry_signal_date"]); sk["d"] = pd.to_datetime(sk["signal_date"]); sg["d"] = pd.to_datetime(sg["date"])
tr = tr.merge(sg[["symbol", "d", "score"]], on=["symbol", "d"], how="left")

# ---- entry-time features (causal, from OHLCV up to signal date) ----
cx = duckdb.connect(MARKET, read_only=True)
syms = ",".join(repr(s) for s in set(tr.symbol) | set(sk.symbol))
px = cx.execute(f"SELECT symbol,date,open,high,low,close,volume FROM ohlcv WHERE timeframe='1D' AND symbol IN ({syms}) AND date>='2018-01-01' ORDER BY symbol,date").fetchdf()
vni = cx.execute("SELECT date,close FROM ohlcv WHERE timeframe='1D' AND symbol='VN30F1M' ORDER BY date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); vni["date"] = pd.to_datetime(vni["date"])
vni["mkt50"] = vni["close"] / vni["close"].rolling(50).mean() - 1
vmap = dict(zip(vni.date, vni.mkt50))
FEAT = ["dist_ma20", "dist_ma50", "dist_ma100", "dist63hi", "dist20low", "rsi14", "ret5", "ret20", "ret60",
        "atrpct", "gap", "volz", "liq", "hi252prox", "ext_score"]
rows = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist_ma20"] = c / c.rolling(20).mean() - 1; g["dist_ma50"] = c / c.rolling(50).mean() - 1
    g["dist_ma100"] = c / c.rolling(100).mean() - 1; g["dist63hi"] = c / h.rolling(63).max() - 1
    g["dist20low"] = c / l.rolling(20).min() - 1; g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["ret5"] = c.pct_change(5); g["ret20"] = c.pct_change(20); g["ret60"] = c.pct_change(60)
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c
    g["gap"] = o / c.shift() - 1
    g["volz"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    g["liq"] = (c * v).rolling(20).mean()                       # avg turnover (liquidity)
    g["hi252prox"] = c / h.rolling(252).max() - 1               # near 52w high?
    g["ext_score"] = (c / c.rolling(20).mean() - 1) / (g["atrpct"] + 1e-9)   # extension in ATR units
    rows.append(g[["symbol", "date"] + FEAT])
F = pd.concat(rows, ignore_index=True)
FM = {(r.symbol, r.date): r for r in F.itertuples()}


def attach(df):
    out = {f: [] for f in FEAT}; out["mkt50"] = []
    for r in df.itertuples():
        f = FM.get((r.symbol, r.d))
        for c in FEAT:
            out[c].append(getattr(f, c) if f is not None else np.nan)
        out["mkt50"].append(vmap.get(r.d, np.nan))
    for c in out:
        df[c] = out[c]
    return df


tr = attach(tr); sk = attach(sk)


def auc(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    if len(a) < 20 or len(b) < 20:
        return np.nan
    allv = pd.concat([a, b]); rk = allv.rank(); n1 = len(a)
    return (rk.iloc[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(b))


ALL = FEAT + ["mkt50", "score"]
# ===== (A) SELECTED: big-loser vs big-winner =====
print(f"champion selected trades: {len(tr)}  win%={100*(tr.pnl_pct>0).mean():.0f}  avg={100*tr.pnl_pct.mean():+.1f}%", flush=True)
lo = tr[tr.pnl_pct < tr.pnl_pct.quantile(0.15)]; wi = tr[tr.pnl_pct > tr.pnl_pct.quantile(0.85)]
print(f"\n=== (A) selected BIG-LOSER (n={len(lo)}, avg {100*lo.pnl_pct.mean():+.1f}%) vs BIG-WINNER (n={len(wi)}, avg {100*wi.pnl_pct.mean():+.1f}%) ===", flush=True)
print("  feature      | loser-mean | winner-mean | |AUC-.5|", flush=True)
res = []
for f in ALL:
    res.append((f, lo[f].mean(), wi[f].mean(), abs((auc(lo[f], wi[f]) or 0.5) - 0.5)))
for f, lm, wm, s in sorted(res, key=lambda x: -(x[3] if x[3] == x[3] else -1))[:10]:
    print(f"  {f:12s} | {lm:+10.3f} | {wm:+10.3f} | {s:.3f}", flush=True)

# ===== (B) SKIPPED (conv<0.40): winner vs loser =====
skw = sk[sk.pnl_pct > 0.10]; skl = sk[sk.pnl_pct < 0.0]
print(f"\n=== (B) conv-SKIPPED: n={len(sk)}  win%={100*(sk.pnl_pct>0).mean():.0f}  avg={100*sk.pnl_pct.mean():+.1f}% "
      f"(skip WINNER n={len(skw)} vs skip LOSER n={len(skl)}) ===", flush=True)
print("  feature      | skipWin-mean | skipLoss-mean | |AUC-.5|", flush=True)
res2 = []
for f in FEAT + ["mkt50"]:
    res2.append((f, skw[f].mean(), skl[f].mean(), abs((auc(skw[f], skl[f]) or 0.5) - 0.5)))
for f, wm, lm, s in sorted(res2, key=lambda x: -(x[3] if x[3] == x[3] else -1))[:8]:
    print(f"  {f:12s} | {wm:+11.3f} | {lm:+12.3f} | {s:.3f}", flush=True)

# ===== guard-rail check: do the WORST losers cluster at feature extremes? =====
print("\n=== guard-rail: worst-5% selected losers — feature percentile vs all selected ===", flush=True)
worst = tr[tr.pnl_pct < tr.pnl_pct.quantile(0.05)]
for f in ["ext_score", "dist_ma20", "atrpct", "gap", "hi252prox", "volz", "dist63hi"]:
    p = (tr[f] < worst[f].median()).mean()  # where does worst-median sit in full dist
    print(f"  {f:12s}: worst-5% median={worst[f].median():+.3f}  (full-median={tr[f].median():+.3f})", flush=True)
print("(|AUC-.5|>0.10 = separable filter lever; guard = worst losers at extreme feature values)")
print("SELECTION_FORENSIC_DONE", flush=True)
