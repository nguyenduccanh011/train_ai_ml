# -*- coding: utf-8 -*-
"""ROOT-CAUSE: the signal-exit LOSER cohort (-2.8%, round-trip) vs the winners (max_hold/overext_trail).
Is there a SEPARABLE signature AT ENTRY that could filter the losers? Compare entry-time features + market
regime between the two cohorts; report mean-diff + simple AUC. A clean separator = a real filter lever."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
RID = "template/x2_struct_to_k16preempt_cssize-69338138"

con = psycopg2.connect(**PG)
tr = pd.read_sql("SELECT symbol,entry_signal_date,exit_reason,pnl_pct FROM run_trades WHERE run_id=%s "
                 "AND exit_reason<>'preempt' AND entry_signal_date IS NOT NULL", con, params=(RID,))
sg = pd.read_sql("SELECT symbol,date,score,exit_score FROM run_signals WHERE run_id=%s", con, params=(RID,))
con.close()
tr["d"] = pd.to_datetime(tr["entry_signal_date"]); sg["d"] = pd.to_datetime(sg["date"])
tr = tr.merge(sg[["symbol", "d", "score", "exit_score"]], on=["symbol", "d"], how="left")
# label: loser = signal-exit at a loss ; winner = max_hold/overext with profit
tr["loser"] = ((tr.exit_reason == "signal") & (tr.pnl_pct < 0)).astype(int)
tr["winner"] = ((tr.exit_reason.isin(["max_hold", "overext_trail"])) & (tr.pnl_pct > 0)).astype(int)
sub = tr[(tr.loser == 1) | (tr.winner == 1)].copy()
print(f"losers(signal<0)={int(tr.loser.sum())}  winners(mh/trail>0)={int(tr.winner.sum())}", flush=True)

# entry-time features from OHLCV + market regime
cx = duckdb.connect(MARKET, read_only=True)
syms = ",".join(repr(s) for s in tr.symbol.unique())
px = cx.execute(f"SELECT symbol,date,high,low,close,volume FROM ohlcv WHERE timeframe='1D' AND symbol IN ({syms}) AND date>='2018-06-01' ORDER BY symbol,date").fetchdf()
vni = cx.execute("SELECT date,close FROM ohlcv WHERE timeframe='1D' AND symbol='VNINDEX' ORDER BY date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); vni["date"] = pd.to_datetime(vni["date"])
vni["mkt_ma50"] = vni["close"] / vni["close"].rolling(50).mean() - 1
vni["mkt_ma200"] = vni["close"] / vni["close"].rolling(200).mean() - 1
vmap50 = dict(zip(vni.date, vni.mkt_ma50)); vmap200 = dict(zip(vni.date, vni.mkt_ma200))
feats = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist_ma20"] = c / c.rolling(20).mean() - 1; g["dist_ma50"] = c / c.rolling(50).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret5"] = c.pct_change(5); g["ret20"] = c.pct_change(20)
    g["dist63hi"] = c / h.rolling(63).max() - 1; g["dist20lo"] = c / l.rolling(20).min() - 1
    g["atrpct"] = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean() / c
    g["volz"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    feats.append(g[["symbol", "date", "dist_ma20", "dist_ma50", "rsi14", "ret5", "ret20", "dist63hi", "dist20lo", "atrpct", "volz"]])
F = pd.concat(feats, ignore_index=True)
sub = sub.merge(F, left_on=["symbol", "d"], right_on=["symbol", "date"], how="left")
sub["mkt_ma50"] = sub.d.map(vmap50); sub["mkt_ma200"] = sub.d.map(vmap200)

FCOLS = ["score", "exit_score", "dist_ma20", "dist_ma50", "rsi14", "ret5", "ret20", "dist63hi", "dist20lo", "atrpct", "volz", "mkt_ma50", "mkt_ma200"]
print("\n=== entry signature: winner-mean vs loser-mean + separation |AUC-0.5| ===")
print("feature      | winner | loser  | |AUC-.5|")
rows = []
for col in FCOLS:
    w = sub[sub.winner == 1][col].dropna(); lo = sub[sub.loser == 1][col].dropna()
    if len(w) < 30 or len(lo) < 30:
        continue
    # rank-AUC
    allv = pd.concat([w, lo]); rk = allv.rank(); n1 = len(w)
    auc = (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(lo))
    rows.append((col, w.mean(), lo.mean(), abs(auc - 0.5)))
for col, wm, lm, sep in sorted(rows, key=lambda x: -x[3]):
    print(f"{col:12s} | {wm:+6.3f} | {lm:+6.3f} | {sep:.3f}")
print("\n(|AUC-0.5| > ~0.10 = meaningfully separable at entry -> a filter lever exists)")
print("LOSER_FORENSIC_DONE")
