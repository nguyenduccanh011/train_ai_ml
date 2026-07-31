# -*- coding: utf-8 -*-
"""SEQUENTIAL peak analysis (before + at + after) to find the EARLIEST separable point of top vs pullback.
At each LOCAL HIGH during an in-profit hold: label top = no higher close in remaining hold. Features:
 BEFORE (run-up shape: velocity/accel/updays/vol-trend), AT (rsi/ext/wick), AFTER-confirmation (drop by
 bar +1/+2/+3/+5 = causal at that bar). Q: does run-up shape predict top? how early does drop-confirm
 separate? does confirmation work better conditional on run-up shape? Base model 3185."""
from __future__ import annotations
import os, sys, warnings, statistics
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np, duckdb
from scripts.run_template import run_template_experiment
from sklearn.metrics import roc_auc_score

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
W = 5  # local-high window

cx = duckdb.connect(DUCK, read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close,volume FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); S = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True); c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    S[s] = dict(c=c.values, h=h.values, l=l.values, v=v.values, rsi=(100 - 100 / (1 + up / (dn + 1e-9))).values,
                ma20=c.rolling(20).mean().values, ma5=c.rolling(5).mean().values, avgv=v.rolling(20).mean().values,
                idx={d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])})

con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
con.close()

rows = []
for r in tr.itertuples():
    a = S.get(r.symbol)
    if a is None:
        continue
    ei = a["idx"].get(str(r.entry_date)[:10]); xi = a["idx"].get(str(r.exit_date)[:10])
    if ei is None or xi is None or xi <= ei + 3:
        continue
    c = a["c"]; ep = r.entry_price
    for b in range(ei + 2, min(xi - 1, len(c) - 6)):
        if b - W < 0 or b < 11:
            continue
        if c[b] / ep - 1.0 < 0.03:                                   # đang lời ≥3%
            continue
        if c[b] < np.max(c[b - W:b + 1]) - 1e-9:                     # b là đỉnh cục bộ (cao nhất W phiên gần)
            continue
        top = 1 if np.max(c[b + 1:xi + 1]) <= c[b] * 1.001 else 0    # không phá đỉnh về sau = đỉnh thật
        hi, lo, cl = a["h"][b], a["l"][b], a["c"][b]
        v5 = c[b] / c[b - 5] - 1; v10 = c[b] / c[b - 10] - 1; accel = v5 - (c[b - 5] / c[b - 10] - 1)
        updays = int(np.sum(np.diff(c[b - 5:b + 1]) > 0))
        voltr = (a["v"][b - 2:b + 1].mean()) / (a["avgv"][b] + 1e-9)
        rows.append(dict(top=top, ep_gain=c[b] / ep - 1.0,
                         v5=v5, v10=v10, accel=accel, updays=updays, uwick=(hi - cl) / (hi - lo + 1e-9),
                         rsi=a["rsi"][b], ext20=c[b] / a["ma20"][b] - 1.0, voltr=voltr,
                         d1=c[b + 1] / c[b] - 1.0, d2=min(c[b + 1], c[b + 2]) / c[b] - 1.0,
                         d3=min(c[b + 1:b + 4]) / c[b] - 1.0, d5=min(c[b + 1:b + 6]) / c[b] - 1.0,
                         brk_ma5_1=1.0 if c[b + 1] < a["ma5"][b + 1] else 0.0))
D = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
print(f"=== SEQUENTIAL peak: {len(D)} đỉnh-cục-bộ (đang lời≥3%) | P(đỉnh thật, không phá về sau)={100*D.top.mean():.0f}% ===", flush=True)

print("\n(1) SHAPE run-up TRƯỚC đỉnh dự báo 'đỉnh thật'? (AUC):", flush=True)
for f in ["v5", "v10", "accel", "updays", "uwick", "rsi", "ext20", "voltr"]:
    au = roc_auc_score(D.top, D[f]); print(f"  {f:8s} AUC={max(au,1-au):.3f} ({'cao=>đỉnh' if au>0.5 else 'thấp=>đỉnh'})", flush=True)

print("\n(2) XÁC NHẬN SAU đỉnh — rớt tới phiên +N tách đỉnh vs hồi (SỚM NHẤT):", flush=True)
for f, lab in [("d1", "+1p"), ("d2", "+2p"), ("d3", "+3p"), ("d5", "+5p"), ("brk_ma5_1", "phá MA5 +1p")]:
    au = roc_auc_score(D.top, -D[f] if f != "brk_ma5_1" else D[f])
    print(f"  rớt {lab:10s} AUC={max(au,1-au):.3f}", flush=True)
# precision của rule "rớt ≥X% trong 2 phiên = đỉnh": bao nhiêu đúng?
print("\n(3) Rule xác nhận sớm 'rớt ≥X% trong 2 phiên sau đỉnh cục bộ' -> P(đỉnh thật) & % lệnh dính:", flush=True)
for X in (0.0, 0.02, 0.03, 0.05):
    g = D[D.d2 <= -X]
    if len(g):
        print(f"  rớt≥{100*X:.0f}%/2p: n={len(g):6d} ({100*len(g)/len(D):3.0f}% đỉnh-cục-bộ) | P(đỉnh thật)={100*g.top.mean():.0f}% (vs base {100*D.top.mean():.0f}%)", flush=True)

print("\n(4) Xác nhận CÓ tốt hơn khi run-up là BLOW-OFF (accel cao)?", flush=True)
D["blow"] = (D.accel > D.accel.median()).map({True: "accel-cao", False: "accel-thấp"})
for grp, gg in D.groupby("blow"):
    au = roc_auc_score(gg.top, -gg.d2)
    print(f"  {grp:10s}: P(đỉnh)={100*gg.top.mean():.0f}% | AUC(rớt-2p) {max(au,1-au):.3f}", flush=True)
print("PEAKSEQ_DONE", flush=True)
