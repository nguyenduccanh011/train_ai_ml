# -*- coding: utf-8 -*-
"""PEAK/GIVE-BACK forensic on BASE model (template 3185). Analyze trades that give back the most (MFE->exit),
characterize the PEAK region + bars BEFORE the peak, and test whether any causal feature separates
'top/coming-down' vs 'still-rising' -> a better sell (retain profit). Multi-angle:
 (A) give-back distribution.
 (B) peak-bar features: high-giveback vs held-gains — what's distinctive AT the peak (blow-off? climax vol?
     reversal wick? extreme RSI/extension? velocity decel?).
 (C) causal TOP-DETECTOR: per held-in-profit bar, does a feature predict forward-10 return < 0 (sell now)?"""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
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

cx = duckdb.connect(DUCK, read_only=True)
px = cx.execute("SELECT symbol,date,open,high,low,close,volume FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])
S = {}  # per-symbol arrays + indicators
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True); c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / (dn + 1e-9))
    ma20 = c.rolling(20).mean(); ma50 = c.rolling(50).mean(); avgv = v.rolling(20).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
    S[s] = dict(c=c.values, h=h.values, l=l.values, o=o.values, v=v.values, rsi=rsi.values,
                ma20=ma20.values, ma50=ma50.values, avgv=avgv.values, atr=atr.values,
                idx={d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])})

con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
con.close()

rows = []
for r in tr.itertuples():
    a = S.get(r.symbol)
    if a is None:
        continue
    ei = a["idx"].get(str(r.entry_date)[:10]); xi = a["idx"].get(str(r.exit_date)[:10])
    if ei is None or xi is None or xi <= ei + 1:
        continue
    c = a["c"]; seg = c[ei:xi + 1]; ep = r.entry_price
    pk = ei + int(np.argmax(seg))                       # peak bar (max close in hold)
    mfe = c[pk] / ep - 1.0; net = r.exit_price / ep - 1.0; gb = mfe - net
    if pk <= ei or pk < 6:
        continue
    # peak-bar features (causal at peak)
    hi, lo, cl, op = a["h"][pk], a["l"][pk], a["c"][pk], a["o"][pk]
    uwick = (hi - cl) / (hi - lo + 1e-9)                 # upper wick (reversal candle)
    dayret = c[pk] / c[pk - 1] - 1.0
    vol_r = a["v"][pk] / (a["avgv"][pk] + 1e-9)
    rsi_pk = a["rsi"][pk]; ext20 = c[pk] / a["ma20"][pk] - 1.0; ext50 = c[pk] / a["ma50"][pk] - 1.0
    vel5 = c[pk] / c[pk - 5] - 1.0                       # velocity into peak
    vel_prev = c[pk - 1] / c[pk - 6] - 1.0               # velocity before peak
    decel = vel5 - vel_prev
    newhi20 = 1.0 if c[pk] >= np.max(c[pk - 19:pk + 1]) else 0.0
    atrn = a["atr"][pk] / c[pk]
    days_to_peak = pk - ei
    rows.append(dict(symbol=r.symbol, mfe=mfe, net=net, gb=gb, days_to_peak=days_to_peak, hold=xi - ei,
                     uwick=uwick, dayret=dayret, vol_r=vol_r, rsi=rsi_pk, ext20=ext20, ext50=ext50,
                     vel5=vel5, decel=decel, newhi20=newhi20, atrn=atrn))
D = pd.DataFrame(rows)
print(f"=== (A) GIVE-BACK distribution (base model, {len(D)} lệnh) ===", flush=True)
print(f"  MFE TB {100*D.mfe.mean():.1f}% | net TB {100*D.net.mean():.1f}% | give-back TB {100*D.gb.mean():.1f}% ({100*D.gb.mean()/max(D.mfe.mean(),1e-9):.0f}% của MFE)", flush=True)
print(f"  give-back phân vị: p50 {100*D.gb.median():.1f}% | p75 {100*D.gb.quantile(.75):.1f}% | p90 {100*D.gb.quantile(.9):.1f}% | max {100*D.gb.max():.1f}%", flush=True)
print(f"  lệnh chạm MFE≥+8% rồi chốt ≤+2% (đỉnh cao trả gần hết): {(( D.mfe>=0.08)&(D.net<=0.02)).sum()} lệnh ({100*((D.mfe>=0.08)&(D.net<=0.02)).mean():.0f}%)", flush=True)

# (B) peak features: high-giveback vs held-gains
D["gbq"] = pd.qcut(D.gb.rank(method="first"), 4, labels=[1, 2, 3, 4])
hi_gb = D[D.gbq == 4]; lo_gb = D[D.gbq == 1]
print(f"\n=== (B) ĐẶC ĐIỂM TẠI ĐỈNH: give-back CAO (Q4, TB {100*hi_gb.gb.mean():.1f}%) vs THẤP (Q1, {100*lo_gb.gb.mean():.1f}%) ===", flush=True)
print(f"  {'feature':14s} | Q4 giveback-cao | Q1 giveback-thấp | tách?", flush=True)
for f in ["rsi", "ext20", "ext50", "uwick", "dayret", "vol_r", "vel5", "decel", "newhi20", "atrn", "days_to_peak"]:
    q4, q1 = hi_gb[f].mean(), lo_gb[f].mean(); sep = abs(q4 - q1) / (D[f].std() + 1e-9)
    print(f"  {f:14s} | {q4:+12.3f} | {q1:+12.3f} | {sep:.2f}σ {'<<<' if sep > 0.3 else ''}", flush=True)

# (C) causal TOP-DETECTOR: per held-in-profit bar, predict forward-10 < 0
print(f"\n=== (C) TOP-DETECTOR causal: mỗi phiên đang-lời, feature dự báo forward-10 < 0 (nên bán) ===", flush=True)
bars = []
for r in tr.itertuples():
    a = S.get(r.symbol)
    if a is None:
        continue
    ei = a["idx"].get(str(r.entry_date)[:10]); xi = a["idx"].get(str(r.exit_date)[:10])
    if ei is None or xi is None:
        continue
    c = a["c"]; ep = r.entry_price
    for b in range(ei + 2, min(xi, len(c) - 10)):
        if c[b] / ep - 1.0 < 0.03:                       # chỉ xét phiên đang lời ≥3% (giữ-lời)
            continue
        if b < 6:
            continue
        fwd10 = c[b + 10] / c[b] - 1.0
        hi, lo, cl = a["h"][b], a["l"][b], a["c"][b]
        bars.append(dict(top=1 if fwd10 < 0 else 0, rsi=a["rsi"][b], ext20=c[b] / a["ma20"][b] - 1.0,
                         ext50=c[b] / a["ma50"][b] - 1.0, uwick=(hi - cl) / (hi - lo + 1e-9),
                         dayret=c[b] / c[b - 1] - 1.0, vol_r=a["v"][b] / (a["avgv"][b] + 1e-9),
                         vel5=c[b] / c[b - 5] - 1.0, decel=(c[b] / c[b - 5] - 1) - (c[b - 1] / c[b - 6] - 1),
                         disthi20=c[b] / np.max(c[b - 19:b + 1]) - 1.0))
B = pd.DataFrame(bars).replace([np.inf, -np.inf], np.nan).dropna()
print(f"  n phiên đang-lời={len(B)} | tỉ lệ top (forward10<0)={100*B.top.mean():.0f}%", flush=True)
for f in ["rsi", "ext20", "ext50", "uwick", "dayret", "vol_r", "vel5", "decel", "disthi20"]:
    au = roc_auc_score(B.top.values, B[f].values); au = max(au, 1 - au)
    dirn = "cao=>top" if roc_auc_score(B.top.values, B[f].values) > 0.5 else "thấp=>top"
    print(f"  {f:9s} AUC={au:.3f} ({dirn})", flush=True)
# combo top-signal: extreme extension + reversal wick + climax vol
B["topsig"] = ((B.rsi > 72).astype(int) + (B.uwick > 0.5).astype(int) + (B.vol_r > 1.5).astype(int) + (B.ext20 > 0.10).astype(int))
print("  combo topsig (rsi>72 + uwick>.5 + vol>1.5x + ext20>10%), theo số điều kiện thỏa:", flush=True)
for k in range(5):
    g = B[B.topsig == k]
    if len(g):
        print(f"    {k}/4 đk: n={len(g):6d} | P(top forward10<0)={100*g.top.mean():.0f}%", flush=True)
print("PEAK_FORENSIC_DONE", flush=True)
