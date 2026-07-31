# -*- coding: utf-8 -*-
"""Detailed case study of 2 extremes on BASE model (3185): (1) big WINNERS that gave back the most
(high MFE -> low net); (2) bought then DROPPED hard (deep early MAE). Characterize each: run size, peak
timing, exit reason, year; and for case (2) how much the registered levers (early-cut red@s2, overshoot
falling-knife) would have caught."""
from __future__ import annotations
import os, sys, warnings
from collections import Counter
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np, duckdb
from scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"

cx = duckdb.connect(DUCK, read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); S = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    S[s] = dict(c=g["close"].values, h=g["high"].values, l=g["low"].values,
                idx={d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])})

con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date,exit_reason from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
con.close()

rows = []
for r in tr.itertuples():
    a = S.get(r.symbol)
    if a is None:
        continue
    ei = a["idx"].get(str(r.entry_date)[:10]); xi = a["idx"].get(str(r.exit_date)[:10]); si_ = a["idx"].get(str(r.sigd.date()) if hasattr(r, "sigd") else str(pd.to_datetime(r.entry_signal_date).date()))
    if ei is None or xi is None or xi <= ei:
        continue
    c = a["c"]; ep = r.entry_price; seg = c[ei:xi + 1]
    pk = int(np.argmax(seg)); tr_b = int(np.argmin(a["l"][ei:xi + 1]))
    mfe = c[ei + pk] / ep - 1.0; mae = a["l"][ei:xi + 1].min() / ep - 1.0; net = r.exit_price / ep - 1.0
    e2 = c[ei + 2] / ep - 1.0 if ei + 2 <= xi else net
    over = (ep - a["l"][si_:ei + 1].min()) / c[si_] if (si_ is not None and si_ <= ei) else 0.0
    rows.append(dict(symbol=r.symbol, ed=str(r.entry_date)[:10], xd=str(r.exit_date)[:10], yr=int(str(r.entry_date)[:4]),
                     mfe=mfe, mae=mae, net=net, gb=mfe - net, d2peak=pk, d2trough=tr_b, hold=xi - ei,
                     e2=e2, overshoot=over, reason=r.exit_reason))
D = pd.DataFrame(rows)

# ===== CASE 1: winners that gave back the most (ran >=15%, sort by give-back) =====
C1 = D[D.mfe >= 0.15].sort_values("gb", ascending=False)
print(f"=== CASE 1: lệnh chạy LỚN (MFE≥15%) rồi TRẢ LẠI nhiều nhất — {len(C1)} lệnh ===", flush=True)
print(f"  {'mã':5s} {'vào':10s} {'MFE':>6s} {'chốt':>6s} {'trả lại':>7s} {'đỉnh(p)':>7s} {'giữ(p)':>6s}  lý do thoát", flush=True)
for r in C1.head(12).itertuples():
    print(f"  {r.symbol:5s} {r.ed:10s} {100*r.mfe:5.0f}% {100*r.net:5.0f}% {100*r.gb:6.0f}% {r.d2peak:6d} {r.hold:5d}  {r.reason}", flush=True)
print(f"  [gộp] n={len(C1)} | MFE TB {100*C1.mfe.mean():.0f}% chốt {100*C1.net.mean():.0f}% trả {100*C1.gb.mean():.0f}% | đạt đỉnh TB phiên {C1.d2peak.mean():.0f}/giữ {C1.hold.mean():.0f}", flush=True)
print(f"  năm: {dict(sorted(Counter(C1.yr).items()))}", flush=True)
print(f"  lý do thoát: {dict(Counter(C1.reason).most_common())}", flush=True)
print(f"  đạt đỉnh SỚM (≤5 phiên) rồi fade: {100*(C1.d2peak<=5).mean():.0f}% | vẫn dương khi chốt: {100*(C1.net>0).mean():.0f}%", flush=True)

# ===== CASE 2: bought then dropped hard (deep MAE) =====
C2 = D[D.mae <= -0.10].sort_values("net")
print(f"\n=== CASE 2: mua xong GIẢM MẠNH (đáy ≤ −10% so giá vào) — {len(C2)} lệnh ===", flush=True)
print(f"  {'mã':5s} {'vào':10s} {'đáy(MAE)':>8s} {'chốt':>6s} {'e2':>5s} {'overshoot':>9s} {'đáy(p)':>6s}  lý do", flush=True)
for r in C2.head(12).itertuples():
    print(f"  {r.symbol:5s} {r.ed:10s} {100*r.mae:7.0f}% {100*r.net:5.0f}% {100*r.e2:4.0f}% {100*r.overshoot:8.1f}% {r.d2trough:5d}  {r.reason}", flush=True)
print(f"  [gộp] n={len(C2)} ({100*len(C2)/len(D):.0f}% tổng) | MAE TB {100*C2.mae.mean():.0f}% chốt {100*C2.net.mean():.0f}% | đáy TB phiên {C2.d2trough.mean():.0f}", flush=True)
print(f"  năm: {dict(sorted(Counter(C2.yr).items()))}", flush=True)
print(f"  ĐỎ ngay phiên 2 (early-cut bắt): {100*(C2.e2<0).mean():.0f}% | overshoot cao top-decile: {100*(C2.overshoot>D.overshoot.quantile(.9)).mean():.0f}%", flush=True)
print(f"  chốt ÂM: {100*(C2.net<0).mean():.0f}% | nhưng HỒI về dương: {100*(C2.net>0).mean():.0f}%", flush=True)
# so sánh: nhóm giảm-mạnh-rồi-hồi (net>0) vs giảm-mạnh-rồi-lỗ (net<0) — e2 có tách?
rec = C2[C2.net > 0]; ko = C2[C2.net <= 0]
print(f"  [tách] giảm-mạnh-rồi-HỒI (n={len(rec)}): e2 TB {100*rec.e2.mean():+.1f}% | giảm-mạnh-rồi-LỖ (n={len(ko)}): e2 TB {100*ko.e2.mean():+.1f}%", flush=True)
print("EXTREME_DONE", flush=True)
