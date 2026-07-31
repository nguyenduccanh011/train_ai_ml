# -*- coding: utf-8 -*-
"""ROOT-CAUSE leak forensic on the champion's actual trades. Where does alpha escape?
  MFE-capture   : realized pnl / peak (max-favorable) — 1-capture = giveback/retention leak.
  post-exit cont: price N bars after exit vs exit price — money left by exiting early.
  distribution  : P&L tail (winners vs losers), by year.
Quantifies which leak is biggest -> the breakthrough target. Uses champion run_trades (combo, seed 42)."""
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
tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,pnl_pct,exit_reason "
                 "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND exit_reason<>'preempt'", con, params=(RID,))
con.close()
tr["entry_date"] = pd.to_datetime(tr["entry_date"]); tr["exit_date"] = pd.to_datetime(tr["exit_date"])
tr["yr"] = tr["entry_date"].dt.year

cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,high,close FROM ohlcv WHERE timeframe='1D' AND date>='2019-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])
HI, CL, BO = {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    HI[s] = g["high"].to_numpy(); CL[s] = g["close"].to_numpy(); BO[s] = {d: i for i, d in enumerate(g["date"])}

mfe, capt, cont10, cont20 = [], [], [], []
for r in tr.itertuples():
    be = BO.get(r.symbol, {}).get(r.entry_date); bx = BO.get(r.symbol, {}).get(r.exit_date)
    if be is None or bx is None or r.entry_price <= 0:
        mfe.append(np.nan); capt.append(np.nan); cont10.append(np.nan); cont20.append(np.nan); continue
    peak = HI[r.symbol][be:bx + 1].max(); mret = peak / r.entry_price - 1.0
    mfe.append(mret)
    capt.append(r.pnl_pct / mret if mret > 1e-6 else np.nan)
    ca = CL[r.symbol]
    cont10.append(ca[bx + 10] / r.exit_price - 1.0 if bx + 10 < len(ca) and r.exit_price > 0 else np.nan)
    cont20.append(ca[bx + 20] / r.exit_price - 1.0 if bx + 20 < len(ca) and r.exit_price > 0 else np.nan)
tr["mfe"] = mfe; tr["capt"] = capt; tr["cont10"] = cont10; tr["cont20"] = cont20

print(f"champion trades (excl preempt): {len(tr)}  win%={100*(tr.pnl_pct>0).mean():.1f}  avg_pnl={100*tr.pnl_pct.mean():+.2f}%\n")
print("=== LEAK 1: MFE-capture (realized / peak) — 1-capt = giveback ===")
print(f"  avg MFE (peak gain)    = {100*tr.mfe.mean():+.2f}%")
print(f"  avg realized pnl       = {100*tr.pnl_pct.mean():+.2f}%")
print(f"  median capture ratio   = {tr.capt.median():.2f}  (1.0 = sold at peak; low = big giveback)")
print(f"  avg giveback (peak-real)= {100*(tr.mfe-tr.pnl_pct).mean():+.2f}%  (per trade left on table)")
print("\n=== LEAK 2: post-exit continuation (money after exit) ===")
print(f"  avg +10bar after exit  = {100*tr.cont10.mean():+.2f}%   (>0 = exited too early)")
print(f"  avg +20bar after exit  = {100*tr.cont20.mean():+.2f}%")
print("\n=== P&L distribution ===")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    print(f"  p{int(q*100):02d} pnl = {100*tr.pnl_pct.quantile(q):+6.1f}%", end="  ")
print(f"\n  top-10% trades contribute {100*tr.nlargest(int(len(tr)*0.1),'pnl_pct').pnl_pct.sum()/tr.pnl_pct.sum():.0f}% of total pnl")
print("\n=== BY YEAR: win% | avg_pnl | MFE | capture | post-exit+20 ===")
for y, g in tr.groupby("yr"):
    print(f"  {y}: {100*(g.pnl_pct>0).mean():4.0f}% | {100*g.pnl_pct.mean():+5.1f}% | {100*g.mfe.mean():+5.1f}% | {g.capt.median():.2f} | {100*g.cont20.mean():+5.1f}%")
print("\n=== by exit_reason: n | avg_pnl | capture | post-exit+20 ===")
for rs, g in tr.groupby("exit_reason"):
    if len(g) < 20:
        continue
    print(f"  {rs:14s}: {len(g):4d} | {100*g.pnl_pct.mean():+5.1f}% | {g.capt.median():.2f} | {100*g.cont20.mean():+5.1f}%")
print("LEAK_FORENSIC_DONE")
