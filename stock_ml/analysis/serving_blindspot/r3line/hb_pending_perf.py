# -*- coding: utf-8 -*-
"""Performance of FILLED vs EXPIRED pullback orders — does waiting for the -4.5% dip add value?
For each distinct resting order (symbol, signal_date S, outcome), measure forward returns:
  - RAW stock move from the signal close (same basis for both cohorts) at +10/+20/+40 bars.
  - FILLED: return from the LIMIT price (the dip you actually bought) at +H bars after the touch.
  - EXPIRED: opportunity cost = return from the SIGNAL close (market, since it ran away) at +H bars.
If EXPIRED raw-move >> FILLED -> the runaways you skip are the winners (pullback crutch cost). If FILLED
from-limit >> EXPIRED -> the dip-buy entry advantage pays off."""
from __future__ import annotations
import os, sys, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
RID = "template/x2_struct_to_k16preempt_cssize-69338138"
HZ = [10, 20, 40]

con = psycopg2.connect(**PG)
# distinct resting orders (anchored at signal_date) with their outcome + limit
orders = pd.read_sql(
    "SELECT DISTINCT symbol, signal_date, outcome, limit_price, result_date "
    "FROM run_pending WHERE run_id=%s", con, params=(RID,))
con.close()
orders["signal_date"] = pd.to_datetime(orders["signal_date"])
orders["result_date"] = pd.to_datetime(orders["result_date"])
print(f"distinct pullback orders: {len(orders)}  (fill={ (orders.outcome=='fill').sum() }  expire={ (orders.outcome=='expire').sum() })", flush=True)

# per-symbol close series + bar index
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])
close_arr, bar_of = {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    close_arr[s] = g["close"].to_numpy()
    bar_of[s] = {d: i for i, d in enumerate(g["date"])}


def fwd(sym, from_date, from_price, H):
    """return over H trading bars after from_date, measured off from_price."""
    b = bar_of.get(sym, {}).get(pd.Timestamp(from_date))
    ca = close_arr.get(sym)
    if b is None or ca is None or b + H >= len(ca) or not from_price or from_price <= 0:
        return np.nan
    return ca[b + H] / from_price - 1.0


def sig_close(sym, sd):
    b = bar_of.get(sym, {}).get(pd.Timestamp(sd)); ca = close_arr.get(sym)
    return float(ca[b]) if (b is not None and ca is not None) else np.nan


rows = []
for o in orders.itertuples():
    sc = sig_close(o.symbol, o.signal_date)
    rec = {"outcome": o.outcome}
    for H in HZ:
        rec[f"raw{H}"] = fwd(o.symbol, o.signal_date, sc, H)            # stock move from signal (same basis)
        if o.outcome == "fill" and pd.notna(o.result_date):
            rec[f"real{H}"] = fwd(o.symbol, o.result_date, o.limit_price, H)  # bought at dip on touch date
        else:
            rec[f"real{H}"] = fwd(o.symbol, o.signal_date, sc, H)        # expired -> market at signal (opp. cost)
    rows.append(rec)
df = pd.DataFrame(rows)

print("\n=== RAW stock move from SIGNAL close (same basis both cohorts) — avg % ===")
print("cohort   |  n   | +10bar | +20bar | +40bar")
for oc in ["fill", "expire"]:
    g = df[df.outcome == oc]
    print(f"{oc:8s} | {len(g):4d} | " + " | ".join(f"{g[f'raw{H}'].mean()*100:+6.2f}" for H in HZ))

print("\n=== 'REAL' return: FILL from dip-price(limit) vs EXPIRE opp-cost from market(signal) — avg % ===")
print("cohort   |  n   | +10bar | +20bar | +40bar")
for oc in ["fill", "expire"]:
    g = df[df.outcome == oc]
    print(f"{oc:8s} | {len(g):4d} | " + " | ".join(f"{g[f'real{H}'].mean()*100:+6.2f}" for H in HZ))

print("\n=== win-rate (real return > 0) ===")
for oc in ["fill", "expire"]:
    g = df[df.outcome == oc]
    print(f"{oc:8s}: " + " | ".join(f"+{H}b {100*(g[f'real{H}']>0).mean():.0f}%" for H in HZ))
print("PENDING_PERF_DONE")
