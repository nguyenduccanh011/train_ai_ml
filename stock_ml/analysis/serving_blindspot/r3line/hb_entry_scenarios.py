# -*- coding: utf-8 -*-
"""Entry-timing scenarios per pullback order: buy-at-market NOW vs wait-for-pullback vs hybrid vs average.
Distinct orders (symbol, signal_date S, outcome, limit, result_date). Hold H bars FROM ENTRY.
  MARKET     : entry = close[S] on every signal (no wait).
  PULLBACK   : entry = limit at touch (fills only); expired signals NOT traded (cash).
  HYBRID     : entry = limit at touch if fills; else market at window-end close (chase the runaway late).
  AVERAGE    : 50% at market(close[S]) + 50% at limit(touch) if it fills, else the 50% stays market-only.
Reports avg return, deployed %, and per-signal expectation (0 return on non-deployed)."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
RID = "template/x2_struct_to_k16preempt_cssize-69338138"
H = 40; PB_WIN = 40

con = psycopg2.connect(**PG)
o = pd.read_sql("SELECT DISTINCT symbol, signal_date, outcome, limit_price, result_date "
                "FROM run_pending WHERE run_id=%s", con, params=(RID,)); con.close()
o["signal_date"] = pd.to_datetime(o["signal_date"]); o["result_date"] = pd.to_datetime(o["result_date"])

cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])
CA, BO = {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True); CA[s] = g["close"].to_numpy(); BO[s] = {d: i for i, d in enumerate(g["date"])}


def px_at(sym, date):
    b = BO.get(sym, {}).get(pd.Timestamp(date)); ca = CA.get(sym)
    return (b, float(ca[b])) if (b is not None and ca is not None) else (None, None)


def ret_from(sym, entry_bar, entry_px):
    ca = CA.get(sym)
    if entry_bar is None or entry_px is None or ca is None or entry_bar + H >= len(ca) or entry_px <= 0:
        return np.nan
    return ca[entry_bar + H] / entry_px - 1.0


mkt, pb, hyb, avg = [], [], [], []
for r in o.itertuples():
    bS, cS = px_at(r.symbol, r.signal_date)
    if bS is None:
        continue
    r_mkt = ret_from(r.symbol, bS, cS)
    mkt.append(r_mkt)
    if r.outcome == "fill" and pd.notna(r.result_date):
        bT, _ = px_at(r.symbol, r.result_date)
        r_fill = ret_from(r.symbol, bT, r.limit_price)
        pb.append(r_fill); hyb.append(r_fill)
        avg.append(np.nanmean([r_mkt, r_fill]))
    else:  # expired
        pb.append(np.nan)                                   # pullback: not traded
        be = min(bS + PB_WIN, len(CA[r.symbol]) - 1); ce = CA[r.symbol][be]
        hyb.append(ret_from(r.symbol, be, ce))              # hybrid: chase at window-end
        avg.append(r_mkt)                                    # average: only the market half got in


def summ(name, arr):
    a = np.array(arr, dtype=float); dep = np.isfinite(a).mean(); m = np.nanmean(a)
    perSig = np.nanmean(np.where(np.isfinite(a), a, 0.0))    # 0 on non-deployed
    win = np.nanmean(a[np.isfinite(a)] > 0) if np.isfinite(a).any() else np.nan
    print(f"{name:10s}| deployed {dep*100:5.1f}% | avg/deployed {m*100:+6.2f}% | per-signal {perSig*100:+6.2f}% | win {win*100:4.0f}%")


n = len(mkt)
print(f"orders={n}  (fill/expire ratio = {(o.outcome=='fill').sum()}:{(o.outcome=='expire').sum()} "
      f"= {(o.outcome=='fill').sum()/(o.outcome=='expire').sum():.2f}:1)  |  hold H={H} bars\n")
print("scenario  | deployed | avg/deployed | per-signal(exp) | win")
summ("MARKET", mkt)
summ("PULLBACK", pb)
summ("HYBRID", hyb)
summ("AVERAGE", avg)
print("\n(per-signal = expectation counting 0 on signals not traded; the fair 'which entry rule wins')")
print("ENTRY_SCEN_DONE")
