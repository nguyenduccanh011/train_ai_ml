# -*- coding: utf-8 -*-
"""Counterfactual: neu trailing DEFAULT tier la %-trail (ATR-scaled, khong struct-donch80)
thi moi lenh gb_x08 exit khi nao? delta = pnl_actual - pnl_cf (<0: struct-donch lam te hon).
Xap xi co chu dich: bo qua overext-arm 4% band (bao split theo exit_reason de doc caveat),
khong modulator nao khac active trong 2783 (score_k/cons/tier2/pop_lock/mfe_act = None).
armed = peak_gain>=0.27 AND NOT trend_up(MA10 minp1, slope3). trail = clip(2*ATR14/close,0.04,0.16).
Fire khi low[i]/peak - 1 <= -trail; fill close[i+1] (close_next).
"""
import json
import sys

import duckdb
import numpy as np
import pandas as pd
import psycopg2

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
SLIP, RT_COST = 0.0015, 2 * 0.0015 + 0.001
MIN_HOLD, ACT = 2, 0.27

tr = pd.read_csv(EM + "/gbx08_s42_trades.csv",
                 parse_dates=["entry_date", "exit_date", "entry_signal_date"])
pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                      user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, high, low, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in uni_syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])

SYM = {}
for sym, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    c = g["close"].to_numpy(float)
    h = g["high"].to_numpy(float)
    lo = g["low"].to_numpy(float)
    cs = pd.Series(c)
    sma10 = cs.rolling(10, min_periods=1).mean()
    trend_up = (c > sma10.to_numpy()) & ((sma10 - sma10.shift(3)).to_numpy() > 0)
    pc = np.concatenate([[c[0]], c[:-1]])
    trr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    atr14r = pd.Series(trr).rolling(14, min_periods=1).mean().to_numpy() / np.where(c == 0, 1e-9, c)
    SYM[sym] = dict(dates=pd.DatetimeIndex(g["date"]), c=c, h=h, lo=lo,
                    trend_up=trend_up, trail=np.clip(2.0 * atr14r, 0.04, 0.16))

out = []
for _, t in tr.iterrows():
    if t.exit_reason == "open" or t.symbol not in SYM:
        continue
    s = SYM[t.symbol]
    di = s["dates"]
    ei = di.get_indexer([t.entry_date])[0]
    xi = di.get_indexer([t.exit_date])[0]
    if ei < 0 or xi < 0:
        continue
    ep = float(t.entry_price)
    peak = float(s["h"][ei])
    fire = None
    for j in range(ei, xi):  # truoc bar fill thuc te
        peak = max(peak, float(s["h"][j]))
        if j - ei < MIN_HOLD:
            continue
        pg_ = peak / ep - 1.0
        if pg_ < ACT or s["trend_up"][j]:
            continue
        if s["lo"][j] / peak - 1.0 <= -s["trail"][j]:
            fire = j
            break
    if fire is None or fire >= xi - 1:  # khong som hon decision bar thuc te
        continue
    cf_pnl = s["c"][fire + 1] * (1 - SLIP) / ep - 1 - RT_COST
    out.append(dict(symbol=t.symbol, entry_date=t.entry_date.date(), exit_date=t.exit_date.date(),
                    exit_reason=t.exit_reason, pnl=float(t.pnl_pct), cf_date=di[fire].date(),
                    cf_pnl=cf_pnl, delta=float(t.pnl_pct) - cf_pnl,
                    bars_early=int(xi - 1 - fire), year_entry=t.entry_date.year))
C = pd.DataFrame(out)
C.to_csv(EM + "/pdr_structtrail_cf.csv", index=False)
print("trades ma %-trail thuan bat SOM hon exit thuc te:", len(C), "/", len(tr))
for lab, seg in [("ALL", C), (">=2022", C[C.year_entry >= 2022])]:
    neg = seg.delta[seg.delta < 0].sum()
    pos = seg.delta[seg.delta > 0].sum()
    print(f"\n--- {lab}: n={len(seg)} net={seg.delta.sum():+.2f} "
          f"(struct giup {pos:+.2f} / lam hai {neg:+.2f}) "
          f"worse5={int((seg.delta <= -0.05).sum())} helped5={int((seg.delta >= 0.05).sum())}")
    print(seg.groupby("exit_reason").agg(n=("delta", "size"), net=("delta", "sum"),
                                         mean=("delta", "mean")).round(3).to_string())
g = C[C.year_entry >= 2022]
print("\nnet theo nam (>=2022):")
print(g.groupby("year_entry").agg(n=("delta", "size"), net=("delta", "sum")).round(2).to_string())
print("\nPDR:")
print(C[(C.symbol == "PDR") & (C.exit_date.astype(str) == "2024-04-24")].round(3).to_string(index=False))
print("\nworst 10 >=2022:")
print(g.nsmallest(10, "delta").round(3).to_string(index=False))
