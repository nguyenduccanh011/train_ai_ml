# -*- coding: utf-8 -*-
"""Signal-starvation step 1: ti le bat song cua model theo nam (2020+).

- has_signal: co >=1 buy-bar (signals.csv, frame 2643) trong [foot-10, foot+10] (bar).
- has_trade : co >=1 trade gb_x08 (run_id template/gb_x08-32a8dfee) voi entry_signal_date
              trong [foot-10, peak] (tham gia song); moi trade gan cho 1 song dau tien khop.
Output: ss_catch_w1.csv / ss_catch_w2.csv (wave-level, them cot has_signal/has_trade/u_sum)
        + bang theo nam.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = os.path.join(BASE, "exitmap")
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
PRE, POST = 10, 10

sig = pd.read_csv(os.path.join(BASE, "signals.csv"), usecols=["symbol", "date", "signal"])
buys = sig[sig.signal > 0]
buy_by_sym = {s: set(g.date) for s, g in buys.groupby("symbol")}

tr = pd.read_csv(os.path.join(OUT, "gbx08_trades.csv"))
tr["entry_signal_date"] = tr["entry_signal_date"].astype(str).str[:10]
tr_by_sym = {s: g[["entry_signal_date", "pnl_pct"]].reset_index(drop=True)
             for s, g in tr.groupby("symbol")}

universe = sorted(buy_by_sym.keys() | set(pd.read_csv(
    os.path.join(BASE, "signals.csv"), usecols=["symbol"]).symbol.unique()))
con = sqlite3.connect(DB)
q = "select symbol, date from ohlcv where symbol in (%s) and date >= '2018-01-01' order by symbol, date" % (
    ",".join("?" * len(universe)))
cal = pd.read_sql(q, con, params=universe)
con.close()
dates_by_sym = {s: g["date"].to_numpy() for s, g in cal.groupby("symbol")}

for wname in ["w1", "w2"]:
    ep = pd.read_csv(os.path.join(OUT, f"ss_waves_{wname}.csv"))
    ep = ep[(ep.foot_date >= "2020-01-01") & (ep.foot_date < "2026-07-01")].reset_index(drop=True)
    has_sig = np.zeros(len(ep), bool)
    has_trd = np.zeros(len(ep), bool)
    u_sum = np.zeros(len(ep))
    n_trd = np.zeros(len(ep), int)
    used_trades = {s: np.zeros(len(g), bool) for s, g in tr_by_sym.items()}
    for i, r in enumerate(ep.itertuples()):
        d = dates_by_sym[r.symbol]
        lo_i = max(0, r.foot_idx - PRE)
        hi_i = min(len(d) - 1, r.foot_idx + POST)
        win = set(d[lo_i:hi_i + 1])
        bs = buy_by_sym.get(r.symbol)
        if bs and (win & bs):
            has_sig[i] = True
        tt = tr_by_sym.get(r.symbol)
        if tt is not None:
            d_lo, d_hi = d[lo_i], d[min(r.peak_idx, len(d) - 1)]
            m = ((tt.entry_signal_date >= d_lo) & (tt.entry_signal_date <= d_hi)
                 & ~used_trades[r.symbol])
            if m.any():
                has_trd[i] = True
                u_sum[i] = tt.pnl_pct[m].sum()
                n_trd[i] = int(m.sum())
                used_trades[r.symbol][m.to_numpy()] = True
    ep["has_signal"] = has_sig
    ep["has_trade"] = has_trd
    ep["u_sum"] = u_sum
    ep["n_trades"] = n_trd
    ep.to_csv(os.path.join(OUT, f"ss_catch_{wname}.csv"), index=False)

    ep["year"] = ep.foot_date.str[:4]
    g = ep.groupby("year").agg(
        n=("symbol", "size"),
        sig_pct=("has_signal", lambda x: 100 * x.mean()),
        trd_pct=("has_trade", lambda x: 100 * x.mean()),
        gain_mean=("gain_pct", "mean"),
        u_caught=("u_sum", "sum"),
    )
    # u tren song bat duoc
    cap = ep[ep.has_trade]
    g["u_per_caught"] = cap.groupby(cap.foot_date.str[:4]).u_sum.mean()
    print(f"\n=== {wname.upper()} catch (frame-2643 signal ±{PRE}bar; trade gb_x08 foot-10..peak) ===")
    print(g.round(2).to_string())
    # blind waves cho autopsy
    blind = ep[~ep.has_signal & (ep.year >= "2024")]
    print(f"blind (khong signal ±10bar) 2024-26H1: {len(blind)} / {len(ep[ep.year>='2024'])}")
print("\nDONE ss_01")
