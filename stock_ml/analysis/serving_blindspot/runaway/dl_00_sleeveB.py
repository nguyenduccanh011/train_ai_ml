# -*- coding: utf-8 -*-
"""DUAL-LAYER prep — sleeve B = runaway cohort at-market, exit trail10.

Tai tao dung logic rw_02_schemes_sep.py (scheme trail10, sequential per-symbol
occupancy) nhung luu day du entry_date/exit_date/entry_fill de portfolio sim dung.
Verify: taken == 1009, pnl sum ~ +205.8u (autopsy).
"""
import sqlite3

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = BASE + r"\runaway"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
SLIP_IN, SLIP_OUT, FEE = 1.0015, 0.9985, 0.004

def net(e, x):
    return (x * SLIP_OUT) / (e * SLIP_IN) - 1.0 - FEE

uf = pd.read_csv(f"{BASE}/unfilled_signals.csv")
con = sqlite3.connect(DB)
px = pd.read_sql("select symbol,date,close from ohlcv order by symbol,date", con)
con.close()

A = {}
for s, g in px.groupby("symbol"):
    g = g.reset_index(drop=True)
    A[s] = dict(dates=g.date.to_numpy(), idx={d: i for i, d in enumerate(g.date)},
                c=g.close.to_numpy(float))

u = uf[uf.drop_reason == "unfilled"].copy()
co = u[u.window_end_close > u.signal_close].sort_values(["symbol", "signal_date"])
print(f"runaway signals: {len(co)}")

rows = []
open_until = {}
for r in co.itertuples():
    a = A[r.symbol]
    i = a["idx"][r.signal_date]
    n = len(a["c"])
    if i + 1 >= n:
        continue
    if open_until.get(r.symbol, -1) >= i:
        continue
    ei = i + 1
    ec = a["c"][ei]
    # trail10: peak-close -10%, exit close bar t+1 sau trigger
    peak = ec
    xi, xreason = n - 1, "end_of_data"
    for t in range(ei + 1, n):
        peak = max(peak, a["c"][t])
        if a["c"][t] < peak * 0.90:
            xi, xreason = min(t + 1, n - 1), "trail"
            break
    xc = a["c"][xi]
    open_until[r.symbol] = xi
    rows.append(dict(symbol=r.symbol, signal_date=r.signal_date,
                     entry_date=a["dates"][ei], exit_date=a["dates"][xi],
                     entry_close=ec, exit_close=xc,
                     entry_fill=ec * SLIP_IN, hold=xi - ei,
                     pnl_net=net(ec, xc), exit_reason=xreason))

d = pd.DataFrame(rows)
print(f"taken: {len(d)}  pnl_u: {d.pnl_net.sum():+.1f}  hold_med: {d.hold.median():.0f}  "
      f"WR: {(d.pnl_net > 0).mean():.2f}")
print("per-year n:", d.groupby(d.signal_date.str[:4]).size().to_dict())
print("exit reasons:", d.exit_reason.value_counts().to_dict())

# concurrency (so vi the mo dong thoi) tren lich union
cal = sorted(set(px[px.symbol.isin(d.symbol.unique())].date))
cal = [c for c in cal if d.entry_date.min() <= c <= d.exit_date.max()]
ev = {}
for r in d.itertuples():
    ev.setdefault(r.entry_date, [0, 0])[0] += 1
    ev.setdefault(r.exit_date, [0, 0])[1] += 1
cur, conc = 0, []
for c in cal:
    e = ev.get(c, (0, 0))
    cur += e[0] - e[1]
    conc.append(cur)
conc = pd.Series(conc)
print("concurrency: mean %.1f med %.0f p75 %.0f p90 %.0f max %.0f" %
      (conc.mean(), conc.median(), conc.quantile(.75), conc.quantile(.90), conc.max()))

d.to_csv(f"{OUT}/dl_sleeveB_trail10.csv", index=False)
print(f"saved {OUT}/dl_sleeveB_trail10.csv")
