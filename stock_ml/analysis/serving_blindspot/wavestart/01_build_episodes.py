# -*- coding: utf-8 -*-
"""Build missed_moves.csv: >=15%-in-21-bar upswing episodes on the top150 universe.

Episode = start bar t that is a trailing 21-bar low of `low`, refined to the argmin
of low over [t, t+3], with max(close[t+1..t+21])/close[t] - 1 >= 0.15.
Non-overlapping (greedy; next scan resumes after the peak bar).
captured = any champion trade (trades_raw.csv) whose [entry_date, exit_date]
overlaps [start_date, end_date] for the symbol.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = os.path.join(BASE, "wavestart")
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"

GAIN_MIN = 0.15
FWD = 21
LOOKBACK = 21  # trailing low window (bars incl. current)

sig = pd.read_csv(os.path.join(BASE, "signals.csv"), usecols=["symbol"])
universe = sorted(sig["symbol"].unique())
assert len(universe) == 150, len(universe)

con = sqlite3.connect(DB)
q = "select symbol, date, open, high, low, close, volume from ohlcv where symbol in (%s) order by symbol, date" % (
    ",".join("?" * len(universe)))
oh = pd.read_sql(q, con, params=universe)
con.close()

trades = pd.read_csv(os.path.join(BASE, "trades_raw.csv"),
                     usecols=["symbol", "entry_date", "exit_date"])
trades["exit_date"] = trades["exit_date"].fillna("2099-12-31")
tr_by_sym = {s: g[["entry_date", "exit_date"]].to_numpy() for s, g in trades.groupby("symbol")}

episodes = []
for sym, g in oh.groupby("symbol"):
    g = g.reset_index(drop=True)
    dates = g["date"].to_numpy()
    lo = g["low"].to_numpy(float)
    cl = g["close"].to_numpy(float)
    n = len(g)
    # trailing 21-bar low of low
    lo_ser = pd.Series(lo)
    trail_min = lo_ser.rolling(LOOKBACK, min_periods=LOOKBACK).min().to_numpy()
    t = LOOKBACK
    first_2020 = np.searchsorted(dates, "2020-01-01")
    t = max(t, first_2020)
    while t < n - FWD:
        if lo[t] == trail_min[t]:
            # refine start to argmin of low over [t, t+3]
            s = t + int(np.argmin(lo[t:min(t + 4, n)]))
            if s < n - FWD:
                fwd_max = cl[s + 1:s + 1 + FWD].max()
                gain = fwd_max / cl[s] - 1.0
                if gain >= GAIN_MIN:
                    peak = s + 1 + int(np.argmax(cl[s + 1:s + 1 + FWD]))
                    sd, ed = dates[s], dates[peak]
                    cap = False
                    for e, x in tr_by_sym.get(sym, []):
                        if e <= ed and x >= sd:
                            cap = True
                            break
                    episodes.append(dict(symbol=sym, start_date=sd, end_date=ed,
                                         start_idx=s, peak_idx=peak,
                                         start_close=cl[s], peak_close=fwd_max,
                                         gain_pct=round(gain * 100, 2), captured=cap))
                    t = peak + 1
                    continue
        t += 1

ep = pd.DataFrame(episodes)
ep.to_csv(os.path.join(OUT, "missed_moves.csv"), index=False)
big = ep[ep.gain_pct >= 30]
print("episodes:", len(ep), "| symbols:", ep.symbol.nunique())
print("captured rate all: %.1f%%" % (100 * ep.captured.mean()))
print(">=30%%: n=%d missed=%.1f%%" % (len(big), 100 * (1 - big.captured.mean())))
print("15-30%%: n=%d missed=%.1f%%" % ((ep.gain_pct < 30).sum(),
                                       100 * (1 - ep[ep.gain_pct < 30].captured.mean())))
print(ep.groupby(ep.start_date.str[:4]).size())
