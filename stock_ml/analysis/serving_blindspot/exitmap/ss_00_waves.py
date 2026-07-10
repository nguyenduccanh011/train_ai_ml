# -*- coding: utf-8 -*-
"""Signal-starvation step 0: kiem ke song co hoc doc lap voi model (universe top150).

Hai dinh nghia song (ghi ro trong SIGNAL_STARVATION_2024.md):
  W1 "dip-rally": chan song = trailing-21-bar low cua low (refine argmin [t,t+3]),
     max(close[t+1..t+40])/close[foot] - 1 >= 15%. Non-overlap greedy (resume sau peak).
     — cung ho dinh nghia voi wavestart/01_build_episodes.py nhung horizon 40 bar
       (khop pullback window 40 cua champion).
  W2 "reclaim-MA20": close cat len MA20 sau >=5 bar lien tuc duoi MA20,
     max(close[t+1..t+40])/close[t] - 1 >= 10%. Non-overlap greedy.
Dem 2019 -> 2026H1. Universe = 150 ma cua frame serving 2643 (signals.csv).
"""
import os
import sqlite3

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = os.path.join(BASE, "exitmap")
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"

FWD = 40
W1_GAIN = 0.15
W1_LOOKBACK = 21
W2_GAIN = 0.10
W2_BELOW = 5

sig = pd.read_csv(os.path.join(BASE, "signals.csv"), usecols=["symbol"])
universe = sorted(sig["symbol"].unique())
assert len(universe) == 150, len(universe)

con = sqlite3.connect(DB)
q = "select symbol, date, open, high, low, close, volume from ohlcv where symbol in (%s) and date >= '2018-01-01' order by symbol, date" % (
    ",".join("?" * len(universe)))
oh = pd.read_sql(q, con, params=universe)
con.close()

w1_rows, w2_rows = [], []
for sym, g in oh.groupby("symbol"):
    g = g.reset_index(drop=True)
    dates = g["date"].to_numpy()
    lo = g["low"].to_numpy(float)
    cl = g["close"].to_numpy(float)
    vol = g["volume"].to_numpy(float)
    n = len(g)
    ma20 = pd.Series(cl).rolling(20, min_periods=20).mean().to_numpy()
    adv20 = pd.Series(cl * vol).rolling(20, min_periods=20).mean().to_numpy()
    first = int(np.searchsorted(dates, "2019-01-01"))

    # ---- W1 dip-rally ----
    trail_min = pd.Series(lo).rolling(W1_LOOKBACK, min_periods=W1_LOOKBACK).min().to_numpy()
    t = max(W1_LOOKBACK, first)
    while t < n - 1:
        if lo[t] == trail_min[t]:
            s = t + int(np.argmin(lo[t:min(t + 4, n)]))
            hi_end = min(s + 1 + FWD, n)
            if hi_end > s + 1:
                fwd_max = cl[s + 1:hi_end].max()
                gain = fwd_max / cl[s] - 1.0
                if gain >= W1_GAIN:
                    peak = s + 1 + int(np.argmax(cl[s + 1:hi_end]))
                    # toc do song: so bar tu foot den lan dau +15%
                    hit = np.nonzero(cl[s + 1:hi_end] >= cl[s] * (1 + W1_GAIN))[0]
                    bars_to_15 = int(hit[0]) + 1 if len(hit) else -1
                    w1_rows.append(dict(
                        symbol=sym, foot_idx=s, foot_date=dates[s],
                        peak_idx=peak, peak_date=dates[peak],
                        gain_pct=round(gain * 100, 2), bars_to_15=bars_to_15,
                        foot_close=cl[s],
                        below_ma20=bool(cl[s] < ma20[s]) if not np.isnan(ma20[s]) else None,
                        drawdown_pre=round((cl[s] / np.nanmax(cl[max(0, s - 60):s + 1]) - 1) * 100, 2) if s > 0 else None,
                        adv20_bil=round(adv20[s] / 1e9, 2) if not np.isnan(adv20[s]) else None,
                    ))
                    t = peak + 1
                    continue
        t += 1

    # ---- W2 reclaim-MA20 ----
    below = cl < ma20
    t = max(25, first)
    while t < n - 1:
        if (not np.isnan(ma20[t]) and cl[t] >= ma20[t]
                and below[max(0, t - W2_BELOW):t].all() and t - W2_BELOW >= 0):
            hi_end = min(t + 1 + FWD, n)
            if hi_end > t + 1:
                fwd_max = cl[t + 1:hi_end].max()
                gain = fwd_max / cl[t] - 1.0
                if gain >= W2_GAIN:
                    peak = t + 1 + int(np.argmax(cl[t + 1:hi_end]))
                    w2_rows.append(dict(
                        symbol=sym, foot_idx=t, foot_date=dates[t],
                        peak_idx=peak, peak_date=dates[peak],
                        gain_pct=round(gain * 100, 2),
                        adv20_bil=round(adv20[t] / 1e9, 2) if not np.isnan(adv20[t]) else None,
                    ))
                    t = peak + 1
                    continue
        t += 1

w1 = pd.DataFrame(w1_rows)
w2 = pd.DataFrame(w2_rows)
w1.to_csv(os.path.join(OUT, "ss_waves_w1.csv"), index=False)
w2.to_csv(os.path.join(OUT, "ss_waves_w2.csv"), index=False)

# symbol-nam co du lieu (de chuan hoa)
oh["year"] = oh["date"].str[:4]
cov = oh[oh.year >= "2019"].groupby(["year", "symbol"]).size()
active = (cov >= 150).groupby("year").sum()

for name, ep in [("W1 dip-rally +15%/40bar", w1), ("W2 reclaim-MA20 +10%/40bar", w2)]:
    ep["year"] = ep.foot_date.str[:4]
    ep.loc[ep.foot_date >= "2026-07-01", "year"] = "drop"  # chua du fwd window
    byy = ep[ep.year != "drop"].groupby("year").agg(
        n=("symbol", "size"), gain_mean=("gain_pct", "mean"))
    byy["n_sym_active"] = active
    byy["waves_per_100sym"] = (byy.n / byy.n_sym_active * 100).round(1)
    print("\n===", name, "=== total:", len(ep))
    print(byy.round(2).to_string())
print("\nDONE ss_00")
