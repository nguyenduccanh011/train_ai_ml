# -*- coding: utf-8 -*-
"""Step 4: daily_activity.csv — operational load per trading day (complaint #2).

Columns:
  date                  trading session (union of universe bar dates, >= 2020-01-02)
  n_buy_state           symbols whose model state is BUY (+1) that day
  n_new_buy_signals     new pullback-limit orders spawned that day (buy bar, symbol flat,
                        not blocked by entry gate/cooldown) — what a human must PLACE
  n_live_pending_limits limits live at session start: spawned <=40 bars ago, not yet
                        touched, symbol still flat (any of them can fill intraday)
  n_fills               positions opened that day (limit touched)
  n_exits               positions closed that day (next-bar-close sell fill)
  n_open_positions      positions held during the day
"""
import os
import sys
from collections import defaultdict

SERVING = r"C:\Users\DUC CANH PC\Desktop\stock-serving"
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
WINDOW = 40
OOS_START = "2020-01-02"

os.chdir(SERVING)
sys.path.insert(0, SERVING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from serving.ohlcv_store import OhlcvStore  # noqa: E402

signals = pd.read_csv(os.path.join(OUT, "signals.csv"))
uf = pd.read_csv(os.path.join(OUT, "unfilled_signals.csv"))
trades = pd.read_csv(os.path.join(OUT, "trades_raw.csv"))

ohlcv = OhlcvStore("data/ohlcv.db").load()
ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date.astype(str)

sym_arrays = {}
for sym, g in ohlcv.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    sym_arrays[sym] = {"dates": g["date"].to_numpy(),
                       "idx": {d: k for k, d in enumerate(g["date"])},
                       "l": g["low"].to_numpy(float)}

# position mask per symbol
pos_mask = {s: np.zeros(len(A["dates"]), dtype=bool) for s, A in sym_arrays.items()}
fills = defaultdict(int)
exits = defaultdict(int)
for t in trades.itertuples():
    A = sym_arrays.get(t.symbol)
    if A is None:
        continue
    eidx = A["idx"].get(str(t.entry_date))
    if eidx is None:
        continue
    fills[str(t.entry_date)] += 1
    if t.exit_reason == "end_of_data":
        xidx = len(A["dates"]) - 1
        pos_mask[t.symbol][eidx:xidx + 1] = True
    else:
        xidx = A["idx"].get(str(t.exit_date))
        exits[str(t.exit_date)] += 1
        if xidx is not None:
            pos_mask[t.symbol][eidx:xidx] = True

# spawned limits: (symbol, bar_idx, limit_price)
SPAWN_REASONS = {"unfilled", "unfilled_window_open", "skipped_pending_wait",
                 "last_bar_not_evaluated"}
spawns = []
for r in uf[uf["drop_reason"].isin(SPAWN_REASONS)].itertuples():
    A = sym_arrays.get(r.symbol)
    if A is None or np.isnan(r.limit_price):
        continue
    i = A["idx"].get(str(r.signal_date))
    if i is not None:
        spawns.append((r.symbol, i, float(r.limit_price)))
for t in trades.itertuples():
    A = sym_arrays.get(t.symbol)
    if A is None:
        continue
    i = A["idx"].get(str(t.entry_signal_date))
    if i is not None:
        spawns.append((t.symbol, i, float(t.entry_price) / 1.0015))

new_sig = defaultdict(int)
live = defaultdict(int)
for sym, i, limit in spawns:
    A = sym_arrays[sym]
    dates, lows = A["dates"], A["l"]
    n = len(dates)
    new_sig[dates[i]] += 1
    stop = min(i + WINDOW, n - 1)
    for k in range(i + 1, stop + 1):
        if pos_mask[sym][k]:
            break  # position open -> stacked limits cancelled
        live[dates[k]] += 1
        if lows[k] <= limit:
            break  # fills during session k (was live at session start)

open_pos = defaultdict(int)
for sym, m in pos_mask.items():
    dates = sym_arrays[sym]["dates"]
    for k in np.nonzero(m)[0]:
        open_pos[dates[k]] += 1

buy_state = signals[signals["signal"] > 0].groupby("date").size().to_dict()

all_days = sorted(d for d in {d for A in sym_arrays.values() for d in A["dates"]}
                  if d >= OOS_START)
da = pd.DataFrame({
    "date": all_days,
    "n_buy_state": [buy_state.get(d, 0) for d in all_days],
    "n_new_buy_signals": [new_sig.get(d, 0) for d in all_days],
    "n_live_pending_limits": [live.get(d, 0) for d in all_days],
    "n_fills": [fills.get(d, 0) for d in all_days],
    "n_exits": [exits.get(d, 0) for d in all_days],
    "n_open_positions": [open_pos.get(d, 0) for d in all_days],
})
da.to_csv(os.path.join(OUT, "daily_activity.csv"), index=False, encoding="utf-8")
print("daily_activity.csv rows:", len(da), da["date"].min(), "->", da["date"].max())
print(da[["n_buy_state", "n_new_buy_signals", "n_live_pending_limits",
          "n_fills", "n_open_positions"]].describe().round(2))
recent = da[da["date"] >= "2025-01-01"]
print("2025+ means:", recent[["n_new_buy_signals", "n_live_pending_limits",
                              "n_fills", "n_open_positions"]].mean().round(2).to_dict())
print("last day:", da.iloc[-1].to_dict())
print("DONE step 4")
