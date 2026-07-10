# -*- coding: utf-8 -*-
"""Step 3: unfilled_signals.csv — every BUY signal that did NOT become a trade fill.

drop_reason values (engine-exact, from instrumented run — not simulated):
  unfilled              limit placed, low never touched it in the 40-bar window
  unfilled_window_open  same, but the 40-bar window is truncated by end-of-data (still live)
  entry_market_gate     skipped by the market-weak entry gate at the signal bar
  cooldown              skipped by the 4-bar re-entry cooldown after a losing exit
  in_position           symbol already held a position on the signal bar
  skipped_pending_wait  bar lies between an earlier signal and its eventual fill (engine
                        sequential scan jumps over it; operationally still a stacked limit)
  last_bar_not_evaluated signal on the final stored bar (engine loop stops at n-1)
  unknown               (should be ~0)
limit_price: engine-recorded for 'unfilled'; close*(1-eff_depth) (conviction-scaled,
engine-exact depth array) for the other reasons (hypothetical limit).
atmarket_ret*: close[i+1+k]/close[i+1]-1 (what an at-market next-bar-close entry returns).
max_runup_42: max(close[i+1 .. i+42]) / close[i+1] - 1.
"""
import os
import sys

SERVING = r"C:\Users\DUC CANH PC\Desktop\stock-serving"
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
WINDOW = 40

os.chdir(SERVING)
sys.path.insert(0, SERVING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from serving.ohlcv_store import OhlcvStore  # noqa: E402

signals = pd.read_csv(os.path.join(OUT, "signals.csv"))
events = pd.read_csv(os.path.join(OUT, "events.csv"))
trades = pd.read_csv(os.path.join(OUT, "trades_raw.csv"))
depths = pd.read_parquet(os.path.join(OUT, "depths.parquet"))

ohlcv = OhlcvStore("data/ohlcv.db").load()
ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date.astype(str)

sym_arrays = {}
for sym, g in ohlcv.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    sym_arrays[sym] = {
        "dates": g["date"].to_numpy(),
        "idx": {d: k for k, d in enumerate(g["date"])},
        "l": g["low"].to_numpy(float),
        "c": g["close"].to_numpy(float),
    }

depth_map = {}
for sym, g in depths.groupby("symbol"):
    depth_map[sym] = dict(zip(g["date"], g["eff_depth"]))

# event lookup by (symbol, bar_idx)
ev_map = {(r.symbol, int(r.bar_idx)): r for r in events.itertuples()}

# per-symbol masks: in-position and pending-wait
pos_mask = {s: np.zeros(len(A["c"]), dtype=bool) for s, A in sym_arrays.items()}
for t in trades.itertuples():
    A = sym_arrays.get(t.symbol)
    if A is None:
        continue
    eidx = A["idx"].get(str(t.entry_date))
    if eidx is None:
        continue
    if t.exit_reason == "end_of_data":
        xidx = len(A["c"]) - 1
        pos_mask[t.symbol][eidx:xidx + 1] = True
    else:
        xidx = A["idx"].get(str(t.exit_date))
        if xidx is None:
            continue
        pos_mask[t.symbol][eidx:xidx] = True  # exit fill bar itself is re-evaluated

pend_mask = {s: np.zeros(len(A["c"]), dtype=bool) for s, A in sym_arrays.items()}
for r in events[events["kind"] == "filled"].itertuples():
    m = pend_mask.get(r.symbol)
    if m is not None:
        m[int(r.bar_idx) + 1: int(r.aux_idx)] = True

buys = signals[signals["signal"] > 0]
rows = []
counts: dict[str, int] = {}
for b in buys.itertuples():
    sym, date = b.symbol, b.date
    A = sym_arrays.get(sym)
    if A is None:
        continue
    i = A["idx"].get(date)
    if i is None:
        continue
    n = len(A["c"])
    ev = ev_map.get((sym, i))
    if ev is not None and ev.kind == "filled":
        continue  # became a trade
    if ev is not None:
        if ev.kind == "missed":
            reason = "unfilled_window_open" if i + WINDOW > n - 1 else "unfilled"
            limit = float(ev.limit_price)
        elif ev.kind == "skip_market_weak":
            reason = "entry_market_gate"
            limit = np.nan
        elif ev.kind == "skip_cooldown":
            reason = "cooldown"
            limit = np.nan
        else:
            reason = ev.kind
            limit = np.nan
    elif pos_mask[sym][i]:
        reason = "in_position"
        limit = np.nan
    elif pend_mask[sym][i]:
        reason = "skipped_pending_wait"
        limit = np.nan
    elif i >= n - 1:
        reason = "last_bar_not_evaluated"
        limit = np.nan
    else:
        reason = "unknown"
        limit = np.nan
    if np.isnan(limit):
        dep = depth_map.get(sym, {}).get(date, np.nan)
        limit = A["c"][i] * (1.0 - dep) if not np.isnan(dep) else np.nan

    c, l = A["c"], A["l"]
    w_end = min(i + WINDOW, n - 1)
    max_low = l[i + 1: w_end + 1].min() if w_end > i else np.nan
    nb = i + 1  # next bar (at-market baseline)

    def amret(k):
        return c[nb + k] / c[nb] - 1.0 if (nb < n and nb + k < n) else np.nan

    runup_win = c[nb: min(nb + 43, n)]
    max_runup = runup_win.max() / c[nb] - 1.0 if nb < n else np.nan
    rows.append({
        "symbol": sym, "signal_date": date, "signal_close": c[i],
        "limit_price": limit, "max_low_in_window": max_low,
        "window_end_close": c[w_end] if w_end > i else np.nan,
        "atmarket_ret5": amret(5), "atmarket_ret10": amret(10),
        "atmarket_ret21": amret(21), "atmarket_ret42": amret(42),
        "max_runup_42": max_runup, "drop_reason": reason,
    })
    counts[reason] = counts.get(reason, 0) + 1

uf = pd.DataFrame(rows)
uf.to_csv(os.path.join(OUT, "unfilled_signals.csv"), index=False, encoding="utf-8")
print("unfilled_signals.csv rows:", len(uf))
print(pd.Series(counts).sort_values(ascending=False))
nf = uf[uf["drop_reason"].isin(["unfilled", "unfilled_window_open"])]
print("pure unfilled:", len(nf),
      "median atmarket_ret21:", np.nanmedian(nf["atmarket_ret21"]),
      "mean atmarket_ret21:", np.nanmean(nf["atmarket_ret21"]),
      "mean max_runup_42:", np.nanmean(nf["max_runup_42"]))
print("DONE step 3")
