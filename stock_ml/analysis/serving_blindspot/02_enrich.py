# -*- coding: utf-8 -*-
"""Step 2: enrich closed trades -> trades_metrics.csv (one row per CLOSED trade).

Price-path metrics use RAW prices (slippage stripped from fills):
  limit_raw = entry_price / (1+slippage), exit_raw = exit_price / (1-slippage).
pnl_pct stays the engine's net figure (costs+slippage included).
holding_days / wait_bars / bars_to_peak are BAR counts, not calendar days.
"""
import os
import sys

SERVING = r"C:\Users\DUC CANH PC\Desktop\stock-serving"
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
SLIP = 0.0015

os.chdir(SERVING)
sys.path.insert(0, SERVING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from serving.ohlcv_store import OhlcvStore  # noqa: E402

trades = pd.read_csv(os.path.join(OUT, "trades_raw.csv"))
closed = trades[trades["exit_reason"] != "end_of_data"].copy()
print("closed trades:", len(closed), "open:", int((trades['exit_reason'] == 'end_of_data').sum()))

store = OhlcvStore("data/ohlcv.db")
ohlcv = store.load()
if "date" not in ohlcv.columns:
    raise SystemExit(f"unexpected ohlcv cols {list(ohlcv.columns)}")
ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date.astype(str)

# missed back-adjustment ex-dates (fake price gaps in the store)
badj = pd.read_csv(os.path.join(SERVING, "sieutinhieu_missed_adjustments.csv"))
bad_map: dict[str, list[str]] = {}
for _, r in badj.iterrows():
    bad_map.setdefault(str(r["symbol"]), []).append(str(r["ex_date"]))

sym_arrays = {}
for sym, g in ohlcv.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    sym_arrays[sym] = {
        "dates": g["date"].to_numpy(),
        "idx": {d: k for k, d in enumerate(g["date"])},
        "o": g["open"].to_numpy(float),
        "h": g["high"].to_numpy(float),
        "l": g["low"].to_numpy(float),
        "c": g["close"].to_numpy(float),
    }

rows = []
skipped = 0
for _, t in closed.iterrows():
    sym = t["symbol"]
    A = sym_arrays.get(sym)
    if A is None:
        skipped += 1
        continue
    idx = A["idx"]
    sidx = idx.get(str(t["entry_signal_date"]))
    eidx = idx.get(str(t["entry_date"]))
    xidx = idx.get(str(t["exit_date"]))
    if sidx is None or eidx is None or xidx is None:
        skipped += 1
        continue
    c, h, l, o = A["c"], A["h"], A["l"], A["o"]
    n = len(c)
    limit_raw = float(t["entry_price"]) / (1.0 + SLIP)
    exit_raw = float(t["exit_price"]) / (1.0 - SLIP)

    def preret(k):
        return c[sidx] / c[sidx - k] - 1.0 if sidx - k >= 0 else np.nan

    pre5, pre20, pre60 = preret(5), preret(20), preret(60)
    hold_h = h[eidx:xidx + 1]
    hold_c = c[eidx:xidx + 1]
    hold_l = l[eidx:xidx + 1]
    mfe = hold_h.max() / limit_raw - 1.0
    mae = hold_l.min() / limit_raw - 1.0
    bars_to_peak = int(np.argmax(hold_h))
    peak_close = hold_c.max()
    giveback = exit_raw / peak_close - 1.0

    def postret(k):
        return c[xidx + k] / c[xidx] - 1.0 if xidx + k < n else np.nan

    post_win = c[xidx + 1: min(xidx + 22, n)]
    post_runup21 = post_win.max() / c[xidx] - 1.0 if len(post_win) else np.nan
    sma20 = c[max(0, eidx - 19):eidx + 1].mean()
    peak20 = h[max(0, eidx - 19):eidx + 1].max()

    pnl = float(t["pnl_pct"])
    if pnl <= -0.15:
        bucket = "big_loss"
    elif pnl <= -0.05:
        bucket = "loss"
    elif pnl < 0.05:
        bucket = "small"
    elif pnl < 0.15:
        bucket = "win"
    else:
        bucket = "big_win"

    spans_bad = any(str(t["entry_signal_date"]) < xd <= str(t["exit_date"])
                    for xd in bad_map.get(sym, []))

    rows.append({
        "symbol": sym,
        "entry_date": t["entry_date"],
        "exit_date": t["exit_date"],
        "holding_days": int(t["holding_days"]),
        "pnl_pct": pnl,
        "exit_reason": t["exit_reason"],
        "entry_price": float(t["entry_price"]),
        "exit_price": float(t["exit_price"]),
        "signal_date": t["entry_signal_date"],
        "wait_bars": int(eidx - sidx),
        "touch_bar_red": bool(c[eidx] < o[eidx]),
        "pre_ret5": pre5, "pre_ret20": pre20, "pre_ret60": pre60,
        "signal_to_fill_drop": limit_raw / c[sidx] - 1.0,
        "entry_drop_from_peak20": limit_raw / peak20 - 1.0,
        "entry_dist_sma20": limit_raw / sma20 - 1.0,
        "hot_run": bool(pre20 >= 0.12) if not np.isnan(pre20) else False,
        "mfe_pct": mfe, "mae_pct": mae,
        "bars_to_peak": bars_to_peak,
        "giveback_pct": giveback,
        "post_ret5": postret(5), "post_ret10": postret(10),
        "post_ret21": postret(21), "post_ret42": postret(42),
        "post_max_runup21": post_runup21,
        "sold_too_early": bool(post_runup21 >= 0.05) if not np.isnan(post_runup21) else False,
        "bucket": bucket,
        "spans_bad_adjustment": spans_bad,
    })

tm = pd.DataFrame(rows)
tm.to_csv(os.path.join(OUT, "trades_metrics.csv"), index=False, encoding="utf-8")
print("trades_metrics.csv rows:", len(tm), "skipped:", skipped)
print(tm["bucket"].value_counts())
print("sold_too_early:", int(tm["sold_too_early"].sum()),
      "hot_run:", int(tm["hot_run"].sum()),
      "touch_bar_red:", int(tm["touch_bar_red"].sum()),
      "spans_bad_adjustment:", int(tm["spans_bad_adjustment"].sum()))
print("DONE step 2")
