# -*- coding: utf-8 -*-
"""Step 5: sanity checks + headline stats (printed as JSON at the end)."""
import json
import os
import sqlite3
import sys

SERVING = r"C:\Users\DUC CANH PC\Desktop\stock-serving"
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
BUNDLE_ID = "bundle_n2_2643_wavestruct_la05_lamp02_top150_2025-01-01_wf"

os.chdir(SERVING)
sys.path.insert(0, SERVING)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from serving.ohlcv_store import OhlcvStore  # noqa: E402

tm = pd.read_csv(os.path.join(OUT, "trades_metrics.csv"))
uf = pd.read_csv(os.path.join(OUT, "unfilled_signals.csv"))
da = pd.read_csv(os.path.join(OUT, "daily_activity.csv"))
depths = pd.read_parquet(os.path.join(OUT, "depths.parquet"))
trades_raw = pd.read_csv(os.path.join(OUT, "trades_raw.csv"))

ohlcv = OhlcvStore("data/ohlcv.db").load()
ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date.astype(str)

# ---------- SPOT-CHECK 3 trades against engine arithmetic ----------
print("=== SPOT CHECKS (3 most recent closed trades) ===")
depth_map = {(s, d): v for s, d, v in zip(depths["symbol"], depths["date"], depths["eff_depth"])}
checks = []
for t in tm.sort_values("entry_date").tail(3).itertuples():
    dep = depth_map.get((t.symbol, str(t.signal_date)))
    g = ohlcv[ohlcv["symbol"] == t.symbol].set_index("date")
    sig_close = g.loc[str(t.signal_date), "close"]
    limit = sig_close * (1 - dep)
    entry_expected = limit * 1.0015
    low_fill = g.loc[str(t.entry_date), "low"]
    exit_expected = g.loc[str(t.exit_date), "close"] * 0.9985
    ok_entry = abs(entry_expected / t.entry_price - 1) < 1e-6
    ok_touch = low_fill <= limit * (1 + 1e-9)
    ok_exit = abs(exit_expected / t.exit_price - 1) < 1e-6
    checks.append(ok_entry and ok_touch and ok_exit)
    print(f"{t.symbol} sig {t.signal_date} depth {dep:.4f} limit {limit:.3f} "
          f"entry {t.entry_price:.3f} (exp {entry_expected:.3f} ok={ok_entry}) "
          f"low@fill {low_fill:.3f} touch={ok_touch} "
          f"exit {t.exit_price:.3f} (exp {exit_expected:.3f} ok={ok_exit})")
print("spot checks all OK:", all(checks))

# ---------- LEDGER cross-check (2026-07-08 confirmed) ----------
print("=== LEDGER CHECK 2026-07-08 ===")
con = sqlite3.connect("data/signals.db")
led = pd.read_sql(
    "SELECT symbol, signal FROM signal_log WHERE bundle_id=? AND signal_date=? AND status='confirmed'",
    con, params=(BUNDLE_ID, "2026-07-08"))
con.close()
sig_csv = pd.read_csv(os.path.join(OUT, "signals.csv"))
regen = sig_csv[sig_csv["date"] == "2026-07-08"].set_index("symbol")["signal"].to_dict()
mism = [(r.symbol, int(r.signal), int(regen.get(r.symbol, 0))) for r in led.itertuples()
        if int(r.signal) != int(regen.get(r.symbol, 0))]
print(f"ledger rows: {len(led)}, mismatches vs regenerated: {len(mism)}", mism[:10])

# ---------- HEADLINE STATS ----------
pnl = tm["pnl_pct"]
wins = pnl[pnl > 0]
losses = pnl[pnl < 0]
pf = wins.sum() / abs(losses.sum()) if len(losses) else np.inf
worst = tm.nsmallest(10, "pnl_pct")[
    ["symbol", "signal_date", "entry_date", "exit_date", "pnl_pct", "exit_reason",
     "wait_bars", "hot_run", "touch_bar_red", "spans_bad_adjustment"]]

drop_counts = uf["drop_reason"].value_counts().to_dict()
n_filled = len(trades_raw)
n_unfilled = drop_counts.get("unfilled", 0) + drop_counts.get("unfilled_window_open", 0)
placed = n_filled + n_unfilled
nf = uf[uf["drop_reason"].isin(["unfilled", "unfilled_window_open"])]

stats = {
    "n_trades_total": int(len(trades_raw)),
    "n_trades_closed": int(len(tm)),
    "n_open_positions_now": int((trades_raw["exit_reason"] == "end_of_data").sum()),
    "win_rate_closed": round(float((pnl > 0).mean()), 4),
    "avg_pnl_pct": round(float(pnl.mean()), 5),
    "median_pnl_pct": round(float(pnl.median()), 5),
    "sum_pnl_units": round(float(pnl.sum()), 2),
    "profit_factor": round(float(pf), 3),
    "bucket_counts": tm["bucket"].value_counts().to_dict(),
    "bucket_pnl_sum": tm.groupby("bucket")["pnl_pct"].sum().round(2).to_dict(),
    "exit_reason_counts": trades_raw["exit_reason"].value_counts().to_dict(),
    "buy_signal_disposition": {
        "filled": n_filled, **drop_counts,
        "total_buy_bars": int(n_filled + len(uf)),
    },
    "fill_rate_of_placed_limits": round(n_filled / placed, 4),
    "unfilled_atmarket_ret21_mean": round(float(np.nanmean(nf["atmarket_ret21"])), 4),
    "unfilled_atmarket_ret21_median": round(float(np.nanmedian(nf["atmarket_ret21"])), 4),
    "unfilled_max_runup42_mean": round(float(np.nanmean(nf["max_runup_42"])), 4),
    "avg_pending_limits_per_day_all": round(float(da["n_live_pending_limits"].mean()), 1),
    "avg_pending_limits_per_day_2025plus": round(
        float(da[da["date"] >= "2025-01-01"]["n_live_pending_limits"].mean()), 1),
    "avg_new_buy_signals_per_day_2025plus": round(
        float(da[da["date"] >= "2025-01-01"]["n_new_buy_signals"].mean()), 1),
    "avg_open_positions_2025plus": round(
        float(da[da["date"] >= "2025-01-01"]["n_open_positions"].mean()), 1),
    "last_day_pending_limits_mine_vs_serving": [
        int(da.iloc[-1]["n_live_pending_limits"]), 171],
    "sold_too_early_count": int(tm["sold_too_early"].sum()),
    "sold_too_early_rate": round(float(tm["sold_too_early"].mean()), 4),
    "hot_run_count": int(tm["hot_run"].sum()),
    "touch_bar_red_rate": round(float(tm["touch_bar_red"].mean()), 4),
    "big_loss_spans_bad_adjustment": int(
        tm[(tm["bucket"] == "big_loss") & tm["spans_bad_adjustment"]].shape[0]),
    "date_range_trades": [str(tm["signal_date"].min()), str(tm["exit_date"].max())],
}
print("=== HEADLINE ===")
print(json.dumps(stats, indent=1))
print("=== WORST 10 ===")
print(worst.to_string(index=False))
worst.to_csv(os.path.join(OUT, "worst10.csv"), index=False, encoding="utf-8")
with open(os.path.join(OUT, "headline_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=1)
print("DONE step 5")
