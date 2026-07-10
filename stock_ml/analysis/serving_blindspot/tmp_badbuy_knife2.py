# -*- coding: utf-8 -*-
"""Follow-up: red/green at -8% cutoff; window-cap limit-day arithmetic; wait x mae."""
import os, json
import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
tm = pd.read_csv(os.path.join(OUT, "trades_metrics.csv"))
tm["win"] = tm["pnl_pct"] > 0

print("== red/green with loss cutoffs -8% and -15% ==")
for flag, lbl in [(True, "red"), (False, "green")]:
    g = tm[tm["touch_bar_red"] == flag]
    print(json.dumps({"touch": lbl, "n": len(g),
                      "wr": round(float(g["win"].mean()), 4),
                      "avg_pnl": round(float(g["pnl_pct"].mean()), 4),
                      "rate_le_m8": round(float((g["pnl_pct"] <= -0.08).mean()), 4),
                      "rate_le_m15": round(float((g["pnl_pct"] <= -0.15).mean()), 4)}))

print("== stop-vs-actual arithmetic (static) ==")
for thr in [-0.08, -0.12, -0.20]:
    g = tm[tm["mae_pct"] <= thr]
    stopped = thr * len(g)  # every such trade exits at the stop level (optimistic: no gap-through)
    print(json.dumps({"stop": thr, "n_hit": len(g),
                      "actual_sum_u": round(float(g["pnl_pct"].sum()), 2),
                      "stopped_sum_u": round(stopped, 2),
                      "delta_u_stop_minus_actual": round(stopped - float(g["pnl_pct"].sum()), 2)}))

print("== limit-days arithmetic: w40 now vs w20 cap ==")
uf = pd.read_csv(os.path.join(OUT, "unfilled_signals.csv"))
raw = pd.read_csv(os.path.join(OUT, "trades_raw.csv"))
raw["wait"] = np.nan
# recompute wait for all fills incl. open from dates via trades_metrics where possible
tmw = tm.set_index(["symbol", "entry_date"])["wait_bars"]
n_uf_closedwin = int((uf["drop_reason"] == "unfilled").sum())
n_uf_open = int((uf["drop_reason"] == "unfilled_window_open").sum())
wait_sum_closed = float(tm["wait_bars"].sum())
mean_wait = float(tm["wait_bars"].mean())
n_open_trades = int((raw["exit_reason"] == "end_of_data").sum())
# current limit-days (placed limits only; stacked-limit cancels ignored on both sides)
cur = n_uf_closedwin * 40 + n_uf_open * 20 + wait_sum_closed + n_open_trades * mean_wait
# w20 cap: unfilled live 20; fills with wait<=20 keep wait; fills wait>=21 become unfilled@20
n_fill_gt20 = int((tm["wait_bars"] >= 21).sum())
wait_sum_le20 = float(tm.loc[tm["wait_bars"] <= 20, "wait_bars"].sum())
cap = (n_uf_closedwin * 20 + n_uf_open * 15 + wait_sum_le20
       + n_fill_gt20 * 20 + n_open_trades * min(mean_wait, 20))
print(json.dumps({"limit_days_now": int(cur), "limit_days_w20cap": int(cap),
                  "reduction_pct": round(1 - cap / cur, 3),
                  "n_unfilled_closedwin": n_uf_closedwin, "n_unfilled_open": n_uf_open,
                  "wait_sum_closed_fills": int(wait_sum_closed),
                  "n_fills_wait_ge21": n_fill_gt20}))

print("== wait buckets x avg mae / mfe / bars_to_peak ==")
wb = pd.cut(tm["wait_bars"], bins=[0, 2, 10, 20, 40], labels=["1-2", "3-10", "11-20", "21-40"])
for lbl, g in tm.groupby(wb, observed=True):
    print(json.dumps({"wait": str(lbl), "n": len(g),
                      "avg_mae": round(float(g["mae_pct"].mean()), 4),
                      "avg_mfe": round(float(g["mfe_pct"].mean()), 4),
                      "avg_pnl": round(float(g["pnl_pct"].mean()), 4),
                      "avg_hold": round(float(g["holding_days"].mean()), 1),
                      "pnl_per_hold_bar": round(float(g["pnl_pct"].sum() / g["holding_days"].sum()), 5)}))

print("== hot & fast & red big winners (counterexample to combo filter) ==")
k = tm[(tm["wait_bars"] <= 2) & tm["touch_bar_red"] & (tm["pre_ret20"] >= 0.12)].nlargest(4, "pnl_pct")
print(k[["symbol", "signal_date", "entry_date", "exit_date", "pnl_pct", "pre_ret20", "mae_pct"]].to_string(index=False))
print("DONE")
