# -*- coding: utf-8 -*-
"""Adversarial recomputation of bad-buys claims. Read-only except this file."""
import sqlite3

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
SLIP = 0.0015
BIG = -0.15

t = pd.read_csv(OUT + r"\trades_metrics.csv")
print("n_closed", len(t), "sum_pnl", round(t.pnl_pct.sum(), 2),
      "WR", round((t.pnl_pct > 0).mean(), 4))

def stats(g, label):
    n = len(g)
    print(f"{label}: n={n} ({n/len(t)*100:.1f}%) WR={(g.pnl_pct>0).mean()*100:.1f}% "
          f"avg={g.pnl_pct.mean()*100:+.2f}% sum={g.pnl_pct.sum():+.2f}u "
          f"le8={(g.pnl_pct<=-0.08).mean()*100:.2f}% "
          f"le15={(g.pnl_pct<=BIG).mean()*100:.2f}% (n={(g.pnl_pct<=BIG).sum()}) "
          f"mae={g.mae_pct.mean()*100:.2f}% mfe={g.mfe_pct.mean()*100:.2f}%")

print("\n=== C1: candle split ===")
for red, g in t.groupby("touch_bar_red"):
    stats(g, f"red={red}")

print("\n=== C2: drop red cohort ===")
red = t[t.touch_bar_red]
bl = red[red.pnl_pct <= BIG]
print(f"n_removed={len(red)} ({len(red)/len(t)*100:.1f}%) pnl_removed={red.pnl_pct.sum():+.2f}u "
      f"share_of_total={red.pnl_pct.sum()/t.pnl_pct.sum()*100:.1f}% "
      f"bigloss_removed={len(bl)} ({bl.pnl_pct.sum():+.2f}u) net_delta={-red.pnl_pct.sum():+.2f}u")
tot_bl = t[t.pnl_pct <= BIG]
print(f"total bigloss={len(tot_bl)} sum={tot_bl.pnl_pct.sum():+.2f}u")

print("\n=== C3: wait_bars gradient ===")
t["wb"] = pd.cut(t.wait_bars, bins=[0, 1, 2, 5, 10, 20, 40])
for b, g in t.groupby("wb", observed=True):
    stats(g, f"wait {b}")

print("\n=== C4: hot_run x fast 2x2 ===")
print("hot_run == pre_ret20>=0.12 ?", (t.hot_run == (t.pre_ret20 >= 0.12)).mean())
t["fast"] = t.wait_bars <= 2
for (h, f), g in t.groupby(["hot_run", "fast"]):
    stats(g, f"hot={h} fast={f}")

print("\n=== C5: pre_ret20 deciles + skip thresholds ===")
print("pre_ret20 NaN:", t.pre_ret20.isna().sum())
t["d_pre20"] = pd.qcut(t.pre_ret20, 10, labels=False)
for d, g in t.groupby("d_pre20"):
    lo, hi = g.pre_ret20.min(), g.pre_ret20.max()
    stats(g, f"D{int(d)} [{lo:+.3f},{hi:+.3f}]")
for thr in (0.12, 0.20, 0.25, 0.30):
    m = t[t.pre_ret20 >= thr]
    b = m[m.pnl_pct <= BIG]
    print(f"skip>={thr}: n={len(m)} removed={m.pnl_pct.sum():+.2f}u "
          f"({m.pnl_pct.sum()/t.pnl_pct.sum()*100:.1f}%) net={-m.pnl_pct.sum():+.2f}u "
          f"bigloss={len(b)} ({b.pnl_pct.sum():+.2f}u)")

print("\n=== C6: entry_drop_from_peak20 deciles ===")
t["d_drop"] = pd.qcut(t.entry_drop_from_peak20, 10, labels=False)
for d, g in t.groupby("d_drop"):
    lo, hi = g.entry_drop_from_peak20.min(), g.entry_drop_from_peak20.max()
    stats(g, f"D{int(d)} [{lo:+.3f},{hi:+.3f}]")

print("\n=== C7+C8: MAE ladder + flat hard stop ===")
for thr in (-0.08, -0.12, -0.20):
    g = t[t.mae_pct <= thr]
    n = len(g)
    rec = (g.pnl_pct > 0).sum()
    stopped = thr * n
    actual = g.pnl_pct.sum()
    print(f"mae<={thr}: n={n} ({n/len(t)*100:.1f}%) recovered_win={rec} ({rec/n*100:.1f}%) "
          f"avg_actual_exit={g.pnl_pct.mean()*100:+.2f}% stop_sum={stopped:+.2f}u "
          f"actual_sum={actual:+.2f}u delta(stop-actual)={stopped-actual:+.2f}u")

print("\n=== C9: cheaper-before join with ohlcv.db ===")
con = sqlite3.connect(r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db")
ohlcv = pd.read_sql("select symbol, date, low, close from ohlcv", con)
arr = {}
for sym, g in ohlcv.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    arr[sym] = ({d: k for k, d in enumerate(g["date"])},
                g["low"].to_numpy(float), g["close"].to_numpy(float))
res = []
touch_ok = 0
n_eval = 0
for r in t.itertuples():
    A = arr.get(r.symbol)
    if A is None:
        continue
    idx, l, c = A
    sidx = idx.get(str(r.signal_date))
    eidx = idx.get(str(r.entry_date))
    if sidx is None or eidx is None:
        continue
    limit_raw = r.entry_price / (1.0 + SLIP)
    n_eval += 1
    if l[eidx] <= limit_raw * 1.0005:
        touch_ok += 1
    c5s = c[sidx - 5] < limit_raw if sidx >= 5 else np.nan
    lo10 = l[sidx - 10:sidx].min() < limit_raw if sidx >= 10 else np.nan
    c5f = c[eidx - 5] < limit_raw if eidx >= 5 else np.nan
    res.append((c5s, lo10, c5f, r.pnl_pct))
R = pd.DataFrame(res, columns=["c5s", "lo10", "c5f", "pnl"])
print("joined:", n_eval, "fill-bar-touch sanity:", touch_ok / n_eval)
print(f"c5_before_signal_cheaper: {R.c5s.mean()*100:.1f}% (valid {R.c5s.notna().sum()})")
print(f"low10_before_signal_cheaper: {R.lo10.mean()*100:.1f}%")
print(f"c5_before_fill_cheaper: {R.c5f.mean()*100:.1f}% | pnl True "
      f"{R[R.c5f==True].pnl.mean()*100:+.2f}% vs False {R[R.c5f==False].pnl.mean()*100:+.2f}%")

print("\n=== C10: eff_depth ===")
dep = pd.read_parquet(OUT + r"\depths.parquet")
dep["date"] = dep["date"].astype(str)
m = t.merge(dep, left_on=["symbol", "signal_date"], right_on=["symbol", "date"], how="left")
print("matched:", m.eff_depth.notna().sum(), "/", len(m))
m["db"] = pd.cut(m.eff_depth, bins=[0.020, 0.030, 0.040, 0.0449, 0.0451])
print("below 0.020:", (m.eff_depth <= 0.020).sum(), "min:", m.eff_depth.min())
g = m[m.eff_depth <= 0.030]
stats(g, "eff_depth<=0.030")
for b, g in m.groupby("db", observed=True):
    stats(g, f"depth {b}")
for thr in (0.0449, 0.040):
    d = m[m.eff_depth >= thr]
    print(f"floor@{thr}: n_drop={len(d)} net={-d.pnl_pct.sum():+.2f}u")

print("\n=== C11: wait 21-40 churn + limit-days ===")
g = t[t.wait_bars >= 21]
stats(g, "wait>=21")
print("share of profit:", round(g.pnl_pct.sum() / t.pnl_pct.sum() * 100, 1), "%")
print("mfe wait<=2:", round(t[t.wait_bars <= 2].mfe_pct.mean() * 100, 1), "%")
w = t[(t.pnl_pct <= BIG)].sort_values("pnl_pct")
print("named knives check:")
for sym in ("BID", "NKG", "FRT"):
    s = t[(t.symbol == sym) & (t.pnl_pct <= BIG)][
        ["symbol", "signal_date", "entry_date", "wait_bars", "pnl_pct"]]
    print(s.to_string(index=False) if len(s) else f"{sym}: none")
unf = pd.read_csv(OUT + r"\unfilled_signals.csv")
raw = pd.read_csv(OUT + r"\trades_raw.csv")
n_open = int((raw.exit_reason == "end_of_data").sum())
n_unf = len(unf)
sum_wait_closed = t.wait_bars.sum()
open_est = n_open * 9.3  # claim's estimate for open positions' wait
cur = n_unf * 40 + 264 * 20 + sum_wait_closed + open_est
kept = t[t.wait_bars <= 20].wait_bars.sum()
w20 = n_unf * 20 + 264 * 20 + kept + len(g) * 20 + open_est
print(f"n_unfilled={n_unf} n_open={n_open} sum_wait_closed={sum_wait_closed} "
      f"kept(le20)={kept}")
print(f"cur_limit_days={cur:.0f} w20={w20:.0f} reduction={(1-w20/cur)*100:.1f}%")

print("\n=== C12: big_loss portrait + combo ===")
b = t[t.pnl_pct <= BIG]
print(f"n={len(b)} sum={b.pnl_pct.sum():+.2f}u red={b.touch_bar_red.mean()*100:.1f}% "
      f"(base {t.touch_bar_red.mean()*100:.1f}%) fast={(b.wait_bars<=2).mean()*100:.1f}% "
      f"(base {(t.wait_bars<=2).mean()*100:.1f}%) pre20={b.pre_ret20.mean()*100:+.2f}% "
      f"(base {t.pre_ret20.mean()*100:+.2f}%) hot={b.hot_run.mean()*100:.1f}%")
combo = t[(t.wait_bars <= 2) & t.touch_bar_red & (t.pre_ret20 >= 0.12)]
cb = combo[combo.pnl_pct <= BIG]
print(f"combo: n={len(combo)} sum={combo.pnl_pct.sum():+.2f}u net={-combo.pnl_pct.sum():+.2f}u "
      f"bigloss={len(cb)} ({cb.pnl_pct.sum():+.2f}u)")
print(combo.nlargest(5, "pnl_pct")[["symbol", "entry_date", "pnl_pct"]].to_string(index=False))

print("\n=== C13: scale of big losses ===")
g20 = t[t.mae_pct <= -0.20]
print(f"bigloss sum={b.pnl_pct.sum():+.2f}u = {abs(b.pnl_pct.sum())/t.pnl_pct.sum()*100:.1f}% "
      f"of total {t.pnl_pct.sum():+.2f}u | mae<=-20%: n={len(g20)} sum={g20.pnl_pct.sum():+.2f}u")
