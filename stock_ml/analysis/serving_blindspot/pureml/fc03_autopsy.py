# -*- coding: utf-8 -*-
"""FAMILY_CHAMPIONS step 3: autopsy trade-level 1 vo dich (s42) vs gb_x08 (s42).

Usage: python fc03_autopsy.py <trades_csv> <label>
Per-year pnl/PF/WR/hold, exit_reason mix, overlap gb_x08, diem chet, ngach thang.
(Adapted tu pm05_autopsy.py.)
"""
import sys

import numpy as np
import pandas as pd

GB_CSV = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap\gbx08_s42_trades.csv"

CSV, LABEL = sys.argv[1], sys.argv[2]
t = pd.read_csv(CSV, parse_dates=["entry_date", "exit_date"])
g = pd.read_csv(GB_CSV, parse_dates=["entry_date", "exit_date"])


def pf(s):
    gains = s[s > 0].sum(); losses = -s[s <= 0].sum()
    return gains / losses if losses > 0 else np.inf


def yearly(df, datecol):
    grp = df.groupby(df[datecol].dt.year)["pnl_pct"]
    return pd.DataFrame({
        "n": grp.size(), "pnl_sum": grp.sum().round(2), "wr": grp.apply(lambda s: (s > 0).mean()).round(3),
        "pf": grp.apply(pf).round(2), "avg": grp.mean().round(4),
        "med_hold": df.groupby(df[datecol].dt.year)["holding_days"].median(),
        "worst": grp.min().round(3),
    })


print(f"{LABEL} n={len(t)} pnl_sum={t.pnl_pct.sum():.2f} | gb_x08 n={len(g)} pnl_sum={g.pnl_pct.sum():.2f}")
print(f"\n== {LABEL} per ENTRY-year =="); print(yearly(t, "entry_date").to_string())
print(f"\n== {LABEL} per EXIT-year ==");  print(yearly(t, "exit_date").to_string())
print(f"\n== exit_reason mix {LABEL} =="); print(t.groupby("exit_reason")["pnl_pct"].agg(["size", "sum", "mean"]).round(3).to_string())
print("\n== hold quantiles", LABEL, ":", t.holding_days.quantile([.25, .5, .75, .9, .99]).to_dict())
print("== hold quantiles gb_x08:", g.holding_days.quantile([.25, .5, .75, .9, .99]).to_dict())

tj = t.copy(); gj = g.copy()
tj["key"] = tj.symbol + "|" + tj.entry_date.dt.strftime("%Y-%m-%d")
gj["key"] = gj.symbol + "|" + gj.entry_date.dt.strftime("%Y-%m-%d")
exact = set(tj.key) & set(gj.key)
print(f"\n== entry overlap exact: {len(exact)} = {len(exact)/len(t):.1%} of {LABEL}, {len(exact)/len(g):.1%} of gb_x08")

gsym = {s: df.entry_date.values for s, df in gj.groupby("symbol")}


def fuzzy_match(row):
    arr = gsym.get(row.symbol)
    if arr is None:
        return False
    return bool((np.abs(arr - np.datetime64(row.entry_date)) <= np.timedelta64(5, "D")).any())


tj["fuzzy"] = tj.apply(fuzzy_match, axis=1)
print(f"== entry overlap fuzzy(±5d): {tj.fuzzy.mean():.1%} of {LABEL} entries have a gb_x08 entry nearby")

m = tj.merge(gj, on="key", suffixes=("_t", "_g"))
if len(m):
    m["d_pnl"] = m.pnl_pct_t - m.pnl_pct_g
    m["d_hold"] = m.holding_days_t - m.holding_days_g
    print(f"\n== shared trades n={len(m)}: {LABEL} pnl_sum={m.pnl_pct_t.sum():.2f} vs gb_x08 {m.pnl_pct_g.sum():.2f} "
          f"(d={m.d_pnl.sum():+.2f}); mean d_hold={m.d_hold.mean():+.1f}d")
    print(m.groupby(m.entry_date_t.dt.year).agg(n=("d_pnl", "size"), var=("pnl_pct_t", "sum"),
          gbx08=("pnl_pct_g", "sum"), d=("d_pnl", "sum")).round(2).to_string())

only_t = tj[~tj.key.isin(exact)]
only_g = gj[~gj.key.isin(exact)]
print(f"\n== {LABEL}-only n={len(only_t)} pnl={only_t.pnl_pct.sum():.2f} wr={(only_t.pnl_pct>0).mean():.3f} pf={pf(only_t.pnl_pct):.2f}")
print(only_t.groupby(only_t.entry_date.dt.year)["pnl_pct"].agg(["size", "sum"]).round(2).to_string())
print(f"\n== gb_x08-only n={len(only_g)} pnl={only_g.pnl_pct.sum():.2f} wr={(only_g.pnl_pct>0).mean():.3f} pf={pf(only_g.pnl_pct):.2f}")
print(only_g.groupby(only_g.entry_date.dt.year)["pnl_pct"].agg(["size", "sum"]).round(2).to_string())

print(f"\n== top-15 losers {LABEL} ==")
print(t.nsmallest(15, "pnl_pct")[["symbol", "entry_date", "exit_date", "holding_days", "pnl_pct", "exit_reason"]].round(3).to_string(index=False))
print("\n== pnl quantiles", LABEL, ":", t.pnl_pct.quantile([.01, .05, .25, .5, .75, .95, .99]).round(3).to_dict())
print("== pnl quantiles gb_x08:", g.pnl_pct.quantile([.01, .05, .25, .5, .75, .95, .99]).round(3).to_dict())

tw = t.nlargest(15, "pnl_pct").copy()


def gb_same(row):
    sub = gj[(gj.symbol == row.symbol) & (abs(gj.entry_date - row.entry_date) <= pd.Timedelta(days=10))]
    if len(sub) == 0:
        return "MISS"
    return f"{sub.pnl_pct.max():.2f}"


tw["gb_pnl_near"] = tw.apply(gb_same, axis=1)
print(f"\n== top-15 winners {LABEL} (vs gb_x08 cung ma ±10d) ==")
print(tw[["symbol", "entry_date", "exit_date", "holding_days", "pnl_pct", "gb_pnl_near"]].round(3).to_string(index=False))
print("FC03_DONE")
