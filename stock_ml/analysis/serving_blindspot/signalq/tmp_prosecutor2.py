"""PROSECUTOR follow-up: (a) red-candle / hot-run cohort integrity, (b) 2026 fake-bull anatomy,
(c) 2022 washout-window overlap of added big losses."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

REPO = Path(__file__).resolve().parents[4]
D = Path(__file__).parent
stack = pd.read_csv(D / "st_dsb60_snr08_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
champ = pd.read_csv(D / "st_champ2646_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
stack["k"] = stack.symbol + "|" + stack.entry_date.astype(str)
champ["k"] = champ.symbol + "|" + champ.entry_date.astype(str)

uni = sorted(set(stack.symbol) | set(champ.symbol))
con = duckdb.connect(str(REPO / "market_data" / "market.duckdb"), read_only=True)
px = con.execute(
    "SELECT symbol, date, open, close FROM ohlcv WHERE symbol IN ({}) AND date >= '2019-01-01'".format(
        ",".join("'" + s + "'" for s in uni))).df()
con.close()
px["date"] = pd.to_datetime(px["date"])
piv = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
opiv = px.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()

# (a) cohort features per trade: entry-day candle color, pre_ret20 before entry
ret20 = piv.pct_change(20)
def add_cohorts(df):
    red, hot = [], []
    for r in df.itertuples():
        try:
            c = piv.at[r.entry_date, r.symbol]; o = opiv.at[r.entry_date, r.symbol]
            red.append(bool(c < o))
        except KeyError:
            red.append(np.nan)
        try:
            # pre_ret20 at the bar BEFORE entry fill (causal)
            idx = piv.index.get_loc(r.entry_date)
            hot.append(float(ret20.iloc[idx - 1][r.symbol]))
        except Exception:
            hot.append(np.nan)
    df = df.copy(); df["red"] = red; df["pre_ret20"] = hot
    return df

stack = add_cohorts(stack); champ = add_cohorts(champ)
hot_thr = champ.pre_ret20.quantile(0.9)
for nm, df in [("champ", champ), ("stack", stack)]:
    rc = df[df.red == True]  # noqa: E712
    hr = df[df.pre_ret20 >= hot_thr]
    print(f"{nm}: red-candle cohort n={len(rc)} pnl={rc.pnl_pct.sum():+.1f}u "
          f"({rc.pnl_pct.sum()/df.pnl_pct.sum()*100:.1f}% pnl) | "
          f"hot-run(pre_ret20>={hot_thr:.3f}) n={len(hr)} pnl={hr.pnl_pct.sum():+.1f}u")
removed = champ[~champ.k.isin(set(stack.k))]
added = stack[~stack.k.isin(set(champ.k))]
print(f"removed trades: n={len(removed)} red%={removed.red.mean()*100:.1f} (champ base {champ.red.mean()*100:.1f}) "
      f"hot%={(removed.pre_ret20>=hot_thr).mean()*100:.1f} (base {(champ.pre_ret20>=hot_thr).mean()*100:.1f}) "
      f"removed-red pnl {removed[removed.red==True].pnl_pct.sum():+.2f}")  # noqa: E712
print(f"added trades:   n={len(added)} red%={added.red.mean()*100:.1f} hot%={(added.pre_ret20>=hot_thr).mean()*100:.1f}")

# (b) 2026 fake-bull anatomy: EW index vs MA60 (bull mask, persist 3), decline depth
ew = (1.0 + piv.pct_change().mean(axis=1).fillna(0.0)).cumprod()
ma60 = ew.rolling(60).mean()
above = ew > ma60
bull = above.rolling(3).min().astype(bool)  # persist 3 consecutive
w = ew.loc["2025-12-01":"2026-04-01"]
peak = w.loc[:"2026-03-12"].max()
peak_d = w.loc[:"2026-03-12"].idxmax()
trough = w.loc["2026-01-01":"2026-03-20"].min()
trough_d = w.loc["2026-01-01":"2026-03-20"].idxmin()
print(f"\nEW index 2026: peak {peak:.3f} @ {peak_d.date()} -> trough {trough:.3f} @ {trough_d.date()} "
      f"({(trough/peak-1)*100:.1f}%)")
b26 = bull.loc["2026-01-01":"2026-03-31"]
flips = b26[b26 != b26.shift(1)]
print("bull-mask state changes Jan-Mar 2026:")
prev = b26.iloc[0]
print(f"  state at 2026-01-01: {prev}")
for d, v in flips.items():
    print(f"  {d.date()}: bull={v}  (EW {ew[d]:.3f} vs MA60 {ma60[d]:.3f})")
# drawdown from Jan peak to the day bull mask turned off
off_days = b26[~b26].index
if len(off_days):
    first_off = off_days[0]
    print(f"first non-bull day 2026: {first_off.date()}  EW decline from peak by then: "
          f"{(ew[first_off]/peak-1)*100:.1f}%")

# (c) 2022 added big losses vs washout days
rets = piv.pct_change()
mret = rets.mean(axis=1)
roll = mret.rolling(5).sum()
z = (roll - roll.rolling(60).mean()) / (roll.rolling(60).std() + 1e-9)
drops = pd.DatetimeIndex(z.index[z <= -1.75])
bl22 = stack[(stack.pnl_pct <= -0.10) & (stack.entry_date.dt.year == 2022) & ~stack.k.isin(set(champ.k))]
print("\n2022 added big losses: entry vs nearest washout day after entry, exit lag")
for r in bl22.itertuples():
    dd = drops[(drops >= r.entry_date)]
    nxt = dd[0] if len(dd) else None
    lag = (r.exit_date - nxt).days if nxt is not None else None
    print(f"  {r.symbol} {r.entry_date.date()}->{r.exit_date.date()} {r.pnl_pct:+.3f} "
          f"first washout after entry {nxt.date() if nxt is not None else '-'} exit-lag {lag}d "
          f"bull@entry={bool(bull.get(r.entry_date, False))}")
print("\nDONE2")
