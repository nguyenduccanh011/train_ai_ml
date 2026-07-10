# -*- coding: utf-8 -*-
"""pr2 (cong to) toi 2: NAV sim khung chuan (BASE mode dl_01_dual_sim, K25, 100% NAV).

Sleeve A = trades CSV, size = NAV/K khi du cash, leg neo 2 dau entry_price/pnl CSV
(mark-to-market daily bang close db, ratio noi suy). Them stats slot-doi:
avg open legs, % ngay legs<K va cash idle, idle cash fraction trung binh.

Usage: python pr2_04_navsim.py <csv> <label> [K]
"""
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd

BASE_DIR = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
DB_PATH = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
DATE_LO, DATE_HI = "2020-01-01", "2026-07-08"

CSV = sys.argv[1]
LABEL = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 25

ta = pd.read_csv(CSV)
symbols = sorted(set(ta.symbol))
con = sqlite3.connect(DB_PATH)
px = pd.read_sql_query(
    "SELECT symbol, date, close FROM ohlcv WHERE symbol IN (%s) AND date>=? AND date<=?"
    % ",".join("?" * len(symbols)), con, params=symbols + [DATE_LO, DATE_HI])
con.close()

sym_close, sym_idx, sym_dates = {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date")
    sym_dates[s] = g["date"].tolist()
    sym_close[s] = g["close"].tolist()
    sym_idx[s] = {d: i for i, d in enumerate(sym_dates[s])}
calendar = sorted(set(px["date"]))

trades, skipped = [], 0
for r in ta.itertuples():
    s = r.symbol
    ed = str(r.entry_date)[:10]
    xd = str(r.exit_date)[:10]
    if ed not in sym_idx.get(s, {}) or xd not in sym_idx.get(s, {}):
        skipped += 1
        continue
    trades.append(dict(symbol=s, entry_date=ed, exit_date=xd,
                       i0=sym_idx[s][ed], i1=sym_idx[s][xd],
                       p0=float(r.entry_price), net=float(r.pnl_pct)))
print(f"{LABEL}: {len(trades)} trades usable (skip {skipped})")

entries = defaultdict(list)
for t in trades:
    entries[t["entry_date"]].append(t)
for d in entries:
    entries[d].sort(key=lambda t: t["symbol"])


def dparse(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def leg_value(leg, dt):
    s = leg["symbol"]
    j = sym_idx[s].get(dt)
    if j is None:
        return leg["last_val"]
    i0, i1 = leg["i0"], leg["i1"]
    j = min(max(j, i0), i1)
    ratio = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
    v = leg["invested"] * (sym_close[s][j] * ratio) / leg["p0"]
    leg["last_val"] = v
    return v


cash = 1.0
legs = []
exits = defaultdict(list)
skip_cash = fills = 0
nav_series = []
occ_days = idle_frac_sum = 0
occ_hist = defaultdict(int)

for dt in calendar:
    for leg in exits.get(dt, ()):
        cash += leg["invested"] * (1.0 + leg["net"])
        legs.remove(leg)
    pos = sum(leg_value(l, dt) for l in legs)
    nav_now = cash + pos
    for t in entries.get(dt, ()):
        size = nav_now / K
        if cash + 1e-12 >= size:
            s = t["symbol"]
            c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
            exit_eff = t["p0"] * (1.0 + t["net"])
            leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"],
                       p0=t["p0"], ratio0=t["p0"] / c0, ratio1=exit_eff / c1, last_val=size)
            cash -= size
            legs.append(leg)
            exits[t["exit_date"]].append(leg)
            fills += 1
        else:
            skip_cash += 1
    pos = sum(leg_value(l, dt) for l in legs)
    nav = cash + pos
    nav_series.append((dt, nav, len(legs), cash / nav))
    occ_hist[len(legs)] += 1
    occ_days += len(legs)
    idle_frac_sum += cash / nav

ns = pd.DataFrame(nav_series, columns=["date", "nav", "n_open", "cash_frac"])
ns["date"] = pd.to_datetime(ns["date"])
nav = ns["nav"]
final = nav.iloc[-1]
years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
cagr = final ** (1 / years) - 1
dd = nav / nav.cummax() - 1
maxdd = dd.min()
maxdd_date = ns["date"].iloc[dd.idxmin()].date()
# thoi gian duoi dinh dai nhat
under = (dd < -1e-9).astype(int)
grp = (under.diff() != 0).cumsum()
runs = under.groupby(grp).agg(["sum", "size"])
longest_uw = int(runs[runs["sum"] > 0]["size"].max()) if (under > 0).any() else 0
yearly = {}
ns2 = ns.set_index("date")
for y, g in ns2.groupby(ns2.index.year):
    prev = ns2["nav"][ns2.index.year < y]
    start = prev.iloc[-1] if len(prev) else 1.0
    yearly[int(y)] = g["nav"].iloc[-1] / start - 1.0

nd = len(calendar)
print(f"{LABEL} K{K}: final={final:.2f} CAGR={cagr*100:.2f}% MaxDD={maxdd*100:.2f}% ({maxdd_date}) "
      f"longest_underwater={longest_uw}d fills={fills} skip_cash={skip_cash}")
print(f"  slot: avg_open={occ_days/nd:.1f}/{K} avg_idle_cash={idle_frac_sum/nd*100:.1f}% "
      f"days_full(>={K-1})={sum(v for k, v in occ_hist.items() if k >= K-1)/nd*100:.0f}% "
      f"days_<10open={sum(v for k, v in occ_hist.items() if k < 10)/nd*100:.0f}%")
print("  yearly: " + " ".join(f"{y}:{v*100:+.1f}%" for y, v in sorted(yearly.items())))
ns.to_csv(BASE_DIR + f"/pureml/pr2_nav_{LABEL}.csv", index=False)
