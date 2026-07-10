# -*- coding: utf-8 -*-
"""Preflight: gb_x08 run_id/trades tu DB + xac minh convention exit 'signal' fill."""
import sqlite3
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"


# --- exit convention check: sell signal bar -> exit fill bar? ---
sig = pd.read_csv(f"{BASE}/signals.csv")
tr = pd.read_csv(f"{BASE}/trades_raw.csv")
oc = sqlite3.connect(DB)
sample = tr[tr.exit_reason == "signal"].sample(20, random_state=1)
ok_next, ok_same, other = 0, 0, 0
for r in sample.itertuples():
    g = pd.read_sql("select date, close from ohlcv where symbol=? order by date", oc, params=(r.symbol,))
    didx = {d: i for i, d in enumerate(g.date)}
    xi = didx.get(str(r.exit_date))
    if xi is None:
        continue
    ss = sig[(sig.symbol == r.symbol) & (sig.signal < 0)]
    sell_dates = set(ss.date)
    prev_bar = g.date.iloc[xi - 1]
    if prev_bar in sell_dates:
        ok_next += 1
    elif str(r.exit_date) in sell_dates:
        ok_same += 1
    else:
        other += 1
    # also check exit_price vs close
    print(r.symbol, r.exit_date, "exit_px", round(r.exit_price, 4),
          "close(exit)", g.close.iloc[xi], "close*0.9985", round(g.close.iloc[xi] * 0.9985, 4),
          "prev_is_sell", prev_bar in sell_dates, "same_is_sell", str(r.exit_date) in sell_dates)
print("next-bar-fill:", ok_next, "same-bar:", ok_same, "other:", other)

# entry convention: entry_price vs limit*1.0015
dep = pd.read_parquet(f"{BASE}/depths.parquet")
dmap = {(s, d): v for s, d, v in zip(dep.symbol, dep.date, dep.eff_depth)}
s2 = tr.sample(20, random_state=2)
for r in s2.itertuples():
    g = pd.read_sql("select date, close from ohlcv where symbol=? order by date", oc, params=(r.symbol,))
    didx = {d: i for i, d in enumerate(g.date)}
    si = didx.get(str(r.entry_signal_date))
    if si is None:
        continue
    d = dmap.get((r.symbol, str(r.entry_signal_date)))
    lim = g.close.iloc[si] * (1 - d) if d is not None else float("nan")
    print(r.symbol, r.entry_signal_date, "entry_px", round(r.entry_price, 4),
          "limit*1.0015", round(lim * 1.0015, 4), "depth", d)
oc.close()
