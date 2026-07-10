"""EXIT MAP gb_x08 — washout-release cohort: exits right after the market-drop gate lifts.
Exact engine reconstruction of market_drop_dates (zscore mode, w=5, lb=60, thr=-1.75, no clip).
Doubles as a validation of decision-bar alignment: signal-exits ON a drop date should be ~0.
"""
import json
import duckdb
import numpy as np
import pandas as pd
import psycopg2

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
E = pd.read_csv(EM + "/gbx08_enriched2.csv", parse_dates=["entry_date", "exit_date"])
pd.set_option("display.width", 220)

pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in uni_syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
mret = piv.pct_change(fill_method=None).mean(axis=1)
roll = mret.rolling(5).sum()
z = (roll - roll.rolling(60).mean()) / (roll.rolling(60).std() + 1e-9)
drop = (z <= -1.75)
drop_dates = set(drop.index[drop])
print("drop dates:", len(drop_dates))

cal = piv.index  # trading calendar
pos = {d: i for i, d in enumerate(cal)}
drop_arr = np.zeros(len(cal), dtype=bool)
for d in drop_dates:
    drop_arr[pos[d]] = True
# bars since last drop date (inf if never)
last = -10**9
since = np.empty(len(cal), dtype=np.int64)
for i in range(len(cal)):
    if drop_arr[i]:
        last = i
    since[i] = i - last

# decision bar = trading bar before exit_date
E = E[E.exit_reason == "signal"].copy()
dec_since = []
on_drop = []
for d in E.exit_date:
    i = pos.get(d)
    if i is None or i == 0:
        dec_since.append(np.nan); on_drop.append(np.nan); continue
    dec_since.append(since[i - 1])
    on_drop.append(bool(drop_arr[i - 1]))
E["since_drop"] = dec_since
E["on_drop"] = on_drop

print("\nVALIDATION: signal-exits with decision bar ON a drop date (engine suppresses -> expect ~0):")
print("  n =", int(pd.Series(on_drop).fillna(False).sum()), "of", len(E))

print("\nWASHOUT-RELEASE cohort: exits k bars after the gate lifts")
E["rallied"] = E.post_max_c >= 0.05
for lo, hi in [(1, 1), (2, 3), (4, 6), (7, 15), (16, 60), (61, 10**8)]:
    C = E[(E.since_drop >= lo) & (E.since_drop <= hi)]
    if len(C) == 0:
        continue
    cf = (C.post_end_c * (1 + C.pnl_pct)).sum()
    print(f"  since_drop {lo:>3}-{hi if hi<10**8 else 'inf':>3}: n={len(C):4d} pnl={C.pnl_pct.sum():+7.1f} "
          f"P(rallied5)={C.rallied.mean():.3f} cf_end20={cf:+6.1f}u cf_peak={(C.post_max_c*(1+C.pnl_pct)).sum():+6.1f}u")

C = E[(E.since_drop >= 1) & (E.since_drop <= 3)]
print("\nrelease 1-3 bars cohort by rule:")
print(C.groupby("label").apply(lambda g: pd.Series(dict(
    n=len(g), pnl=g.pnl_pct.sum(), cf_end20_u=(g.post_end_c * (1 + g.pnl_pct)).sum(),
    rallied=g.rallied.mean()))).round(3).to_string())
print("\nrelease 1-3 bars by exit year (cf_end20):")
print(C.groupby("year_exit").apply(lambda g: pd.Series(dict(
    n=len(g), pnl=g.pnl_pct.sum(), cf_end20_u=(g.post_end_c * (1 + g.pnl_pct)).sum()))).round(2).to_string())
print("\nrelease 1-3, only in-profit (gain_d>0) trades:")
P_ = C[C.gain_d > 0]
print(f"  n={len(P_)} pnl={P_.pnl_pct.sum():+.1f} cf_end20={(P_.post_end_c*(1+P_.pnl_pct)).sum():+.1f}u")
N_ = C[C.gain_d <= 0]
print(f"  losers at decision: n={len(N_)} pnl={N_.pnl_pct.sum():+.1f} cf_end20={(N_.post_end_c*(1+N_.pnl_pct)).sum():+.1f}u")
