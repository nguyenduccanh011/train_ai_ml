"""CHEAP EV probe before any engine change: does entry-conviction (the sizing blend) predict
LEFT-ON-TABLE return AFTER exit? For each frontier trade, look up the price H bars after exit_date
and compute post-exit forward return. If high-conviction trades keep running post-exit (positive
fwd return, IC>0 vs conviction), then conviction-conditional hold-EXTENSION has EV -> worth an engine
change. If fwd return ~0 or negative for high-conviction, extension is dead -> pivot. Also split by
exit_reason so we see WHICH exit rule is prematurely cutting the conviction winners.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd, duckdb, psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
FRONT = "template/x2_struct_to-69338138"
BLEND = ["dist20low", "dist_ma20", "rsi14", "ret20"]
HORIZONS = [3, 5, 10, 20]

con = psycopg2.connect(**PG)
tr = pd.read_sql(
    "SELECT symbol, entry_signal_date, exit_date, exit_reason, pnl_pct "
    "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL "
    "AND entry_signal_date IS NOT NULL",
    con,
    params=(FRONT,),
)
con.close()
tr["sigd"] = pd.to_datetime(tr["entry_signal_date"])
tr["exd"] = pd.to_datetime(tr["exit_date"])
print(f"{len(tr)} frontier trades; exit_reason:\n{tr['exit_reason'].value_counts()}")

# per-symbol OHLCV -> conviction blend at signal date + forward return after exit
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute(
    "SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
    "ORDER BY symbol,date"
).fetchdf()
cx.close()
px["date"] = pd.to_datetime(px["date"])
parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    c, l = g["close"], g["low"]
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1
    g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["ret20"] = c / c.shift(20) - 1
    g["bar"] = np.arange(len(g))
    parts.append(g)
P = pd.concat(parts, ignore_index=True)
for col in BLEND:
    P[col + "_z"] = (P[col] - P[col].mean()) / (P[col].std() + 1e-9)
P["blend"] = P[[c + "_z" for c in BLEND]].mean(axis=1)

# map (symbol,date)-> row bar index and close for forward lookups
key = {(r.symbol, r.date): (int(r.bar), float(r.close)) for r in P.itertuples()}
sym_close = {s: g.sort_values("bar")["close"].to_numpy() for s, g in P.groupby("symbol")}

# conviction at signal date
conv_lk = {
    (r.symbol, r.date): (float(r.blend) if pd.notna(r.blend) else np.nan) for r in P.itertuples()
}
tr["conv"] = [conv_lk.get((s, d), np.nan) for s, d in zip(tr["symbol"], tr["sigd"])]


# forward return after exit_date over H bars
def fwd(sym, exd, H):
    kd = key.get((sym, exd))
    if kd is None:
        return np.nan
    b0, p0 = kd
    arr = sym_close.get(sym)
    if arr is None or b0 + H >= len(arr) or p0 <= 0:
        return np.nan
    return arr[b0 + H] / p0 - 1.0


for H in HORIZONS:
    tr[f"fwd{H}"] = [fwd(s, d, H) for s, d in zip(tr["symbol"], tr["exd"])]

m = tr.dropna(subset=["conv"])
print(f"\n=== post-exit forward return vs entry-conviction (n={len(m)}) ===")
print("horizon | mean_fwd | IC(conv,fwd) | hi-conv_tercile_fwd | lo-conv_tercile_fwd")
q = m["conv"].quantile([1 / 3, 2 / 3]).values
hi = m[m["conv"] >= q[1]]
lo = m[m["conv"] <= q[0]]
for H in HORIZONS:
    col = f"fwd{H}"
    sub = m[["conv", col]].dropna()
    ic = sub["conv"].corr(sub[col], method="spearman")
    print(
        f"  {H:3d}   | {m[col].mean() * 100:+6.2f}% | {ic:+.4f}      | "
        f"{hi[col].mean() * 100:+6.2f}%          | {lo[col].mean() * 100:+6.2f}%"
    )

print("\n=== same, restricted to trades cut by max_hold / signal (the extendable exits) ===")
for rule in ["max_hold", "signal", "trailing_struct"]:
    sub = m[m["exit_reason"] == rule]
    if len(sub) < 50:
        print(f"  {rule}: n={len(sub)} (skip)")
        continue
    qs = sub["conv"].quantile([1 / 3, 2 / 3]).values
    hh = sub[sub["conv"] >= qs[1]]
    ll = sub[sub["conv"] <= qs[0]]
    row = f"  {rule:15s} n={len(sub):4d}: "
    for H in [5, 10]:
        col = f"fwd{H}"
        row += f"fwd{H} hi={hh[col].mean() * 100:+5.2f}% lo={ll[col].mean() * 100:+5.2f}%  "
    print(row)
print("HOLDEXT_DIAG_DONE")
