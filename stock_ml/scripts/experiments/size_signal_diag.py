"""Which entry-time signal predicts per-trade RETURN MAGNITUDE (the thing sizing needs)? Entry score
IC~0 vs realized pnl (why entry-score sizing was null). Scan candidate per-symbol signals computable
from OHLCV at each frontier trade's entry, correlate (Spearman) with realized pnl_pct. The signal with
the highest |IC| is the sizing conviction signal; if none has meaningful IC, per-trade magnitude is
unpredictable and fair-sizing is hopeless.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd, duckdb, psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
FRONT = "template/x2_struct_to-69338138"

con = psycopg2.connect(**PG)
tr = pd.read_sql(
    "SELECT symbol, entry_signal_date, pnl_pct FROM run_trades WHERE run_id=%s "
    "AND pnl_pct IS NOT NULL AND entry_signal_date IS NOT NULL",
    con,
    params=(FRONT,),
)
sg = pd.read_sql(
    "SELECT symbol, date, score FROM run_signals WHERE run_id=%s", con, params=(FRONT,)
)
con.close()
tr["d"] = pd.to_datetime(tr["entry_signal_date"])
sg["d"] = pd.to_datetime(sg["date"])
tr = tr.merge(sg[["symbol", "d", "score"]], on=["symbol", "d"], how="left")
print(f"{len(tr)} frontier trades")

# compute per-symbol OHLCV features, sample at entry dates
cx = duckdb.connect(MARKET, read_only=True)
syms = sorted(tr["symbol"].unique())
inl = ",".join(repr(s) for s in syms)
px = cx.execute(
    f"SELECT symbol,date,open,high,low,close,volume FROM ohlcv WHERE timeframe='1D' "
    f"AND symbol IN ({inl}) AND date>='2018-06-01' ORDER BY symbol,date"
).fetchdf()
cx.close()
px["date"] = pd.to_datetime(px["date"])
feats = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy()
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    ma10, ma20, ma50, ma100 = (
        c.rolling(10).mean(),
        c.rolling(20).mean(),
        c.rolling(50).mean(),
        c.rolling(100).mean(),
    )
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    g["dist_ma10"] = c / ma10 - 1
    g["dist_ma20"] = c / ma20 - 1
    g["dist_ma50"] = c / ma50 - 1
    g["dist_ma100"] = c / ma100 - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["ret5"] = c / c.shift(5) - 1
    g["ret20"] = c / c.shift(20) - 1
    g["rvol20"] = c.pct_change().rolling(20).std()
    g["dist20low"] = c / l.rolling(20).min() - 1
    g["dist63high"] = c / h.rolling(63).max() - 1
    g["volz"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    g["ma20_slope"] = ma20 / ma20.shift(10) - 1
    feats.append(g)
F = pd.concat(feats, ignore_index=True)
FCOLS = [
    "dist_ma10",
    "dist_ma20",
    "dist_ma50",
    "dist_ma100",
    "rsi14",
    "ret5",
    "ret20",
    "rvol20",
    "dist20low",
    "dist63high",
    "volz",
    "ma20_slope",
]
m = tr.merge(
    F[["symbol", "date"] + FCOLS], left_on=["symbol", "d"], right_on=["symbol", "date"], how="left"
)

print(
    f"\n=== IC (Spearman) of entry-time signal vs realized pnl_pct — n={m['pnl_pct'].notna().sum()} ==="
)
print("signal        | IC     | |IC| rank")
ics = []
for col in ["score"] + FCOLS:
    sub = m[[col, "pnl_pct"]].dropna()
    if len(sub) < 100:
        continue
    ic = sub[col].corr(sub["pnl_pct"], method="spearman")
    ics.append((col, ic))
for col, ic in sorted(ics, key=lambda x: -abs(x[1])):
    print(f"{col:13s} | {ic:+.4f}")
print("SIZESIG_DONE")
