"""Week-0 Task 3b: SNR histogram over champion universe (vn_stock_default, 61 symbols).
Replicates engine._market_snr_dates statistic (engine.py:2487-2500):
  snr = rolling(W).sum of universe-mean daily returns / cross-sectional std of per-symbol W-bar returns
W = exit_snr_extend_window = 20 (champion-relevant default).
"""
import duckdb
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DB = "f:/PROJECTS/train_ai_ml/market_data/market.duckdb"
W = 20

con = psycopg2.connect(**PG)
cur = con.cursor()
cur.execute("""SELECT us.symbol FROM universe_symbols us
               JOIN universe_sets s ON us.universe_id = s.id
               WHERE s.slug='vn_stock_default' ORDER BY us.symbol""")
syms = [r[0] for r in cur.fetchall()]
con.close()
print(f"universe vn_stock_default: {len(syms)} symbols")

d = duckdb.connect(DB, read_only=True)
ph = ",".join("?" * len(syms))
bars = d.execute(
    f"SELECT symbol, date, close FROM ohlcv WHERE symbol IN ({ph}) AND timeframe='1D' ORDER BY date",
    syms,
).df()
bars["date"] = pd.to_datetime(bars["date"])
d.close()
print(f"bars: {len(bars)} rows, dates {bars['date'].min()} .. {bars['date'].max()}")

piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
uni = rets.mean(axis=1).rolling(W).sum()
disp = rets.rolling(W).sum().std(axis=1)
snr = (uni / (disp + 1e-9)).dropna()

print(f"\nSNR (window={W}) over {len(snr)} dates:")
qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
for q in qs:
    print(f"  q{int(q*100):02d} = {snr.quantile(q):+.3f}")
print(f"  mean={snr.mean():+.3f} std={snr.std():.3f} min={snr.min():+.3f} max={snr.max():+.3f}")

print("\nfraction of dates with snr >= threshold (regime activation rate):")
for th in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
    frac = float((snr >= th).mean())
    print(f"  thr={th:>4}: {frac:6.1%}  ({int((snr >= th).sum())} dates)")

# histogram (bins of 0.25)
print("\nhistogram (bin width 0.25):")
import numpy as np
lo, hi = np.floor(snr.min() * 4) / 4, np.ceil(snr.max() * 4) / 4
bins = np.arange(lo, hi + 0.25, 0.25)
cnt, edges = np.histogram(snr, bins=bins)
for c, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
    if c:
        print(f"  [{e0:+.2f},{e1:+.2f}): {c:5d} {'#' * max(1, c // 20)}")

# last-2-year activation (recency check)
recent = snr[snr.index >= snr.index.max() - pd.Timedelta(days=730)]
print(f"\nlast-2yr ({len(recent)} dates) activation:")
for th in [0.7, 1.0, 1.2, 1.5]:
    print(f"  thr={th}: {float((recent >= th).mean()):6.1%}")
print("SNR_HIST_DONE")
