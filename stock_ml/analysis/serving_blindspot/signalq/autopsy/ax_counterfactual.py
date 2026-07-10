"""Counterfactual screens on the 51 deferred trades (seed 42) before spending leaderboard runs.
1. min_gain X: trade only deferred if peak-at-defer >= X  -> dpnl kept = sum dpnl | peak_c >= X
2. window W: recompute SNR(W); deferral requires snr_W(exit_c) >= 0.8
3. giveback cap during deferral: exit when close drops >= X from running peak (uses duckdb closes)
"""
import duckdb
import numpy as np
import pandas as pd
import psycopg2, json

OUT = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/signalq/autopsy"
D = pd.read_csv(OUT + "/deferred_trades.csv", parse_dates=["entry_date", "exit_c", "exit_s"])

pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, high, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) ORDER BY date"
    .format(",".join(f"'{s}'" for s in uni_syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
highs = bars.pivot_table(index="date", columns="symbol", values="high", aggfunc="last").sort_index()
rets = piv.pct_change(fill_method=None)

def snr_series(w):
    return (rets.mean(axis=1).rolling(w).sum() / (rets.rolling(w).sum().std(axis=1) + 1e-9))

print("== 1. min_gain screen (gate: peak at defer >= X) ==")
for x in [0.27, 0.35, 0.40, 0.45, 0.50, 0.60]:
    keep = D[D.peak_c >= x]
    drop = D[D.peak_c < x]
    print(f"min_gain {x:.2f}: kept n={len(keep)} dpnl={keep.dpnl.sum():+.3f} | "
          f"dropped n={len(drop)} dpnl_removed={drop.dpnl.sum():+.3f} "
          f"(worse removed {len(drop[drop.dpnl<0])}, better removed {len(drop[drop.dpnl>0])})")

print("\n== 2. window screen (gate: snr_W at defer >= 0.8) ==")
for w in [10, 20, 30, 40, 60]:
    s = snr_series(w)
    gate = D.apply(lambda r: s.asof(r.exit_c) >= 0.8, axis=1)
    keep, drop = D[gate], D[~gate]
    print(f"window {w}: kept n={len(keep)} dpnl={keep.dpnl.sum():+.3f} | removed dpnl={drop.dpnl.sum():+.3f} "
          f"(worse removed {len(drop[drop.dpnl<0])}, better removed {len(drop[drop.dpnl>0])})")

print("\n== 3. threshold screen (gate: snr_20 at defer >= T) ==")
s20 = snr_series(20)
for t in [0.8, 0.9, 1.0, 1.1]:
    gate = D.apply(lambda r: s20.asof(r.exit_c) >= t, axis=1)
    keep, drop = D[gate], D[~gate]
    print(f"thr {t:.1f}: kept n={len(keep)} dpnl={keep.dpnl.sum():+.3f} | removed dpnl={drop.dpnl.sum():+.3f}")

print("\n== 3b. SNR upper band (gate: T_lo <= snr < T_hi — defer only in moderate SNR) ==")
for hi in [1.1, 1.3, 1.5]:
    gate = D.apply(lambda r: 0.8 <= s20.asof(r.exit_c) < hi, axis=1)
    keep, drop = D[gate], D[~gate]
    print(f"band [0.8,{hi}): kept n={len(keep)} dpnl={keep.dpnl.sum():+.3f} | removed dpnl={drop.dpnl.sum():+.3f}")

print("\n== 4. giveback cap during deferral (exit at close when drop-from-running-peak >= X) ==")
for cap in [0.05, 0.08, 0.10, 0.12, 0.15]:
    tot = 0.0; fired = 0; saved_on_worse = 0.0; cost_on_better = 0.0
    for _, r in D.iterrows():
        h = highs[r.symbol]; c = piv[r.symbol]
        # running peak from entry; walk deferral window (exit_c exclusive .. exit_s)
        idx = c.loc[r.exit_c:r.exit_s].index
        newp = r.pnl_s
        c_exit_s = c.asof(r.exit_s)
        for d in idx[1:]:
            pk = h.loc[r.entry_date:d].max()
            if 1.0 - c.loc[d] / pk >= cap:
                # exit here instead of exit_s: scale final pnl back by the close ratio
                newp = (1.0 + r.pnl_s) * (c.loc[d] / c_exit_s) - 1.0
                fired += 1
                break
        dd = newp - r.pnl_s
        if r.dpnl < 0: saved_on_worse += dd
        else: cost_on_better += dd
        tot += newp - r.pnl_c
    print(f"cap {cap:.2f}: fired={fired}/51 new_total_dpnl={tot:+.3f} (vs {D.dpnl.sum():+.3f}) "
          f"delta_on_worse={saved_on_worse:+.3f} delta_on_better={cost_on_better:+.3f}")
