"""Regime checks: (1) SNR at the delayed exit (exit_s) — does the extension end into a cooling tape?
(2) worse-group portrait vs better-group: year x SNR x liquidity terciles. (3) +/-1 bar SNR robustness."""
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
    "SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) ORDER BY date"
    .format(",".join(f"'{s}'" for s in uni_syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
snr = rets.mean(axis=1).rolling(20).sum() / (rets.rolling(20).sum().std(axis=1) + 1e-9)

idx = snr.index
def at(d, off=0):
    i = idx.searchsorted(pd.Timestamp(d))
    i = min(max(i + off, 0), len(idx) - 1)
    return float(snr.iloc[i])

D["snr_exit_s"] = D.exit_s.map(lambda d: at(d))
D["snr_exit_s_m1"] = D.exit_s.map(lambda d: at(d, -1))
D["snr_defer_m1"] = D.exit_c.map(lambda d: at(d, -1))
D["snr_defer_m2"] = D.exit_c.map(lambda d: at(d, -2))

print("SNR at delayed exit (exit_s): median", round(D.snr_exit_s.median(), 3),
      "| pct < 0.8:", round((D.snr_exit_s < 0.8).mean() * 100, 1), "%")
print("SNR at defer bar (-1):", round(D.snr_defer_m1.median(), 3),
      "| pct >= 0.8 at any of {exit_c, -1, -2}:",
      round(((D.snr_at_defer >= 0.8) | (D.snr_defer_m1 >= 0.8) | (D.snr_defer_m2 >= 0.8)).mean() * 100, 1), "%")
print("SNR drop defer->exit_s: median", round((D.snr_exit_s - D.snr_at_defer).median(), 3))

D["grp"] = np.where(D.dpnl < -1e-9, "worse", np.where(D.dpnl > 1e-9, "better", "flat"))
print("\n== worse-group concentration ==")
print(pd.crosstab(D.year_entry, D.grp))
D["liq_ter"] = pd.qcut(D.tval60, 3, labels=["low", "mid", "high"])
print(pd.crosstab(D.liq_ter, D.grp))
print("\nsum dpnl by liquidity tercile:")
print(D.groupby("liq_ter", observed=True).dpnl.agg(["sum", "count"]).round(3))
D["snrter"] = pd.qcut(D.snr_at_defer, 3, labels=["snr_lo", "snr_mid", "snr_hi"])
print("\nsum dpnl by SNR-at-defer tercile:")
print(D.groupby("snrter", observed=True).dpnl.agg(["sum", "count"]).round(3))
print("\nsum dpnl by dhold bucket:")
D["holdb"] = pd.cut(D.dhold, [0, 3, 10, 30, 200], labels=["1-3", "4-10", "11-30", ">30"])
print(D.groupby("holdb", observed=True).dpnl.agg(["sum", "count"]).round(3))
print("\nmega-runner dependence: sum dpnl excl top2 =", round(D.dpnl.sum() - D.dpnl.nlargest(2).sum(), 3))

# was the worse group identifiable at defer time? logistic-style split checks
print("\n== ex-ante separators (worse rate | sum dpnl) ==")
for name, mask in [
    ("gain_at_defer>=0.45", D.gain_at_defer >= 0.45),
    ("giveback_at_defer<=0.10", D.giveback_at_defer <= 0.10),
    ("sym_snr>=1.0", D.sym_snr_at_defer >= 1.0),
    ("snr>=1.1", D.snr_at_defer >= 1.1),
    ("tval60<40e6", D.tval60 < 4e7),
    ("combo: snr>=1.1 & gain>=0.45", (D.snr_at_defer >= 1.1) & (D.gain_at_defer >= 0.45)),
]:
    sub, oth = D[mask], D[~mask]
    if len(sub) == 0: continue
    print(f"{name:32s}: n={len(sub):2d} worse%={(sub.dpnl<0).mean()*100:4.1f} dpnl={sub.dpnl.sum():+.3f} | "
          f"rest n={len(oth):2d} worse%={(oth.dpnl<0).mean()*100:4.1f} dpnl={oth.dpnl.sum():+.3f}")
