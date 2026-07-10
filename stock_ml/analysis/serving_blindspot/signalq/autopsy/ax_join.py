"""AUTOPSY exit_snr_extend (t2730 vs champion t2646, seed 42).
Join trades on (symbol, entry_date); dissect the deferred-exit group.
Sources (verified):
  champ  = st_champ2646_s42_trades.csv  (== DB run_trades champion seed42, pnl 127.253, n=1384)
  snr08  = sv_snr08_s42_trades.csv      (== t2730 seed42 per run_exit_family.log line 810, pnl 128.177, n=1376)
"""
import duckdb
import numpy as np
import pandas as pd
import psycopg2, json

SQ = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/signalq"
OUT = SQ + "/autopsy"

champ = pd.read_csv(SQ + "/st_champ2646_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
snr = pd.read_csv(SQ + "/sv_snr08_s42_trades.csv", parse_dates=["entry_date", "exit_date"])

# ---- universe & bars ----
pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()
print(f"universe: {len(uni_syms)} symbols")

duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, high, close, traded_value FROM ohlcv "
    "WHERE timeframe='1D' AND symbol IN ({}) ORDER BY date".format(
        ",".join(f"'{s}'" for s in uni_syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])

piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
W = 20
uni_trend = rets.mean(axis=1).rolling(W).sum()
disp = rets.rolling(W).sum().std(axis=1)
snr_series = uni_trend / (disp + 1e-9)          # exact engine formula
highs = bars.pivot_table(index="date", columns="symbol", values="high", aggfunc="last").sort_index()
tval = bars.pivot_table(index="date", columns="symbol", values="traded_value", aggfunc="last").sort_index()

# per-symbol SNR analog: rolling20 return / rolling20 std of daily ret * sqrt window (trend cleanliness)
sym_ret20 = rets.rolling(W).sum()
sym_vol20 = rets.rolling(W).std()
sym_snr = sym_ret20 / (sym_vol20 * np.sqrt(W) + 1e-9)

# ---- join ----
m = champ.merge(snr, on=["symbol", "entry_date"], how="outer", suffixes=("_c", "_s"), indicator=True)
print("\nmerge:", m._merge.value_counts().to_dict())
both = m[m._merge == "both"].copy()
same = both[both.exit_date_c == both.exit_date_s]
diff = both[both.exit_date_c != both.exit_date_s].copy()
print(f"matched pairs: {len(both)} | identical exits: {len(same)} | different exits (deferred): {len(diff)}")
print("entry price identical on matched:", (abs(both.entry_price_c - both.entry_price_s) < 1e-9).all())
print("same-exit pnl identical:", (abs(same.pnl_pct_c - same.pnl_pct_s) < 1e-9).all())

co = m[m._merge == "left_only"]   # champ-only entries (knock-on)
so = m[m._merge == "right_only"]  # snr-only entries
print(f"\nknock-on: champ-only n={len(co)} pnl={co.pnl_pct_c.sum():.3f} | "
      f"snr-only n={len(so)} pnl={so.pnl_pct_s.sum():.3f}")

diff["dpnl"] = diff.pnl_pct_s - diff.pnl_pct_c
diff["dhold"] = diff.holding_days_s - diff.holding_days_c
print(f"\ndeferred group: n={len(diff)}  sum dpnl={diff.dpnl.sum():.3f}")
print("exit_reason champ:", diff.exit_reason_c.value_counts().to_dict())
print("exit_reason snr  :", diff.exit_reason_s.value_counts().to_dict())

# decomposition of total delta
tot = snr.pnl_pct.sum() - champ.pnl_pct.sum()
print(f"\nTOTAL dpnl {tot:.3f} = deferred {diff.dpnl.sum():.3f} "
      f"+ knockon ({so.pnl_pct_s.sum():.3f} - {co.pnl_pct_c.sum():.3f})")

# ---- peak / giveback per deferred trade ----
rows = []
for _, r in diff.iterrows():
    hs = highs[r.symbol]
    seg_c = hs.loc[r.entry_date:r.exit_date_c]
    seg_s = hs.loc[r.entry_date:r.exit_date_s]
    ep = r.entry_price_c
    peak_c = seg_c.max() / ep - 1.0
    peak_s = seg_s.max() / ep - 1.0
    d_exit_c = r.exit_date_c
    # state at the moment the champion would have sold (deferral moment)
    snr_at = snr_series.asof(d_exit_c)
    ssnr_at = sym_snr[r.symbol].asof(d_exit_c) if r.symbol in sym_snr else np.nan
    close_at = piv[r.symbol].asof(d_exit_c)
    gain_at = close_at / ep - 1.0
    giveback_at = peak_c - gain_at                       # already off its peak when deferred
    tv = tval[r.symbol].loc[:r.entry_date].tail(60).median() if r.symbol in tval else np.nan
    rows.append(dict(symbol=r.symbol, entry_date=r.entry_date, exit_c=r.exit_date_c, exit_s=r.exit_date_s,
                     pnl_c=r.pnl_pct_c, pnl_s=r.pnl_pct_s, dpnl=r.dpnl, dhold=r.dhold,
                     reason_c=r.exit_reason_c, reason_s=r.exit_reason_s,
                     peak_c=peak_c, peak_s=peak_s,
                     giveback_final_s=peak_s - r.pnl_pct_s, giveback_final_c=peak_c - r.pnl_pct_c,
                     snr_at_defer=snr_at, sym_snr_at_defer=ssnr_at,
                     gain_at_defer=gain_at, giveback_at_defer=giveback_at,
                     tval60=tv, year_entry=r.entry_date.year, year_exit_c=r.exit_date_c.year))
D = pd.DataFrame(rows)
D.to_csv(OUT + "/deferred_trades.csv", index=False)

print("\n== dpnl distribution (deferred) ==")
print(D.dpnl.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(4))
worse = D[D.dpnl < -1e-9]
better = D[D.dpnl > 1e-9]
print(f"worse: {len(worse)} ({len(worse)/len(D)*100:.1f}%) sum={worse.dpnl.sum():.3f}")
print(f"better: {len(better)} ({len(better)/len(D)*100:.1f}%) sum={better.dpnl.sum():.3f}")
print("\nworst 12:")
cols = ["symbol", "entry_date", "exit_c", "exit_s", "pnl_c", "pnl_s", "dpnl", "dhold",
        "reason_s", "peak_s", "giveback_at_defer", "snr_at_defer", "sym_snr_at_defer"]
print(D.nsmallest(12, "dpnl")[cols].to_string(index=False))
print("\nbest 8:")
print(D.nlargest(8, "dpnl")[cols].to_string(index=False))

print("\n== by entry year ==")
g = D.groupby("year_entry").agg(n=("dpnl", "size"), worse=("dpnl", lambda x: (x < 0).sum()),
                                sum_dpnl=("dpnl", "sum"), mean_dhold=("dhold", "mean"))
print(g.round(3))
print("\n== by exit_c year ==")
g2 = D.groupby("year_exit_c").agg(n=("dpnl", "size"), worse=("dpnl", lambda x: (x < 0).sum()),
                                  sum_dpnl=("dpnl", "sum"))
print(g2.round(3))

print("\n== worse vs better group features (median) ==")
feat = ["dhold", "peak_c", "gain_at_defer", "giveback_at_defer", "snr_at_defer",
        "sym_snr_at_defer", "tval60", "pnl_c"]
cmp = pd.DataFrame({"worse_med": worse[feat].median(), "better_med": better[feat].median(),
                    "worse_mean": worse[feat].mean(), "better_mean": better[feat].mean()})
print(cmp.round(4))
