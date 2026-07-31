# -*- coding: utf-8 -*-
"""Audit finding #10: capacity/liquidity realism on the registered gtrail run.
Measures on REAL run data:
  A) touch-depth of fills: (entry_price_limit - fill_day_low)/entry_price -> marginal-touch fractions
  B) ADV20 (VND) of the symbol at entry -> illiquidity quantiles
  C) position weights held (run_portfolio_daily.weight) -> concentration
  D) preempt evictions: count + assumed same-day full-size close-price sale
"""
import psycopg2, duckdb, pandas as pd, numpy as np

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
RID = "template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138"
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"

con = psycopg2.connect(**PG)
tr = pd.read_sql("""select symbol, entry_date, entry_price, exit_date, exit_price,
                    exit_reason, pnl_pct, entry_signal_date, holding_days
                    from run_trades where run_id=%s""", con, params=(RID,))
pf = pd.read_sql("select date, symbol, weight, entry_weight from run_portfolio_daily where run_id=%s and weight>0",
                 con, params=(RID,))
print(f"trades={len(tr)}  portfolio_rows={len(pf)}  exit_reasons={tr.exit_reason.value_counts().to_dict()}")

cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("""select symbol, date, open, low, close, volume from ohlcv
                   where timeframe='1D' and date>='2018-06-01' order by symbol, date""").fetchdf()
cx.close()
px["date"] = pd.to_datetime(px["date"])
# ADV20 in native price units * shares; VN prices in DuckDB usually thousand-VND
px["turnover"] = px["close"] * px["volume"]
px["adv20"] = px.groupby("symbol")["turnover"].transform(lambda s: s.rolling(20).mean())

pxi = px.set_index(["symbol", "date"])
tr["entry_date"] = pd.to_datetime(tr["entry_date"])
tr["entry_signal_date"] = pd.to_datetime(tr["entry_signal_date"])

rows = []
for r in tr.itertuples():
    key = (r.symbol, r.entry_date)
    if key not in pxi.index:
        continue
    bar = pxi.loc[key]
    low, close, adv, vol = float(bar["low"]), float(bar["close"]), float(bar["adv20"]), float(bar["volume"])
    # signal close for limit check
    sig_close = np.nan
    if pd.notna(r.entry_signal_date) and (r.symbol, r.entry_signal_date) in pxi.index:
        sig_close = float(pxi.loc[(r.symbol, r.entry_signal_date)]["close"])
    depth = (r.entry_price - low) / r.entry_price if r.entry_price else np.nan
    is_limit = (pd.notna(sig_close) and abs(r.entry_price / (sig_close * 0.955) - 1.0) < 0.002)
    rows.append(dict(symbol=r.symbol, entry_date=r.entry_date, entry_price=r.entry_price,
                     low=low, depth=depth, adv20=adv, day_turnover=close * vol,
                     is_limit_fill=is_limit, pnl=r.pnl_pct, reason=r.exit_reason))
d = pd.DataFrame(rows)
print(f"\nmatched entry bars: {len(d)}/{len(tr)}   limit-priced fills (entry=sig_close*0.955): {d.is_limit_fill.sum()}")

lim = d[d.is_limit_fill]
print("\n--- A) touch depth (limit - day_low)/limit on limit fills ---")
for thr in (0.001, 0.003, 0.005, 0.01):
    frac = (lim.depth < thr).mean()
    print(f"  depth < {thr*100:.1f}%: {frac*100:.1f}%  (n={int((lim.depth<thr).sum())})")
print(f"  exact-touch depth==0: {(lim.depth <= 1e-9).mean()*100:.1f}%")
print(f"  PnL of marginal (<0.3%) fills: mean {lim[lim.depth<0.003].pnl.mean()*100:.2f}% vs rest {lim[lim.depth>=0.003].pnl.mean()*100:.2f}%  n={int((lim.depth<0.003).sum())}")

print("\n--- B) ADV20 turnover (price-unit x shares; thousand-VND prices => value ~ x1000 VND) ---")
q = d.adv20.quantile([0.01, 0.05, 0.10, 0.25, 0.50])
print(q.to_string())
print(f"  min ADV20 = {d.adv20.min():,.0f}  ({d.loc[d.adv20.idxmin(),'symbol']} {d.loc[d.adv20.idxmin(),'entry_date'].date()})")
print(f"  entries with ADV20 < 5e6 (≈5B VND/day if prices thousand-VND): {(d.adv20<5e6).sum()} ({(d.adv20<5e6).mean()*100:.1f}%)")
print(f"  entries with ADV20 < 1e6 (≈1B VND/day): {(d.adv20<1e6).sum()} ({(d.adv20<1e6).mean()*100:.1f}%)")

print("\n--- C) position weights (run_portfolio_daily) ---")
print(f"  rows={len(pf)}  max weight={pf.weight.max():.3f} ({pf.loc[pf.weight.idxmax(),'symbol']} {pf.loc[pf.weight.idxmax(),'date']})")
for thr in (0.12, 0.15, 0.18, 0.20):
    print(f"  holding-days weight>{thr:.2f}: {(pf.weight>thr).sum()} ({(pf.weight>thr).mean()*100:.1f}%)")
print(f"  max entry_weight = {pf.entry_weight.max():.3f}; entry_weight>0.15: {(pf.entry_weight>0.15).sum()} rows")

# participation ratio at a modest 5B-VND book: pos = w*5e9 VND; ADV in VND = adv20*1000
d2 = d.merge(pf.groupby(["symbol"]).weight.max().rename("wmax"), on="symbol", how="left")
d["adv_vnd"] = d.adv20 * 1000.0
med_w = pf.entry_weight[pf.entry_weight > 0].median()
d["pos_5B"] = 5e9 * med_w
d["participation_5B"] = d["pos_5B"] / d["adv_vnd"]
print(f"\n--- capacity at 5B-VND AUM (median entry weight {med_w:.3f}) ---")
print(f"  entries where position > 20% of ADV20: {(d.participation_5B>0.2).sum()} ({(d.participation_5B>0.2).mean()*100:.1f}%)")
print(f"  entries where position > 100% of ADV20: {(d.participation_5B>1.0).sum()} ({(d.participation_5B>1.0).mean()*100:.1f}%)")
print(f"  worst participation: {d.participation_5B.max():.1f}x ADV ({d.loc[d.participation_5B.idxmax(),'symbol']} {d.loc[d.participation_5B.idxmax(),'entry_date'].date()})")

print("\n--- D) preempt evictions ---")
pe = tr[tr.exit_reason == "preempt"]
print(f"  preempt trades: {len(pe)} ({len(pe)/len(tr)*100:.1f}%)")

# contribution of marginal fills to total pnl
tot = d.pnl.sum(); marg = lim[lim.depth < 0.003].pnl.sum()
print(f"\n--- edge concentration --- sum pnl_pct all={tot:.2f}, marginal(<0.3%) limit fills={marg:.2f} ({marg/tot*100:.1f}%)")
con.close()
