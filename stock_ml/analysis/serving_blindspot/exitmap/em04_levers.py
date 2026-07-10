"""EXIT MAP gb_x08 — lever sizing: overlap, boundaries, counterfactuals with existing knobs."""
import json
import duckdb
import numpy as np
import pandas as pd
import psycopg2

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
E = pd.read_csv(EM + "/gbx08_enriched.csv", parse_dates=["entry_date", "exit_date"])
pd.set_option("display.width", 220)

print("0) overlap of force flags at decision bar (signal-labeled trades)")
S = E[E.label.str.startswith(("force", "head"))]
print(S.groupby(["f_dl12", "f_nb", "f_lb"]).size().to_string())

# ---- market bull mask (downleg_skip_bull sizing: ma_win 50 persist 3 on EW proxy) ----
pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, high, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in uni_syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
mret = piv.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
lvl = (1.0 + mret.fillna(0.0)).cumprod()
ma50 = lvl.rolling(50, min_periods=50).mean()
bull = ((lvl > ma50).rolling(3, min_periods=3).sum() >= 3).where(ma50.notna(), False)
peak20 = lvl.rolling(20, min_periods=1).max()
crash8 = (lvl / peak20 - 1.0) <= -0.08
bull_cb = bull & ~crash8

dec_dates = E.exit_date - pd.tseries.offsets.BDay(0)  # approx: decision = exit-1 trading bar
# more precisely use per-symbol prev bar; approximate with exit_date-1 calendar asof
E["bull_at"] = [bool(bull.asof(d - pd.Timedelta(days=1))) if not pd.isna(bull.asof(d - pd.Timedelta(days=1))) else False for d in E.exit_date]
E["bull_cb_at"] = [bool(bull_cb.asof(d - pd.Timedelta(days=1))) if not pd.isna(bull_cb.asof(d - pd.Timedelta(days=1))) else False for d in E.exit_date]

print("\n1) LEVER downleg_skip_bull: downleg12-labeled exits split by market-bull at decision")
D12 = E[E.label == "force_downleg12"].copy()
for m in ["bull_at", "bull_cb_at"]:
    g = D12.groupby(m).apply(lambda g: pd.Series(dict(
        n=len(g), pnl=g.pnl_pct.sum(),
        cf_end20_u=(g.post_end_c * (1 + g.pnl_pct)).sum(),
        cf_pos=( (g.post_end_c > 0).mean() ),
        rallied5=(g.post_max_c >= 0.05).mean())))
    print(f"\n  mask={m}")
    print(g.round(3).to_string())
print("\n  downleg12 in-bull cf_end20 by exit year:")
B = D12[D12.bull_cb_at]
print(B.groupby("year_exit").apply(lambda g: pd.Series(dict(
    n=len(g), pnl=g.pnl_pct.sum(), cf_end20_u=(g.post_end_c * (1 + g.pnl_pct)).sum()))).round(2).to_string())

print("\n2) LEVER tighten runner lock (trailing_struct_donch_win 80->40 / lower activate):")
print("   counterfactual: ideal 8%-close-trail from running peak-close (upper bound of tightening)")
# need per-symbol closes to sim; reuse bars
SYM = {s: g.reset_index(drop=True) for s, g in bars.groupby("symbol")}
rows = []
for _, t in E.iterrows():
    g = SYM.get(t.symbol)
    if g is None:
        rows.append(np.nan); continue
    di = pd.DatetimeIndex(g["date"])
    ei = di.get_indexer([t.entry_date])[0]
    xi = di.get_indexer([t.exit_date])[0]
    if ei < 0 or xi < 0:
        rows.append(np.nan); continue
    c = g["close"].to_numpy(float)
    ep = t.entry_price
    # ideal trail: from entry to decision bar, exit decision when close <= 0.92*runpeak(close),
    # armed only after peak-gain >= 15%; fill next close *0.9985-ish (use same-day close, first-order)
    peak = c[ei]
    hit = None
    for j in range(ei + 1, xi):
        peak = max(peak, c[j])
        if peak / ep - 1.0 >= 0.15 and c[j] <= peak * 0.92:
            hit = j
            break
    if hit is None:
        rows.append(np.nan)  # trail never fired before actual exit -> no change
    else:
        fill = c[min(hit + 1, len(c) - 1)] * 0.9985
        cf_pnl = fill / ep - 1.0 - 0.004 + 0.0  # approx net (entry_price already has buy slip)
        rows.append(cf_pnl - t.pnl_pct)
E["d_trail8"] = rows
T8 = E[E.d_trail8.notna()]
print(f"   trades where 8%-trail(arm15%) fires earlier: n={len(T8)} delta_sum={T8.d_trail8.sum():+.1f}u "
      f"(pos {(T8.d_trail8>0).sum()} / neg {(T8.d_trail8<0).sum()})")
print(T8.groupby("label").d_trail8.agg(["count", "sum"]).round(1).to_string())
print(T8.groupby("year_exit").d_trail8.agg(["count", "sum"]).round(1).to_string())

print("\n3) boundary scan sold-then-rallied (X=5%) — candidate discriminators at decision bar")
E["rallied"] = E.post_max_c >= 0.05
S2 = E[E.post_max_c.notna()].copy()
# per-symbol above-MA20 and leg6 at decision bar need enrich — approximate with stored flags:
for col, desc in [("f_nb", "belowMA20p2&nonbull"), ("f_lb", "leg6&lowbreadth"),
                  ("bull_cb_at", "market bull"), ("mkt_drop_at", "mkt drop gate")]:
    ct = pd.crosstab(S2[col], S2.rallied, normalize="index")
    print(f"  {desc:24s} P(rallied|flag): {ct.loc[True, True] if True in ct.index else float('nan'):.3f} "
          f"vs P(rallied|~flag): {ct.loc[False, True]:.3f}")
for col in ["gain_d", "giveback_d", "peak_d", "snr_at", "holding_days"]:
    q = S2.groupby(pd.qcut(S2[col], 5, duplicates="drop")).rallied.mean()
    print(f"  {col} quintile P(rallied): " + " ".join(f"{v:.2f}" for v in q))

print("\n4) LEVER exit_force_suppress (existing knob): soft-gate exits (nonbull bma20p2) in healthy uptrend")
# suppress tokens like abovema20: how many f_nb exits had close>MA50 at decision (healthy)?
NB = E[E.label == "force_nonbull_bma20p2"]
print(f"   f_nb exits: n={len(NB)} pnl={NB.pnl_pct.sum():+.1f} cf_end20={(NB.post_end_c*(1+NB.pnl_pct)).sum():+.1f}u "
      f"rallied5={NB.rallied.mean():.2f}")
E.to_csv(EM + "/gbx08_enriched2.csv", index=False)
print("saved enriched2")
