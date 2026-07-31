# -*- coding: utf-8 -*-
"""Audit: reproduce the overshoot 'falling-knife' filter mechanism of hb_deploy_osdef/gtos.
Checks: (1) is the [sig..fill] window min-low actually the FILL DAY's low (post-fill info)?
(2) would a pre-fill-only version of the filter make the same decisions?
(3) pnl of filtered vs kept trades, per-year; (4) does OSTHR (full-sample pooled p90) leak?"""
import warnings, psycopg2, pandas as pd, numpy as np, duckdb
warnings.filterwarnings("ignore")

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
BASE_RID = "template/x2_struct_to-69338138"
OSDEF_RID = "template/x2_struct_to_k10_cs5ma50_r7ec_osdef-69338138"
PB_PCT = 0.045

con = psycopg2.connect(**PG)
tr = pd.read_sql("select symbol,entry_signal_date,entry_date,entry_price,exit_date,exit_price,pnl_pct "
                 "from run_trades where run_id=%s and exit_date is not null", con, params=(BASE_RID,))
sk = pd.read_sql("select symbol,signal_date,entry_date,pnl_pct from run_skipped "
                 "where run_id=%s and skip_reason='overshoot_fallknife'", con, params=(OSDEF_RID,))
con.close()

cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                "ORDER BY symbol,date").fetchdf()
cx.close()
px["date"] = pd.to_datetime(px["date"])
LO, CLO, DIDX = {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date")
    LO[s] = g["low"].values; CLO[s] = g["close"].values
    DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}

rows = []
for r in tr.itertuples():
    di = DIDX.get(r.symbol, {})
    si = di.get(str(r.entry_signal_date)[:10]); fi = di.get(str(r.entry_date)[:10])
    if si is None or fi is None or fi < si:
        continue
    seg = LO[r.symbol][si:fi + 1]
    csig = CLO[r.symbol][si]
    ov = (r.entry_price - seg.min()) / csig                       # exact hb_deploy code
    argmin_is_fill = int(np.argmin(seg)) == (fi - si)
    # pre-fill-only variant (info available BEFORE the fill day's session completes)
    ov_pre = (r.entry_price - LO[r.symbol][si:fi].min()) / csig if fi > si else np.nan
    limit = csig * (1.0 - PB_PCT)
    rows.append(dict(symbol=r.symbol, ed=str(r.entry_date)[:10], year=int(str(r.entry_date)[:4]),
                     ov=ov, ov_pre=ov_pre, argmin_is_fill=argmin_is_fill,
                     fill_low_le_limit=LO[r.symbol][fi] <= limit + 1e-9,
                     entry_vs_limit=r.entry_price / limit - 1.0,
                     wait_days=fi - si, pnl=r.pnl_pct))
A = pd.DataFrame(rows)
print(f"trades mapped: {len(A)}/{len(tr)}")
print(f"P1 min-low occurs ON fill day: {A.argmin_is_fill.mean()*100:.1f}%")
print(f"P1b fill-day low <= limit (pullback construction): {A.fill_low_le_limit.mean()*100:.1f}%")
print(f"P1c entry_price == limit (mean rel diff): {A.entry_vs_limit.abs().mean():.5f}, max {A.entry_vs_limit.abs().max():.5f}")

OSTHR = float(np.nanpercentile(A.ov.values, 90))
filt = A[A.ov > OSTHR]; kept = A[A.ov <= OSTHR]
print(f"\nP2 OSTHR(p90, this seed) = {OSTHR:.4f}; filtered {len(filt)} trades")
print(f"   filtered: mean pnl {filt.pnl.mean()*100:+.2f}%  neg {int((filt.pnl<0).sum())}/{len(filt)}")
print(f"   kept    : mean pnl {kept.pnl.mean()*100:+.2f}%  neg {int((kept.pnl<0).sum())}/{len(kept)}")
print(f"   filtered min-low-on-fill-day: {filt.argmin_is_fill.mean()*100:.1f}%")
pre_would = filt.ov_pre > OSTHR
print(f"P3 of {len(filt)} filtered, pre-fill-low ALSO exceeds OSTHR: {int(pre_would.sum())} "
      f"({pre_would.mean()*100:.1f}%) -> {100-pre_would.mean()*100:.1f}% of skips REQUIRE the fill day's own low")
print("\nP4 per-year filtered vs kept mean pnl:")
print(A.groupby("year").apply(lambda g: pd.Series(dict(
    n=len(g), n_filt=int((g.ov > OSTHR).sum()),
    filt_pnl=g.loc[g.ov > OSTHR, "pnl"].mean(), kept_pnl=g.loc[g.ov <= OSTHR, "pnl"].mean(),
    filt_sum=g.loc[g.ov > OSTHR, "pnl"].sum()))).to_string())

# P5: threshold non-causality — per-year p90 vs full-sample p90
print("\nP5 per-year p90 of ov (full-sample OSTHR is one 2020-2026 constant):")
print(A.groupby("year")["ov"].quantile(0.9).to_string())

# P6: overlap of this-seed filtered set with the registered run_skipped (seed 42 pooled-thr)
skset = set(zip(sk.symbol, sk.entry_date.astype(str).str[:10]))
fset = set(zip(filt.symbol, filt.ed))
print(f"\nP6 registered run_skipped overshoot_fallknife: {len(sk)} rows; "
      f"overlap with this-seed-123 reconstruction: {len(skset & fset)}")
print(f"   registered skipped pnl: mean {sk.pnl_pct.mean()*100:+.2f}%, neg {int((sk.pnl_pct<0).sum())}/{len(sk)}")
