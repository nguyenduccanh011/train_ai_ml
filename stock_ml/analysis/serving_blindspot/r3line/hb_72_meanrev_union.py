# -*- coding: utf-8 -*-
"""hb_72: probe REGIME-GATED mean-reversion UNION voi momentum (t2936).
Frontier data-backed (hb_71: mean-rev +2.35% dung 2024 dead-year, nhung -2.24% bear 2022).
Test: mean-rev sleeve (oversold-bounce + REGIME-GATE tat trong trending-bear) union chung
so K-slot voi momentum -> NAV/CAGR co nang khong (dac biet dead-year)? Offline, NavSim.

Mean-rev entry: dist_MA20 < -TH & down3 < -0.03, cooldown = 1 trade/symbol tai 1 thoi diem.
Regime-gate: skip khi VNINDEX/MA200 < -BEAR (trending bear = bat dao roi).
Exit: close >= MA20 (mean reached) hoac max_hold MH bars.
"""
from __future__ import annotations
import os, sys, psycopg2, duckdb, pandas as pd, numpy as np
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

HERE = os.path.dirname(__file__)
TH, BEAR, MH = float(os.environ.get("MR_TH", 0.04)), float(os.environ.get("MR_BEAR", 0.05)), 15
STRICT_BULL = os.environ.get("MR_STRICT", "0") == "1"   # require VNI>MA200 (rg>0)
CONFIRM = os.environ.get("MR_CONFIRM", "0") == "1"      # enter next bar only if up
VNI_CSV = "portable_data/vn_stock_ai_dataset_cleaned/context_features/symbol=VNINDEX/timeframe=1D/data.csv"

pg = psycopg2.connect(host='localhost', port=5433, dbname='stockml', user='stockml', password='stockml_dev')
cur = pg.cursor()
cur.execute("select run_id from leaderboard_runs where template_id=2936 and run_seed=42 order by created_at desc limit 1")
rid = cur.fetchone()[0]
mom = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s "
                  "and exit_date is not null", pg, params=(rid,))
cur.execute("select distinct symbol from run_signals where run_id=%s", (rid,)); univ = [r[0] for r in cur.fetchall()]
pg.close()

d = duckdb.connect(r"market_data/market.duckdb", read_only=True); ph = ",".join("?" * len(univ))
px = d.execute(f"select symbol,date,close from ohlcv where timeframe='1D' and symbol in ({ph}) "
               f"and date>='2019-06-01' order by symbol,date", univ).fetchdf()
d.close(); px['date'] = pd.to_datetime(px['date'])

v = pd.read_csv(VNI_CSV); v['d'] = pd.to_datetime(v['timestamp']).dt.tz_localize(None).dt.normalize()
v = v.drop_duplicates('d', keep='last').set_index('d')['close'].astype(float).sort_index()
vni_reg = (v / v.rolling(200, min_periods=100).mean() - 1.0)   # >0 bull, <<0 trending bear

# --- generate regime-gated mean-rev trades ---
mr = []
for s, g in px.groupby('symbol'):
    g = g.set_index('date').sort_index()
    ma20 = g['close'].rolling(20).mean()
    ret = g['close'].pct_change()
    dist = g['close'] / ma20 - 1
    down3 = ret.rolling(3).sum()
    sig = (dist < -TH) & (down3 < -0.03)
    idx = list(g.index); i = 20
    while i < len(idx) - 1:
        dt = idx[i]
        if sig.iloc[i] and not np.isnan(ma20.iloc[i]):
            rg = vni_reg.get(dt, np.nan)
            _gate_bad = (not (rg == rg)) or (rg < 0 if STRICT_BULL else rg < -BEAR)
            if _gate_bad:   # regime-gate: skip trending bear (or require bull if STRICT)
                i += 1; continue
            if CONFIRM and (i + 1 >= len(idx) or g['close'].iloc[i + 1] <= g['close'].iloc[i]):
                i += 1; continue   # wait for bounce confirmation (next bar up)
            _e = i + 1 if CONFIRM else i
            ent = g['close'].iloc[_e]; ent_dt = idx[_e]; i = _e
            j = i + 1
            while j < len(idx) and j <= i + MH:
                if g['close'].iloc[j] >= ma20.iloc[j]:   # mean reached
                    break
                j += 1
            j = min(j, len(idx) - 1)
            mr.append(dict(symbol=s, entry_date=ent_dt, exit_date=idx[j],
                           entry_price=ent, exit_price=g['close'].iloc[j]))
            i = j + 1   # cooldown: no overlap
        else:
            i += 1
mrdf = pd.DataFrame(mr)
mrdf['pnl'] = mrdf['exit_price'] / mrdf['entry_price'] - 1
mrdf['year'] = pd.to_datetime(mrdf['entry_date']).dt.year
print(f"mean-rev trades: {len(mrdf)}  avg_pnl={mrdf.pnl.mean()*100:.2f}%")
print("  by year total-pnl (u) & avg:")
for y in range(2020, 2027):
    my = mrdf[mrdf.year == y]
    print(f"    {y}: n={len(my):4d} tot={my.pnl.sum()*100:+7.1f}u avg={my.pnl.mean()*100 if len(my) else 0:+5.2f}%")

# --- NavSim: momentum-alone vs UNION ---
def nav(df, lo="2020-01-01"):
    p = os.path.join(HERE, "_tmp_union.csv"); df.to_csv(p, index=False)
    sim = NavSim2(p, date_lo=lo)
    return shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)

cols = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]
union = pd.concat([mom[cols], mrdf[cols]], ignore_index=True)
print("\n=== NavSim (K=25, adv) momentum-alone vs UNION ===")
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"), ("2024-01-01", "f24")):
    sm = nav(mom[cols], lo)["mean"]; su = nav(union, lo)["mean"]
    print(f"  {tag:4s}: momentum x{sm:.2f}  UNION x{su:.2f}  ({(su/sm-1)*100:+.1f}%)")
print("HB_72_DONE")
