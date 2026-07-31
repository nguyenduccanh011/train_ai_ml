# -*- coding: utf-8 -*-
"""hb_73: ML-SELECTED mean-rev union. Rule-based mean-rev that bai (hb_72: union -22%);
crux = chat luong thap + nam thua keo compounding. Thu ML-label: walk-forward classifier
du bao cu oversold nao SE bounce (fwd10d>3%), CHI lay top-P -> union voi momentum.
Neu ML-selection lam union > momentum = proof mean-rev sleeve kha thi (frontier user).
Causal: train tren nam < Y (gap), predict nam Y (OOF).
"""
from __future__ import annotations
import os, sys, psycopg2, duckdb, pandas as pd, numpy as np
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402
import lightgbm as lgb  # noqa: E402

HERE = os.path.dirname(__file__)
MH = 15
VNI_CSV = "portable_data/vn_stock_ai_dataset_cleaned/context_features/symbol=VNINDEX/timeframe=1D/data.csv"

pg = psycopg2.connect(host='localhost', port=5433, dbname='stockml', user='stockml', password='stockml_dev')
cur = pg.cursor()
cur.execute("select run_id from leaderboard_runs where template_id=2936 and run_seed=42 order by created_at desc limit 1")
rid = cur.fetchone()[0]
mom = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", pg, params=(rid,))
cur.execute("select distinct symbol from run_signals where run_id=%s", (rid,)); univ = [r[0] for r in cur.fetchall()]
pg.close()

d = duckdb.connect(r"market_data/market.duckdb", read_only=True); ph = ",".join("?" * len(univ))
px = d.execute(f"select symbol,date,high,low,close,volume from ohlcv where timeframe='1D' and symbol in ({ph}) and date>='2018-06-01' order by symbol,date", univ).fetchdf()
d.close(); px['date'] = pd.to_datetime(px['date'])
v = pd.read_csv(VNI_CSV); v['d'] = pd.to_datetime(v['timestamp']).dt.tz_localize(None).dt.normalize()
v = v.drop_duplicates('d', keep='last').set_index('d')['close'].astype(float).sort_index()
vni_reg = (v / v.rolling(200, min_periods=100).mean() - 1.0)
vni_mom = v / v.shift(60) - 1.0

# --- build ALL oversold setups with features + fwd label + realistic exit ---
FEATS = ['dist20', 'dist50', 'down3', 'down5', 'rsi2', 'volr', 'dist52w', 'vni_reg', 'vni_mom', 'atrp']
recs = []
for s, g in px.groupby('symbol'):
    g = g.set_index('date').sort_index()
    ma20 = g['close'].rolling(20).mean(); ma50 = g['close'].rolling(50).mean()
    ret = g['close'].pct_change()
    up = ret.clip(lower=0).rolling(2).mean(); dn = (-ret.clip(upper=0)).rolling(2).mean()
    rsi2 = 100 - 100 / (1 + up / (dn + 1e-9))
    volr = g['volume'] / g['volume'].rolling(20).mean()
    atrp = (g['high'] - g['low']).rolling(14).mean() / g['close']
    dist20 = g['close'] / ma20 - 1; dist50 = g['close'] / ma50 - 1
    d52 = g['close'] / g['close'].rolling(252, min_periods=60).max() - 1
    down3 = ret.rolling(3).sum(); down5 = ret.rolling(5).sum()
    sig = (dist20 < -0.04) & (down3 < -0.03)
    idx = list(g.index); i = 20
    while i < len(idx) - 1:
        if sig.iloc[i] and not np.isnan(ma20.iloc[i]):
            dt = idx[i]; ent = g['close'].iloc[i]
            j = i + 1
            while j < len(idx) and j <= i + MH:
                if g['close'].iloc[j] >= ma20.iloc[j]:
                    break
                j += 1
            j = min(j, len(idx) - 1)
            fwd = g['close'].iloc[j] / ent - 1
            recs.append(dict(symbol=s, entry_date=dt, exit_date=idx[j], entry_price=ent,
                             exit_price=g['close'].iloc[j], year=dt.year, fwd=fwd,
                             dist20=dist20.iloc[i], dist50=dist50.iloc[i], down3=down3.iloc[i],
                             down5=down5.iloc[i], rsi2=rsi2.iloc[i], volr=volr.iloc[i],
                             dist52w=d52.iloc[i], vni_reg=vni_reg.get(dt, np.nan),
                             vni_mom=vni_mom.get(dt, np.nan), atrp=atrp.iloc[i]))
            i = j + 1
        else:
            i += 1
S = pd.DataFrame(recs).dropna(subset=FEATS + ['fwd'])
S['label'] = (S['fwd'] > 0.03).astype(int)
print(f"oversold setups: {len(S)}  base bounce-rate={S.label.mean()*100:.1f}%")

# --- walk-forward ML select: train years < Y-1 (gap), predict Y ---
S['pred'] = np.nan
for Y in range(2020, 2027):
    tr = S[S.year <= Y - 2]  # gap 1 year
    te = S[S.year == Y]
    if len(tr) < 200 or len(te) == 0:
        continue
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15,
                           min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
                           reg_lambda=1.0, verbose=-1)
    m.fit(tr[FEATS], tr['label'])
    S.loc[S.year == Y, 'pred'] = m.predict_proba(te[FEATS])[:, 1]
Sp = S.dropna(subset=['pred'])
print("by-year ML-selected mean-rev (pred>0.6):")
for th in (0.55, 0.6, 0.65):
    sel = Sp[Sp.pred >= th]
    print(f"  P>={th}: n={len(sel):4d} avg_fwd={sel.fwd.mean()*100:+.2f}% (base {Sp.fwd.mean()*100:+.2f}%)  by yr: "
          + " ".join(f"{y}:{sel[sel.year==y].fwd.mean()*100:+.1f}%" for y in range(2022, 2027)))

# --- union NavSim: momentum vs momentum+ML-selected mean-rev ---
def nav(df, lo="2020-01-01"):
    p = os.path.join(HERE, "_tmp_mlu.csv"); df.to_csv(p, index=False)
    return shuffle_stats(NavSim2(p, date_lo=lo), K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]

cols = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]
print("\n=== NavSim momentum vs UNION (ML-selected mean-rev) ===")
for th in (0.6, 0.65):
    sel = Sp[Sp.pred >= th]
    union = pd.concat([mom[cols], sel[cols]], ignore_index=True)
    for lo, tag in (("2020-01-01", "full"), ("2024-01-01", "f24")):
        sm = nav(mom[cols], lo); su = nav(union, lo)
        print(f"  P>={th} {tag}: mom x{sm:.2f}  UNION x{su:.2f} ({(su/sm-1)*100:+.1f}%)")
print("HB_73_DONE")
