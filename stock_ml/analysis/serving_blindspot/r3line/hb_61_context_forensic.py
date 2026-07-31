# -*- coding: utf-8 -*-
"""hb_61: shakeout-vs-top co HOC DUOC khong? Test feature CONTEXT (regime/RS/trend dai)
vs price cuc bo (hb_60 da chung minh yeu). Neu context tach shakeout/top ro -> exit head
moi nen dung context. Neu KHONG -> shakeout-vs-top bat kha du bao tai exit (bao cao that).

Do bang: (a) median diff, (b) AUC don-bien (kha nang phan loai shakeout vs top).
"""
from __future__ import annotations
import psycopg2, duckdb, pandas as pd, numpy as np
pd.set_option('display.width', 200)

FWD, REC, DROP = 20, 0.05, 0.05
pg = psycopg2.connect(host='localhost', port=5433, dbname='stockml', user='stockml', password='stockml_dev')
cur = pg.cursor()
cur.execute("select run_id from leaderboard_runs where template_id=2936 and run_seed=42 order by created_at desc limit 1")
run_id = cur.fetchone()[0]
t = pd.read_sql("select symbol,exit_date,exit_price,exit_reason,pnl_pct from run_trades where run_id=%s", pg, params=(run_id,))
cur.execute("select distinct symbol from run_signals where run_id=%s", (run_id,))
univ = [r[0] for r in cur.fetchall()]
pg.close()
t['exit_date'] = pd.to_datetime(t['exit_date'])
sig = t[t.exit_reason == 'signal'].copy()
print(f"universe={len(univ)} syms, signal-exits={len(sig)}")

d = duckdb.connect(r"market_data/market.duckdb", read_only=True)
ph = ",".join("?" * len(univ))
px = d.execute(f"select symbol,date,high,low,close from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date", univ).fetchdf()
d.close()
px['date'] = pd.to_datetime(px['date'])

# --- market index (equal-weight ret) + breadth (% above MA50) tren toan universe ---
piv = px.pivot_table(index='date', columns='symbol', values='close').sort_index()
mkt_ret = piv.pct_change().mean(axis=1)                       # equal-weight daily ret
mkt_idx = (1 + mkt_ret).cumprod()
mkt_ret20 = mkt_idx.pct_change(20)
above_ma50 = (piv > piv.rolling(50).mean()).mean(axis=1)      # breadth
mkt_ma50 = mkt_idx.rolling(50).mean()
mkt_above = (mkt_idx > mkt_ma50).astype(float)

bysym = {}
for s, g in px.groupby('symbol'):
    g = g.set_index('date').sort_index()
    g['ma50'] = g['close'].rolling(50).mean()
    g['ma100'] = g['close'].rolling(100).mean()
    g['ret20'] = g['close'].pct_change(20)
    g['ret60'] = g['close'].pct_change(60)
    g['hi252'] = g['close'].rolling(252, min_periods=60).max()
    bysym[s] = g

rows = []
for r in sig.itertuples(index=False):
    g = bysym.get(r.symbol)
    if g is None or r.exit_date not in g.index:
        continue
    i = g.index.get_loc(r.exit_date)
    exc = g['close'].iloc[i]
    fwd = g['close'].iloc[i + 1:i + 1 + FWD]
    if len(fwd) < 5:
        continue
    fmax = fwd.max() / exc - 1.0
    fmin = fwd.min() / exc - 1.0
    cls = 'shakeout' if fmax >= REC else ('top' if (fmax < 0.02 and fmin <= -DROP) else 'mixed')
    row = g.iloc[i]; dt = r.exit_date
    rows.append(dict(cls=cls,
        mkt_ret20=mkt_ret20.get(dt, np.nan) * 100,
        breadth=above_ma50.get(dt, np.nan) * 100,
        mkt_above_ma50=mkt_above.get(dt, np.nan),
        rs20=(row['ret20'] - (mkt_ret20.get(dt, np.nan))) * 100,
        rs60=row['ret60'] * 100,
        dist_ma50=(exc / row['ma50'] - 1) * 100 if row['ma50'] > 0 else np.nan,
        dist_ma100=(exc / row['ma100'] - 1) * 100 if row['ma100'] > 0 else np.nan,
        above_ma50=float(exc > row['ma50']) if row['ma50'] > 0 else np.nan,
        dist_hi252=(exc / row['hi252'] - 1) * 100 if row['hi252'] > 0 else np.nan))
f = pd.DataFrame(rows)
sh = f[f.cls == 'shakeout']; tp = f[f.cls == 'top']
print(f"analyzed {len(f)}: shakeout={len(sh)} top={len(tp)} mixed={len(f)-len(sh)-len(tp)}\n")


def auc(a, b):  # P(shakeout feature > top feature); 0.5=no sep, dist from 0.5 = power
    a = a.dropna().values; b = b.dropna().values
    if len(a) < 5 or len(b) < 5:
        return np.nan
    from itertools import product
    # rank-based Mann-Whitney AUC
    allv = np.concatenate([a, b]); order = allv.argsort().argsort() + 1
    ra = order[:len(a)].sum()
    return (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


print(f"=== CONTEXT features: shakeout vs top (median + AUC, |AUC-0.5|>0.10 = co luc) ===")
print(f"  {'feature':14s} {'shakeout':>9s} {'top':>9s} {'AUC':>6s} {'power':>6s}")
for c in ['mkt_ret20', 'breadth', 'mkt_above_ma50', 'rs20', 'rs60', 'dist_ma50', 'dist_ma100', 'above_ma50', 'dist_hi252']:
    a = auc(sh[c], tp[c]); pw = abs(a - 0.5) if a == a else np.nan
    print(f"  {c:14s} {sh[c].median():9.2f} {tp[c].median():9.2f} {a:6.3f} {pw:6.3f}")
print("\nHB_61_DONE")
