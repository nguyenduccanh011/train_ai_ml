# -*- coding: utf-8 -*-
"""hb_60: forensic shakeout-vs-top tren cac lenh 'signal' exit cua ab_noT (t2936).
Phan loai moi signal-exit: sau khi ban, trong N=20 bar gia HOI len (shakeout=ban non)
hay TIEP TUC xuong (top=ban dung). Do co hoi bo phi + test feature phan biet tai
thoi diem exit (dist MA20, MA20 slope, vol ratio, do sau pullback, wick). Chua train.
"""
from __future__ import annotations
import psycopg2, duckdb, pandas as pd, numpy as np
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)

FWD = 20         # cua so nhin toi sau exit
REC = 0.05       # nguong hoi phuc = shakeout
DROP = 0.05      # nguong tiep tuc xuong = top

pg = psycopg2.connect(host='localhost', port=5433, dbname='stockml', user='stockml', password='stockml_dev')
cur = pg.cursor()
cur.execute("select run_id from leaderboard_runs where template_id=2936 and run_seed=42 order by created_at desc limit 1")
run_id = cur.fetchone()[0]
t = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days,pnl_pct,exit_reason "
                "from run_trades where run_id=%s", pg, params=(run_id,))
pg.close()
t['entry_date'] = pd.to_datetime(t['entry_date']); t['exit_date'] = pd.to_datetime(t['exit_date'])
sig = t[t.exit_reason == 'signal'].copy()
print(f"signal exits: {len(sig)}/{len(t)}  (avg pnl {sig.pnl_pct.mean()*100:.2f}%)")

d = duckdb.connect(r"market_data/market.duckdb", read_only=True)
syms = sig.symbol.unique().tolist()
ph = ",".join("?" * len(syms))
px = d.execute(f"select symbol,date,open,high,low,close,volume from ohlcv where timeframe='1D' "
               f"and symbol in ({ph}) order by symbol,date", syms).fetchdf()
d.close()
px['date'] = pd.to_datetime(px['date'])
bysym = {}
for s, g in px.groupby('symbol'):
    g = g.set_index('date').sort_index()
    g['ma20'] = g['close'].rolling(20).mean()
    g['ma20_slope'] = g['ma20'].pct_change(5)
    g['vol_ma20'] = g['volume'].rolling(20).mean()
    g['ret'] = g['close'].pct_change()
    g['atr'] = (g['high'] - g['low']).rolling(14).mean() / g['close']
    bysym[s] = g

rows = []
for r in sig.itertuples(index=False):
    g = bysym.get(r.symbol)
    if g is None or r.exit_date not in g.index:
        continue
    i = g.index.get_loc(r.exit_date)
    ex_close = g['close'].iloc[i]
    fwd = g['close'].iloc[i + 1:i + 1 + FWD]
    if len(fwd) < 5:
        continue
    fmax = fwd.max() / ex_close - 1.0     # upside bo phi neu giu
    fmin = fwd.min() / ex_close - 1.0     # downside tranh duoc
    # features tai thoi diem exit (causal)
    row = g.iloc[i]
    dist_ma20 = ex_close / row['ma20'] - 1.0 if row['ma20'] > 0 else np.nan
    # do sau pullback: gia hien tai so voi peak 20-bar trailing
    trail_peak = g['close'].iloc[max(0, i - 20):i + 1].max()
    pull_depth = ex_close / trail_peak - 1.0
    vol_ratio = row['volume'] / row['vol_ma20'] if row['vol_ma20'] > 0 else np.nan
    lower_wick = (min(row['open'], row['close']) - row['low']) / (row['high'] - row['low'] + 1e-9)
    cls = 'shakeout' if fmax >= REC else ('top' if (fmax < 0.02 and fmin <= -DROP) else 'mixed')
    rows.append(dict(symbol=r.symbol, exit=r.exit_date.date(), pnl=r.pnl_pct * 100,
                     fmax=fmax * 100, fmin=fmin * 100, cls=cls,
                     dist_ma20=dist_ma20 * 100, ma20_slope=row['ma20_slope'] * 100,
                     pull_depth=pull_depth * 100, vol_ratio=vol_ratio,
                     lower_wick=lower_wick, atr=row['atr'] * 100))
f = pd.DataFrame(rows)
print(f"analyzed {len(f)} signal-exits\n")

vc = f['cls'].value_counts()
print("=== PHAN LOAI (fwd %d bar) ===" % FWD)
for k in ('shakeout', 'top', 'mixed'):
    n = int(vc.get(k, 0))
    sub = f[f.cls == k]
    print(f"  {k:9s}: {n:4d} ({n/len(f)*100:4.1f}%)  fmax_mean={sub.fmax.mean():+5.1f}%  "
          f"fmin_mean={sub.fmin.mean():+5.1f}%  pnl_mean={sub.pnl.mean():+5.1f}%")
print(f"\n  Co hoi bo phi (shakeout fmax tong): tb {f[f.cls=='shakeout'].fmax.mean():.1f}% x "
      f"{int(vc.get('shakeout',0))} lenh")

print("\n=== FEATURE phan biet shakeout vs top (median) ===")
feats = ['dist_ma20', 'ma20_slope', 'pull_depth', 'vol_ratio', 'lower_wick', 'atr']
sh = f[f.cls == 'shakeout']; tp = f[f.cls == 'top']
print(f"  {'feature':12s} {'shakeout':>10s} {'top':>10s} {'diff':>8s}")
for c in feats:
    a, b = sh[c].median(), tp[c].median()
    print(f"  {c:12s} {a:10.2f} {b:10.2f} {a-b:+8.2f}")
print("\nHB_60_DONE")
