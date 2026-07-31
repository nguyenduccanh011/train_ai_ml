# -*- coding: utf-8 -*-
"""hb_70: entry forensic — entry thang/thua co cluster theo REGIME thi truong luc vao khong?
Neu co (dac biet giai thich nam 2024 entry lat dau — root-cause-ml-regime-fragile), thi
entry regime-gate hard rule (chi vao khi VNINDEX>MA(N)) la lever tiem nang, giong exit-skip.
Dung VNINDEX (khop gate engine). AUC regime cho win(pnl>0) vs loss, overall + by year.
"""
from __future__ import annotations
import os, psycopg2, pandas as pd, numpy as np

VNI_CSV = "portable_data/vn_stock_ai_dataset_cleaned/context_features/symbol=VNINDEX/timeframe=1D/data.csv"
pg = psycopg2.connect(host='localhost', port=5433, dbname='stockml', user='stockml', password='stockml_dev')
cur = pg.cursor()
cur.execute("select run_id from leaderboard_runs where template_id=2936 and run_seed=42 order by created_at desc limit 1")
rid = cur.fetchone()[0]
t = pd.read_sql("select entry_date,pnl_pct,holding_days from run_trades where run_id=%s", pg, params=(rid,))
pg.close()
t['entry_date'] = pd.to_datetime(t['entry_date'])

v = pd.read_csv(VNI_CSV)
v['d'] = pd.to_datetime(v['timestamp']).dt.tz_localize(None).dt.normalize()
v = v.drop_duplicates('d', keep='last').set_index('d')['close'].astype(float).sort_index()
reg = pd.DataFrame(index=v.index)
for w in (50, 100, 200):
    reg[f'vni_ma{w}'] = (v / v.rolling(w, min_periods=w // 2).mean() - 1.0)
reg['vni_mom60'] = v / v.shift(60) - 1.0
reg['vni_above200'] = (v > v.rolling(200, min_periods=100).mean()).astype(float)

t = t.join(reg, on='entry_date')
t['win'] = (t.pnl_pct > 0).astype(int)
t['year'] = t.entry_date.dt.year
print(f"trades={len(t)}  overall WR={t.win.mean()*100:.1f}%  avg_pnl={t.pnl_pct.mean()*100:.2f}%")


def auc(a, b):
    a = a.dropna().values; b = b.dropna().values
    if len(a) < 5 or len(b) < 5: return np.nan
    allv = np.concatenate([a, b]); o = allv.argsort().argsort() + 1
    return (o[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


feats = ['vni_ma50', 'vni_ma100', 'vni_ma200', 'vni_mom60']
w = t[t.win == 1]; l = t[t.win == 0]
print("\n=== AUC regime tach WIN vs LOSS (overall; >0.5 = regime cao hon o winner) ===")
for c in feats:
    print(f"  {c:12s} AUC={auc(w[c], l[c]):.3f}  win_med={w[c].median():+.3f} loss_med={l[c].median():+.3f}")

print("\n=== Theo NAM: avg_pnl & WR khi VNI>MA200 vs <=MA200 (regime timing) ===")
print(f"  {'year':4s} {'n':>4s} {'pnl_all':>8s} | {'n_bull':>6s} {'pnl_bull':>8s} {'wr_bull':>7s} | {'n_bear':>6s} {'pnl_bear':>8s} {'wr_bear':>7s}")
for y in sorted(t.year.dropna().unique()):
    ty = t[t.year == y]
    bull = ty[ty.vni_above200 == 1]; bear = ty[ty.vni_above200 == 0]
    def fmt(d): return (len(d), d.pnl_pct.mean()*100 if len(d) else 0, d.win.mean()*100 if len(d) else 0)
    nb, pb, wb = fmt(bull); nr, pr, wr = fmt(bear)
    print(f"  {int(y):4d} {len(ty):4d} {ty.pnl_pct.mean()*100:+7.2f}% | {nb:6d} {pb:+7.2f}% {wb:6.1f}% | {nr:6d} {pr:+7.2f}% {wr:6.1f}%")
print("\nHB_70_DONE")
