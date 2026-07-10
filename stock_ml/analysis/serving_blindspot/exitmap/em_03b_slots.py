# -*- coding: utf-8 -*-
"""exitmap step 3b: slot theo-ma (1 vi the/ma). Utilization = % ma-ngay co vi the.
Dead-time diagnosis per-symbol-day: idle & limit treo (doi fill) vs idle & khong
co tin hieu trong 40 bar (doi tin hieu). Proxy tin hieu: frame serving 2643."""
import sqlite3

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = BASE + r"\exitmap"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"

df = pd.read_csv(f"{OUT}\\gbx08_enriched.csv")
sig = pd.read_csv(f"{BASE}/signals.csv")
uf = pd.read_csv(f"{BASE}/unfilled_signals.csv")

con = sqlite3.connect(DB)
cal = pd.read_sql("select date, count(*) c from ohlcv group by date having count(*)>=30 order by date", con)
con.close()
days = cal.date[(cal.date >= "2020-04-07") & (cal.date <= "2026-06-16")].to_numpy()
didx = {d: i for i, d in enumerate(days)}
n = len(days)
yr = np.array([d[:4] for d in days])

syms = sorted(df.symbol.unique())  # 61 ma gb_x08 da giao dich
S = {s: i for i, s in enumerate(syms)}

pos = np.zeros((len(syms), n), bool)   # co vi the
pend = np.zeros((len(syms), n), bool)  # limit dang treo
sig40 = np.zeros((len(syms), n), bool) # co buy-signal trong 40 bar

def rng(d0, d1):
    i0 = didx.get(d0, int(np.searchsorted(days, d0)))
    i1 = didx.get(d1, int(np.searchsorted(days, d1)))
    return max(i0, 0), min(i1, n - 1)

for r in df.itertuples():
    i0, i1 = rng(r.entry_date, r.exit_date)
    pos[S[r.symbol], i0:i1 + 1] = True
    # treo tu sau signal den truoc fill
    j0, _ = rng(r.entry_signal_date, r.entry_signal_date)
    pend[S[r.symbol], j0 + 1:i0] = True

u = uf[uf.drop_reason == "unfilled"]
for r in u.itertuples():
    if r.symbol not in S:
        continue
    j = int(np.searchsorted(days, r.signal_date))
    pend[S[r.symbol], j + 1:min(j + 41, n)] = True

buy = sig[sig.signal > 0]
for r in buy.itertuples():
    if r.symbol not in S:
        continue
    j = didx.get(r.date)
    if j is None:
        continue
    sig40[S[r.symbol], j:min(j + 40, n)] = True

idle = ~pos
diag = np.where(pos, "pos", np.where(pend, "idle_doi_fill",
                np.where(sig40, "idle_sig40_no_pend", "idle_doi_tin_hieu")))

print("PER-SYMBOL-DAY (61 ma x", n, "ngay):")
rows = []
for y in sorted(set(yr)):
    m = yr == y
    tot = m.sum() * len(syms)
    c = pd.Series(diag[:, m].ravel()).value_counts()
    rows.append(dict(year=y, tot=tot, pos=c.get("pos", 0) / tot,
                     doi_fill=c.get("idle_doi_fill", 0) / tot,
                     sig_no_pend=c.get("idle_sig40_no_pend", 0) / tot,
                     doi_tin_hieu=c.get("idle_doi_tin_hieu", 0) / tot))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# khoang idle dai nhat cua toan he (mean pos across syms thap)
mo = pos.mean(axis=0)
lo = pd.Series(mo).rolling(20).mean()
print("\n5 diem day 20-ngay occupancy thap nhat:")
for i in lo.nsmallest(5).index:
    print(f"  {days[i]}: mean-occ-20d={lo[i]:.3f}")

# ---- EXTRAS (fix) ----
df["age_b"] = pd.cut(df.fill_age, [0, 5, 15, 40], labels=["d1-5", "d6-15", "d16-40"])
df["hold_c"] = df.hold.clip(lower=1)
g = df.groupby("age_b", observed=True)
t = pd.DataFrame(dict(u=g.pnl.sum(), slotdays=g.hold_c.sum()))
t["u_per_slotday"] = t.u / t.slotdays
print("\nhieu suat per slot-day theo fill-age:")
print(t.round(4).to_string())

q4 = df.runup20 >= df.runup20.quantile(0.8)
fast = df.fill_age <= 2
for name, m in [("hotQ4+fast", q4 & fast), ("hotQ4+slow", q4 & ~fast),
                ("notHot+fast", ~q4 & fast), ("notHot+slow", ~q4 & ~fast)]:
    gg = df[m]
    print(f"{name}: n={len(gg)} mean={gg.pnl.mean():.4f} WR={(gg.pnl>0).mean():.3f} "
          f"p10={gg.pnl.quantile(0.1):.3f} min={gg.pnl.min():.3f}")

t10 = df[df.pnl <= -0.10]
print(f"\ntail <=-10%: n={len(t10)} u={t10.pnl.sum():.2f} = {t10.pnl.sum()/df[df.pnl<0].pnl.sum()*100:.1f}% u am")

con = sqlite3.connect(DB)
px = pd.read_sql("select symbol,date,close from ohlcv where date>='2019-01-01' order by symbol,date", con)
con.close()
px["ret20"] = px.groupby("symbol").close.pct_change(20)
px["year"] = px.date.str[:4]
mk = px[(px.date >= "2020-04-07") & (px.date <= "2026-06-16")].groupby("year").ret20.agg(["mean", "median"])
mk["pct_up"] = px[(px.date >= "2020-04-07") & (px.date <= "2026-06-16")].groupby("year").ret20.apply(lambda x: (x > 0).mean())
print("\nmarket proxy (universe ret20 theo nam):")
print(mk.round(4).to_string())
