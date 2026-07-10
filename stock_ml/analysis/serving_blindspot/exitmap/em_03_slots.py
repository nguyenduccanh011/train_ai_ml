# -*- coding: utf-8 -*-
"""exitmap step 3: slot utilization & dead-time cua gb_x08 + extras.

Slot occupancy tai lap tu trades (entry_date..exit_date). So slot = max concurrent.
Dead-time diagnosis dung frame serving 2643 (signals.csv + unfilled) lam proxy
(79% trades trung gb_x08) -> ngay ranh: co limit dang treo (doi fill) hay khong
co tin hieu nao trong 40 bar (doi tin hieu)."""
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = BASE + r"\exitmap"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"

df = pd.read_csv(f"{OUT}\\gbx08_enriched.csv")
sig = pd.read_csv(f"{BASE}/signals.csv")
uf = pd.read_csv(f"{BASE}/unfilled_signals.csv")

con = sqlite3.connect(DB)
cal = pd.read_sql("select date, count(*) nsym from ohlcv group by date having count(*)>=30 order by date", con)
px = pd.read_sql("select symbol,date,close from ohlcv order by symbol,date", con)
con.close()

days = cal.date[(cal.date >= "2020-04-07") & (cal.date <= "2026-06-16")].to_numpy()
didx = {d: i for i, d in enumerate(days)}
n = len(days)

# ---- occupancy tu trades gb_x08 ----
occ = np.zeros(n, int)
for r in df.itertuples():
    i0 = didx.get(r.entry_date)
    i1 = didx.get(r.exit_date, n - 1)
    if i0 is None:
        i0 = np.searchsorted(days, r.entry_date)
    if i1 is None:
        i1 = np.searchsorted(days, r.exit_date)
    occ[i0:i1 + 1] += 1

slots = occ.max()
print("max concurrent =", slots, "| phan phoi occupancy (so ngay):")
print(pd.Series(occ).value_counts().sort_index().to_string())

years = pd.Series([d[:4] for d in days])
od = pd.DataFrame(dict(date=days, occ=occ, year=years.values))
print("\nutilization theo nam (occ / slots):")
t = od.groupby("year").agg(days=("occ", "size"), mean_occ=("occ", "mean"),
                           full=("occ", lambda x: (x >= slots).mean()),
                           empty=("occ", lambda x: (x == 0).mean()))
t["util"] = t.mean_occ / slots
print(t.round(3).to_string())

# ---- khoang rong dai nhat (occ==0) & thieu slot (occ<slots) ----
def runs(mask):
    out, st = [], None
    for i, m in enumerate(mask):
        if m and st is None:
            st = i
        elif not m and st is not None:
            out.append((st, i - 1)); st = None
    if st is not None:
        out.append((st, len(mask) - 1))
    return out

empty_runs = sorted(runs(occ == 0), key=lambda r: r[1] - r[0], reverse=True)[:10]
print("\n10 khoang occ==0 dai nhat:")
for a, b in empty_runs:
    print(f"  {days[a]} -> {days[b]} ({b-a+1} ngay)")

# ---- pending limit proxy tu frame 2643 ----
# pending(s, d): co buy-signal cua s trong 40 bar truoc d chua fill (theo unfilled) hoac fill sau d
A = {}
for s, g in px.groupby("symbol"):
    A[s] = {d: i for i, d in enumerate(g.date.to_numpy())}

# build per-day pending count + signal count (frame proxy)
buy = sig[sig.signal > 0][["symbol", "date"]].copy()
buy_bar = []
for r in buy.itertuples():
    a = A.get(r.symbol)
    if a and r.date in a:
        buy_bar.append((r.symbol, r.date))
buy_by_day = defaultdict(int)
for s, d in buy_bar:
    buy_by_day[d] += 1

# pending window: tu unfilled (treo het 40 bar) + trades gb_x08 (treo den fill)
pend = np.zeros(n, int)
u = uf[uf.drop_reason == "unfilled"]
sym_days = {s: sorted(a.keys()) for s, a in A.items()}
for r in u.itertuples():
    a = A.get(r.symbol)
    if not a or r.signal_date not in a:
        continue
    i0 = np.searchsorted(days, r.signal_date) + 1
    i1 = min(i0 + 39, n - 1)
    pend[i0:i1 + 1] += 1
for r in df.itertuples():
    i0 = np.searchsorted(days, r.entry_signal_date) + 1
    i1 = didx.get(r.entry_date, i0)
    if i1 is None or i1 < i0:
        continue
    pend[i0:i1] += 1  # truoc ngay fill

sig40 = np.zeros(n, int)  # co buy-signal nao trong 40 bar gan nhat
cnt = np.zeros(n, int)
for d, c in buy_by_day.items():
    j = didx.get(d)
    if j is not None:
        cnt[j] = c
csum = np.cumsum(cnt)
for i in range(n):
    lo = max(0, i - 39)
    sig40[i] = csum[i] - (csum[lo - 1] if lo > 0 else 0)

od["pend"] = pend
od["sig40"] = sig40
free = od[od.occ < slots].copy()
free["free_slots"] = slots - free.occ
free["diag"] = np.where(free.pend > 0, "doi_fill(limit treo)",
                        np.where(free.sig40 > 0, "signal_window_but_no_pending", "doi_tin_hieu"))
print("\nDIAGNOSIS ngay co slot rong (weighted theo so slot rong):")
print(free.groupby(["year", "diag"]).free_slots.sum().unstack(fill_value=0).to_string())
print("\n% slot-ngay rong theo nam:", )
tot = od.groupby("year").occ.count() * slots
fr = free.groupby("year").free_slots.sum()
print(((fr / tot) * 100).round(1).to_string())

# trong khoang rong dai nhat: pending?
print("\ntrang thai trong 10 khoang rong dai nhat (mean pend, mean sig40):")
for a, b in empty_runs:
    seg = od.iloc[a:b + 1]
    print(f"  {days[a]}->{days[b]}: pend_mean={seg.pend.mean():.1f} sig40_mean={seg.sig40.mean():.1f}")

# ---- EXTRAS ----
print("\n===== EXTRAS =====")
# per-day efficiency theo fill-age bucket
df["age_b"] = pd.cut(df.fill_age, [0, 5, 15, 40], labels=["d1-5", "d6-15", "d16-40"])
df["hold_c"] = df.hold.clip(lower=1)
t = df.groupby("age_b", observed=True).apply(
    lambda g: pd.Series(dict(u=g.pnl.sum(), slotdays=g.hold_c.sum(),
                             u_per_slotday=g.pnl.sum() / g.hold_c.sum())), include_groups=False)
print("hieu suat per slot-day theo fill-age:")
print(t.round(4).to_string())

# interaction: runup20 Q4 x fill nhanh (knife nghi van)
q4 = df.runup20 >= df.runup20.quantile(0.8)
fast = df.fill_age <= 2
for name, m in [("hotQ4+fast", q4 & fast), ("hotQ4+slow", q4 & ~fast),
                ("notHot+fast", ~q4 & fast)]:
    g = df[m]
    print(f"{name}: n={len(g)} mean={g.pnl.mean():.4f} WR={(g.pnl>0).mean():.3f} "
          f"p10={g.pnl.quantile(0.1):.3f} minPnl={g.pnl.min():.3f}")

# tail <=-10%
t10 = df[df.pnl <= -0.10]
print(f"\ntail <=-10%: n={len(t10)} u={t10.pnl.sum():.2f} = {t10.pnl.sum()/df[df.pnl<0].pnl.sum()*100:.1f}% u am")

# market proxy theo nam: universe mean 20d return & % ngay uptrend
px["ret20"] = px.groupby("symbol").close.pct_change(20)
px["year"] = px.date.str[:4]
mk = px[(px.date >= "2020-04-07") & (px.date <= "2026-06-16")].groupby("year").ret20.agg(["mean", "median"])
print("\nmarket proxy (universe ret20):")
print(mk.round(4).to_string())
