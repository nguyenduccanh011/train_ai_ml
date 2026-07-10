# -*- coding: utf-8 -*-
"""HY manh A (offline): buy-bars cua continuation h6 head (t1058, bundle n2_pbfill seed42)
tru di buy-bars hien co cua gb_x08 (buy_union tu sa_scores.parquet) — phan THEM roi vao
nam nao? Quyet dinh co chay hy_ens5 (slot ensemble 5 tren clone 2783) hay khong.

Nguong: zE > -1.2 (native t1058) + cac muc ensemble-style {0.7, 0.9, 1.2}.
"""
import os

import pandas as pd

BASE = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
BUNDLE = r"f:/PROJECTS/train_ai_ml/bundles/bundle_n2_pbfill_p0p03_w25_2026-01-01_wf/prediction_history.parquet"
W, MP = 252, 60

ph = pd.read_parquet(BUNDLE, columns=["symbol", "date", "score"])
ph["date"] = pd.to_datetime(ph["date"])
ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)


def causal_z(s, g):
    def _cz(x):
        m = x.rolling(W, min_periods=MP).mean()
        sd = x.rolling(W, min_periods=MP).std()
        return (x - m) / (sd + 1e-12)
    return s.groupby(g, sort=False).transform(_cz)


ph["z_cont6"] = causal_z(ph["score"], ph["symbol"])

sa = pd.read_parquet(os.path.join(BASE, "scoreaudit", "sa_scores.parquet"),
                     columns=["symbol", "date", "buy_main", "buy2", "buy3", "buy4", "buy5",
                              "buy_union", "gate_entry", "z_score3"])
sa["date"] = pd.to_datetime(sa["date"])
df = sa.merge(ph[["symbol", "date", "z_cont6"]], on=["symbol", "date"], how="inner")
df = df.dropna(subset=["z_cont6"])
df["year"] = df.date.dt.year
print("joined bars:", len(df), df.date.min().date(), "->", df.date.max().date())

# tuong quan voi score3 (continuation h10 cua gb) — head co thong tin MOI khong?
c = df[["z_cont6", "z_score3"]].dropna()
print(f"\ncorr(z_cont6 h6-lean, z_score3 h10-gb) = {c.z_cont6.corr(c.z_score3):.3f} "
      f"(rank {c.z_cont6.rank().corr(c.z_score3.rank()):.3f})")

THRS = [-1.2, 0.7, 0.9, 1.2]
print("\n=== buy-bars continuation h6 vs buy_union gb_x08, theo nam ===")
print(f"{'year':>5} {'bars':>7} {'union':>7} | " +
      " | ".join(f"z>{t}: fire/add" for t in THRS))
for y, g in df.groupby("year"):
    parts = []
    for t in THRS:
        fire = g.z_cont6 > t
        add = fire & ~g.buy_union
        parts.append(f"{int(fire.sum()):>6}/{int(add.sum()):>5}")
    print(f"{y:>5} {len(g):>7} {int(g.buy_union.sum()):>7} | " + " | ".join(parts))

print("\n=== pooled ===")
for t in THRS:
    fire = df.z_cont6 > t
    add = fire & ~df.buy_union
    add2426 = add & (df.year >= 2024)
    fire2426 = fire & (df.year >= 2024)
    print(f"z>{t:>5}: fire {int(fire.sum()):>6} ({fire.mean()*100:.1f}%) | add {int(add.sum()):>6} "
          f"| add 2024-26 {int(add2426.sum()):>5} (tren {int((df.year >= 2024).sum())} bars, "
          f"fire 2024-26 {int(fire2426.sum())})")

# do doi: trong vung 2024-26, bao nhieu add-bar la NGAY MOI (khong co buy_union trong +-3 bar
# cung ma) — proxy cho "tin hieu that su moi" thay vi vien quanh cum cu
sub = df[df.year >= 2024].sort_values(["symbol", "date"]).copy()
g = sub.groupby("symbol", sort=False)
sub["union_near"] = (g.buy_union.transform(lambda s: s.rolling(7, center=True, min_periods=1).max()) > 0)
for t in THRS:
    add = (sub.z_cont6 > t) & ~sub.buy_union
    fresh = add & ~sub.union_near
    print(f"z>{t:>5} @2024-26: add {int(add.sum()):>5}, trong do FRESH (khong union +-3bar): "
          f"{int(fresh.sum()):>5} ({0 if add.sum()==0 else fresh.sum()/add.sum()*100:.0f}%)")

# fresh bars roi vao ma/thang nao (z>0.7)
add = (sub.z_cont6 > 0.7) & ~sub.buy_union & ~sub.union_near
f = sub[add]
print("\nfresh add-bars z>0.7 @2024-26 theo thang:",
      f.groupby(f.date.dt.to_period("M")).size().to_dict())
print("theo ma (top 15):", f.symbol.value_counts().head(15).to_dict())
