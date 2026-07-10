# -*- coding: utf-8 -*-
"""exitmap step 2: (1) anatomy top-50 lenh thua + knife signature + tail u,
(2) fill-age buckets x nam, (3) cau truc xo so / concentration, (5) trend theo nam."""
import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap"
df = pd.read_csv(f"{OUT}\\gbx08_enriched.csv")
FEATS = ["runup20", "runup60", "dist_ma20", "dist_ma60", "snr_sym", "vol_shock",
         "fill_age", "depth10", "mae", "mfe", "hold"]

total_u = df.pnl.sum()
print(f"TOTAL: n={len(df)} u={total_u:.2f} WR={(df.pnl>0).mean():.3f} mean={df.pnl.mean():.4f}")

# ============ 1. TOP-50 LENH THUA ============
w = df.nsmallest(50, "pnl").copy()
print("\n===== 1. TOP-50 THUA NANG NHAT =====")
print("u top50 losers:", round(w.pnl.sum(), 2), " theo nam:")
print(w.groupby("year").agg(n=("pnl", "size"), u=("pnl", "sum")).to_string())
print("\nmedian dac trung: top50-losers vs top50-winners vs ALL")
best = df.nlargest(50, "pnl")
cmp = pd.DataFrame({"losers50": w[FEATS].median(), "winners50": best[FEATS].median(),
                    "all": df[FEATS].median()})
print(cmp.round(4).to_string())

# percentile cua tung loser tren phan phoi toan bo (knife check)
for f in ["runup20", "runup60", "dist_ma20", "vol_shock", "snr_sym", "fill_age"]:
    pct = w[f].apply(lambda v: (df[f] < v).mean())
    print(f"loser50 {f}: median-pctile={pct.median():.2f} | >p80: {(pct>0.8).sum()}/50 | <p20: {(pct<0.2).sum()}/50")

# quintile pnl theo runup20/60 (toan mau)
print("\nquintile PnL theo run-up truoc tin hieu (toan bo 1378):")
for f in ["runup20", "runup60", "dist_ma20", "vol_shock", "snr_sym"]:
    q = pd.qcut(df[f], 5, labels=False, duplicates="drop")
    t = df.groupby(q).agg(n=("pnl", "size"), u=("pnl", "sum"), mean=("pnl", "mean"),
                          WR=("pnl", lambda x: (x > 0).mean()),
                          p10=("pnl", lambda x: x.quantile(0.1)))
    print(f"-- {f}:")
    print(t.round(3).to_string())

# tail <= -15%
tail = df[df.pnl <= -0.15]
neg = df[df.pnl < 0]
print(f"\nTAIL <=-15%: n={len(tail)} u={tail.pnl.sum():.2f} | tong u am={neg.pnl.sum():.2f}"
      f" -> tail chiem {tail.pnl.sum()/neg.pnl.sum()*100:.1f}% u am; = {abs(tail.pnl.sum())/total_u*100:.1f}% tong u duong rong")
print("tail theo nam:")
print(tail.groupby("year").agg(n=("pnl", "size"), u=("pnl", "sum")).to_string())
print("tail <=-20%: n=", (df.pnl <= -0.20).sum(), " u=", round(df[df.pnl <= -0.20].pnl.sum(), 2))

# do sau thuc te sau fill cua losers
print("\ntop50 losers: depth10 median", round(w.depth10.median(), 3),
      "| mae median", round(w.mae.median(), 3), "| mfe median", round(w.mfe.median(), 3),
      "| hold median", w.hold.median())
print("trong top50, so lenh MFE >= +5% truoc khi chet:", int((w.mfe >= 0.05).sum()))
print("trong top50, so lenh fill_age >= 16:", int((w.fill_age >= 16).sum()),
      "| fill_age 1-5:", int((w.fill_age <= 5).sum()))

# ============ 2. FILL-AGE BUCKETS ============
print("\n===== 2. CHAT LUONG FILL THEO VI TRI WINDOW =====")
df["age_b"] = pd.cut(df.fill_age, [0, 5, 15, 40], labels=["d1-5", "d6-15", "d16-40"])
t = df.groupby("age_b", observed=True).agg(
    n=("pnl", "size"), u=("pnl", "sum"), mean=("pnl", "mean"),
    WR=("pnl", lambda x: (x > 0).mean()), hold=("hold", "median"),
    mfe=("mfe", "median"), mae=("mae", "median"), p10=("pnl", lambda x: x.quantile(0.1)))
print(t.round(4).to_string())
print("\nfill-age x nam (mean pnl):")
pv = df.pivot_table(index="year", columns="age_b", values="pnl", aggfunc=["size", "sum", "mean"], observed=True)
print(pv.round(3).to_string())
# fine-grained
fine = pd.cut(df.fill_age, [0, 2, 5, 10, 15, 25, 40])
print("\nfine buckets:")
print(df.groupby(fine, observed=True).agg(n=("pnl", "size"), u=("pnl", "sum"), mean=("pnl", "mean"),
      WR=("pnl", lambda x: (x > 0).mean())).round(4).to_string())

# ============ 3. CAU TRUC XO SO ============
print("\n===== 3. CONCENTRATION =====")
s = df.pnl.sort_values(ascending=False).reset_index(drop=True)
for topn in [14, 28, 69, 138, 276]:  # ~1,2,5,10,20%
    print(f"top {topn} lenh ({topn/len(df)*100:.0f}%): u={s[:topn].sum():.1f} = {s[:topn].sum()/total_u*100:.1f}% tong")
pos = s[s > 0]
print(f"lenh duong: {len(pos)} ({len(pos)/len(df)*100:.0f}%), gross+ = {pos.sum():.1f}, gross- = {s[s<0].sum():.1f}")
# lenh >= 100%?
print("lenh pnl>=+50%:", (df.pnl >= 0.5).sum(), "u=", round(df[df.pnl >= 0.5].pnl.sum(), 1),
      "| >=+100%:", (df.pnl >= 1.0).sum(), "u=", round(df[df.pnl >= 1.0].pnl.sum(), 1))

def gini(x):
    x = np.sort(x - x.min() + 1e-9)
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())

print("\ntheo nam: n, u, %u tu top-10% lenh, gini(pnl shifted):")
for y, g in df.groupby("year"):
    ss = g.pnl.sort_values(ascending=False)
    k = max(1, int(len(g) * 0.1))
    print(f"{y}: n={len(g):4d} u={g.pnl.sum():7.2f} top10%={ss[:k].sum():7.2f}"
          f" ({ss[:k].sum()/g.pnl.sum()*100 if g.pnl.sum()>0 else float('nan'):5.1f}%)"
          f" gini={gini(g.pnl.to_numpy()):.3f}")

# per-symbol
print("\nper-symbol net (61 ma):")
sy = df.groupby("symbol").agg(n=("pnl", "size"), u=("pnl", "sum"),
                              WR=("pnl", lambda x: (x > 0).mean()),
                              years=("year", "nunique"),
                              neg_years=("pnl", "size"))
yr_net = df.groupby(["symbol", "year"]).pnl.sum().unstack()
sy["neg_years"] = (yr_net < 0).sum(axis=1)
sy["pos_years"] = (yr_net > 0).sum(axis=1)
sy = sy.sort_values("u")
print("BOTTOM-12 ma:")
print(sy.head(12).round(2).to_string())
print("TOP-8 ma:")
print(sy.tail(8).round(2).to_string())
chron = sy[(sy.u < 0) & (sy.neg_years >= 2)]
print(f"\nma am rong: {(sy.u<0).sum()}/{len(sy)}, u am tong={sy[sy.u<0].u.sum():.1f};"
      f" ma am >=2 nam: {len(chron)} (u={chron.u.sum():.1f})")
k = max(1, int(len(sy) * 0.1))
print(f"top-{k} ma chiem {sy.u.nlargest(k).sum()/total_u*100:.1f}% tong u")

# ============ 5. TREND THEO NAM ============
print("\n===== 5. HINH THAI THEO NAM =====")
t = df.groupby("year").agg(n=("pnl", "size"), u=("pnl", "sum"), mean=("pnl", "mean"),
                           WR=("pnl", lambda x: (x > 0).mean()), hold=("hold", "median"),
                           mfe=("mfe", "median"), mae=("mae", "median"),
                           p90=("pnl", lambda x: x.quantile(0.9)),
                           big=("pnl", lambda x: (x >= 0.3).sum()))
t["PF"] = df.groupby("year").pnl.apply(lambda x: x[x > 0].sum() / abs(x[x < 0].sum()))
print(t.round(4).to_string())
