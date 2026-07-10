# -*- coding: utf-8 -*-
"""pr3_40: tội 4 — leave-one-year-out trên NAV (per-perm, 20 perm):
NAV_loyo = prod(1+r_y) bỏ năm y; delta vs gb cùng cách; còn >=2sd ở mấy/7 năm?
Kèm tội 2: phân rã monthly log-delta p42-gb, top-episode share."""
import sys
import math
import statistics

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/pr3")
from pr3_lib import NavSim2S, yearly_returns  # noqa: E402

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
N = 20

sims = {"p42": NavSim2S(f"{R2}/r2c_oxt04_p42_s42_trades.csv"),
        "gb": NavSim2S(GB)}

# per-perm yearly returns + mean monthly series
yr = {k: [] for k in sims}       # list of dict year->ret per perm
series = {k: [] for k in sims}   # nav series per perm
for k, sim in sims.items():
    for seed in range(N):
        ns = sim.run_series(K=25, advance_fee=0.0008, order_seed=seed)
        yr[k].append(yearly_returns(ns))
        series[k].append(ns.set_index("date")["nav"])

years = sorted(yr["p42"][0].keys())
print("=== yearly returns mean (p42 | gb | delta điểm %) ===")
for y in years:
    a = statistics.mean(d[y] for d in yr["p42"]) * 100
    b = statistics.mean(d[y] for d in yr["gb"]) * 100
    print(f"{y}: {a:+.1f} | {b:+.1f} | {a-b:+.1f}")

print("\n=== LOYO (bỏ năm y, prod các năm còn lại), delta % vs gb ± sd lan truyền ===")
n_pass = 0
for y in ["none"] + years:
    navs = {}
    for k in sims:
        vals = []
        for d in yr[k]:
            p = 1.0
            for yy in years:
                if yy != y:
                    p *= 1 + d[yy]
            vals.append(p)
        navs[k] = (statistics.mean(vals), statistics.pstdev(vals))
    (ma, sa), (mg, sg) = navs["p42"], navs["gb"]
    dlt = (ma / mg - 1) * 100
    sd = math.sqrt((sa / mg) ** 2 + (sg * ma / mg ** 2) ** 2) * 100
    tag = "PASS>=2sd" if dlt / sd >= 2 else ("~" if dlt / sd >= 1 else "FAIL<1sd")
    if y != "none":
        n_pass += dlt / sd >= 2
    print(f"bỏ {y}: p42 x{ma:.2f}±{sa:.2f} vs gb x{mg:.2f}±{sg:.2f} -> "
          f"{dlt:+.1f}%±{sd:.1f} ({dlt/sd:+.1f}sd) {tag}")
print(f"LOYO >=2sd: {n_pass}/7")

# bỏ 2020+2021 cùng lúc (chain yearly, xấp xỉ f22)
navs = {}
for k in sims:
    vals = []
    for d in yr[k]:
        p = 1.0
        for yy in years:
            if yy not in (2020, 2021):
                p *= 1 + d[yy]
        vals.append(p)
    navs[k] = (statistics.mean(vals), statistics.pstdev(vals))
(ma, sa), (mg, sg) = navs["p42"], navs["gb"]
dlt = (ma / mg - 1) * 100
sd = math.sqrt((sa / mg) ** 2 + (sg * ma / mg ** 2) ** 2) * 100
print(f"bỏ 2020+2021 (chain): {dlt:+.1f}%±{sd:.1f} ({dlt/sd:+.1f}sd)")

# ===== tội 2: monthly log-delta trên mean series =====
mean_nav = {}
for k in sims:
    df = pd.concat(series[k], axis=1)
    mean_nav[k] = df.mean(axis=1)
ratio = (mean_nav["p42"] / mean_nav["gb"]).dropna()
lr = ratio.map(math.log)
m = lr.resample("M").last().dropna()
dm = m.diff().dropna()
total = lr.iloc[-1] - lr.iloc[0]
top = dm.abs().sort_values(ascending=False).head(12)
print(f"\n=== tội 2: monthly log-delta (tổng {total:.3f} = {100*(math.exp(total)-1):+.1f}%) ===")
print("top-12 tháng |delta| (log, % of total):")
for dtm, v in top.items():
    sgn = dm[dtm]
    print(f"  {dtm.strftime('%Y-%m')}: {sgn:+.4f} ({100*sgn/total:+.1f}% tổng)")
pos_months = (dm > 0).sum()
print(f"tháng dương: {pos_months}/{len(dm)}")
# top-3 tháng dương gánh bao nhiêu:
top_pos = dm.sort_values(ascending=False).head(3)
print(f"top-3 tháng dương = {100*top_pos.sum()/total:.0f}% tổng log-delta: "
      + ", ".join(f"{d.strftime('%Y-%m')} {v:+.3f}" for d, v in top_pos.items()))
# episode 60 ngày mạnh nhất
roll = lr.diff(60)
mx = roll.idxmax()
print(f"episode 60d mạnh nhất: kết thúc {mx.date()} {roll.max():+.3f} "
      f"({100*roll.max()/total:.0f}% tổng)")
print("DONE")
