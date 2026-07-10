# -*- coding: utf-8 -*-
"""nh_02: cham diem tot nhat cua worker r2b qua sim chuan v2.

Ung vien (doc tu r2_nav_r2b_*.csv, alphabet K25 khung cu):
  - r2b_oxtrail04: full x16.76 / f22 x3.89 — combo tot nhat song dieu kien f22
  - r2b_p40_45  : full x16.98 (chua co f22 tu worker)
Chi DOC csv trades cua ho; khong dung file ho dang ghi.

Buoc 1: sanity legacy (lag0/alphabet/R0.6) phai ~ khop NAV da cong bo cua ho.
Buoc 2: frontier v2 (20 perm, R0.6, lag2-noadv & lag2-adv0.08%), K22/K25, full+f22.
Delta vs gb K25 lay tu nh_frontier_results.csv (cung che do).
"""
import csv
import math
import sys

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats, R2DIR

ADV = 0.0008
CANDS = [("oxtrail04", f"{R2DIR}/r2b_oxtrail04_s42_trades.csv", 16.757, 3.891),
         ("p40_45", f"{R2DIR}/r2b_p40_45_s42_trades.csv", 16.982, None)]

# gb reference tu frontier
gb = {}
with open(f"{R2DIR}/na_audit/nh_frontier_results.csv") as f:
    for r in csv.DictReader(f):
        if r["sys"] == "gb":
            gb[r["mode"]] = {k: float(r[k]) for k in
                             ("full_mean", "full_sd", "f22_mean", "f22_sd")}


def delta(mc, sc, mg, sg):
    d = (mc / mg - 1) * 100
    sd = (mc / mg) * math.sqrt((sc / mc) ** 2 + (sg / mg) ** 2) * 100
    return d, sd


for name, csv_p, exp_full, exp_f22 in CANDS:
    print(f"\n===== r2b_{name} =====", flush=True)
    sim_full = NavSim2(csv_p, date_lo="2020-01-01")
    sim_f22 = NavSim2(csv_p, date_lo="2022-01-01")
    # sanity legacy
    m = sim_full.run(K=25, roundtrip=0.006, settle_lag=0, advance_fee=None)
    m22 = sim_f22.run(K=25, roundtrip=0.006, settle_lag=0, advance_fee=None)
    print(f"legacy K25 lag0 alphabet: full x{m['final']:.2f} (worker x{exp_full:.2f})"
          f" | f22 x{m22['final']:.2f}"
          + (f" (worker x{exp_f22:.2f})" if exp_f22 else " (worker chua co)"),
          flush=True)
    # frontier v2
    for mode, kw in [("lag2-noadv", dict(settle_lag=2, advance_fee=None)),
                     ("lag2-adv", dict(settle_lag=2, advance_fee=ADV))]:
        g = gb[mode]
        for K in (22, 25):
            stf = shuffle_stats(sim_full, K=K, roundtrip=0.006, **kw)
            st2 = shuffle_stats(sim_f22, K=K, roundtrip=0.006, **kw)
            df, sf = delta(stf["mean"], stf["sd"], g["full_mean"], g["full_sd"])
            d2, s2 = delta(st2["mean"], st2["sd"], g["f22_mean"], g["f22_sd"])
            print(f"{mode:10s} K{K}: full x{stf['mean']:.2f}±{stf['sd']:.2f}"
                  f" DD {stf['dd_mean']*100:.2f}% (vs gb {df:+.1f}%±{sf:.1f})"
                  f" | f22 x{st2['mean']:.2f}±{st2['sd']:.2f}"
                  f" (vs gb {d2:+.1f}%±{s2:.1f})", flush=True)
print("\nDONE", flush=True)
