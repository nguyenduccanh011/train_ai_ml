# -*- coding: utf-8 -*-
"""nh_01: FRONTIER CHINH THUC tren sim chuan v2 (nh_nav2) + bang quyet dinh.

Frontier: shuffle-mean±sd (20 perm), R0.6, hai che do:
  - lag2-noadv : settle T+2, KHONG ung truoc tien ban
  - lag2-adv   : ung truoc tien ban, phi 0.08%/vong (tien xai ngay)
Diem: c2 {K20,K22,K23,K25}, base {K20,K25}, gb {K25}; full-frame + f22.

Bang quyet dinh 3 kich ban x (full, f22): c2-vs-gb delta% ± sd (propagated).
"""
import csv
import math
import sys

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats, CSV_C2, CSV_BASE, CSV_GB

ADV = 0.0008  # phi ung truoc tien ban /vong

print("Load sims...", flush=True)
sims = {}
for tag, csv_p in [("c2", CSV_C2), ("base", CSV_BASE), ("gb", CSV_GB)]:
    for frame, lo in [("full", "2020-01-01"), ("f22", "2022-01-01")]:
        sims[(tag, frame)] = NavSim2(csv_p, date_lo=lo)
        print(f"  {tag} {frame}: {len(sims[(tag,frame)].trades)} trades", flush=True)

POINTS = [("c2", 20), ("c2", 22), ("c2", 23), ("c2", 25),
          ("base", 20), ("base", 25), ("gb", 25)]
MODES = [("lag2-noadv", dict(settle_lag=2, advance_fee=None)),
         ("lag2-adv", dict(settle_lag=2, advance_fee=ADV))]

rows = []
print("\n=== FRONTIER CHUAN v2 (shuffle-mean±sd, 20 perm, R0.6) ===", flush=True)
for mode, kw in MODES:
    print(f"\n--- {mode} ---", flush=True)
    for tag, K in POINTS:
        rec = {"mode": mode, "sys": tag, "K": K}
        for frame in ("full", "f22"):
            st = shuffle_stats(sims[(tag, frame)], K=K, roundtrip=0.006, **kw)
            rec[f"{frame}_mean"] = st["mean"]
            rec[f"{frame}_sd"] = st["sd"]
            rec[f"{frame}_dd"] = st["dd_mean"]
            rec[f"{frame}_ddw"] = st["dd_worst"]
        rows.append(rec)
        print(f"{tag:4s} K{K}: full x{rec['full_mean']:.2f}±{rec['full_sd']:.2f}"
              f" DD {rec['full_dd']*100:.2f}% (worst {rec['full_ddw']*100:.2f}%)"
              f" | f22 x{rec['f22_mean']:.2f}±{rec['f22_sd']:.2f}", flush=True)

out = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit/nh_frontier_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved {out}", flush=True)


def delta(mc, sc, mg, sg):
    d = (mc / mg - 1) * 100
    sd = (mc / mg) * math.sqrt((sc / mc) ** 2 + (sg / mg) ** 2) * 100
    return d, sd


print("\n=== BANG QUYET DINH (c2 vs gb K25, shuffle-mean 20 perm) ===", flush=True)
SCEN = [("ca nhan R0.6 + advance ", 0.006, dict(settle_lag=2, advance_fee=ADV)),
        ("R0.6 khong advance     ", 0.006, dict(settle_lag=2, advance_fee=None)),
        ("von lon R0.9 + advance ", 0.009, dict(settle_lag=2, advance_fee=ADV))]
for lbl, R, kw in SCEN:
    line = lbl
    for frame in ("full", "f22"):
        g = shuffle_stats(sims[("gb", frame)], K=25, roundtrip=R, **kw)
        for K in (22, 25):
            c = shuffle_stats(sims[("c2", frame)], K=K, roundtrip=R, **kw)
            d, sd = delta(c["mean"], c["sd"], g["mean"], g["sd"])
            line += (f" | {frame} K{K}: {d:+.1f}%±{sd:.1f}"
                     f" (c2 x{c['mean']:.2f}, gb x{g['mean']:.2f})")
    print(line, flush=True)
print("DONE", flush=True)
