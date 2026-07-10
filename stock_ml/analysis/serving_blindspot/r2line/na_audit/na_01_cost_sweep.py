# -*- coding: utf-8 -*-
"""na_01: (0) anchor tai lap so da cong bo; (1) cost sweep roundtrip
{0.6 (=hien tai), 0.7, 0.9, 1.1, 1.3, 1.5}% cho c2/base/gb tai cac K then chot,
full-frame va from-2022. In bang + tim phi hoa von loi the R2-vs-gb.
"""
import sys

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from na_navlib import run_sim, fmt

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
C2 = f"{R2}/r2_c2_pb40snr_s42_trades.csv"
BASE = f"{R2}/r2_base_s42_trades.csv"

print("=== ANCHOR (phai khop so cong bo) ===")
anchors = [
    ("c2 K25", C2, 25, "2020-01-01", "x16.34 / -13.63%"),
    ("c2 K22", C2, 22, "2020-01-01", "x18.03 / -15.41%"),
    ("c2 K23", C2, 23, "2020-01-01", "x17.14 / -14.77%"),
    ("base K25", BASE, 25, "2020-01-01", "x14.78 / -12.25%"),
    ("base K20", BASE, 20, "2020-01-01", "x17.70 / -15.21%"),
    ("gb K25", GB, 25, "2020-01-01", "x14.44 / -15.31%"),
    ("c2 K25 f22", C2, 25, "2022-01-01", "x3.80"),
]
for name, csv, k, lo, pub in anchors:
    m = run_sim(csv, K=k, date_lo=lo)
    print(f"{name:12s} pub[{pub:20s}] -> {fmt(m)}")

print("\n=== ANCHOR cost-recompute (roundtrip=0.6% phai ~ y het CSV) ===")
for name, csv, k in [("c2 K22", C2, 22), ("gb K25", GB, 25)]:
    m = run_sim(csv, K=k, roundtrip=0.006)
    print(f"{name:12s} R=0.6% -> {fmt(m)}")

print("\n=== COST SWEEP full-frame ===")
SWEEP = [0.006, 0.007, 0.009, 0.011, 0.013, 0.015]
systems = [("c2 K22", C2, 22), ("c2 K23", C2, 23), ("c2 K25", C2, 25),
           ("base K20", BASE, 20), ("base K25", BASE, 25), ("gb K25", GB, 25)]
res = {}
for name, csv, k in systems:
    row = []
    for R in SWEEP:
        m = run_sim(csv, K=k, roundtrip=R)
        row.append(m)
    res[name] = row
    print(name.ljust(10) + " | " + " | ".join(
        f"R{R*100:.1f}: x{m['final']:6.2f} DD{m['maxdd']*100:6.2f}%" for R, m in zip(SWEEP, row)))
    print(" " * 10 + f"   turnover={row[0]['turnover']:.1f}x/nam")

print("\n=== COST SWEEP from-2022 ===")
res22 = {}
for name, csv, k in systems:
    row = []
    for R in SWEEP:
        m = run_sim(csv, K=k, date_lo="2022-01-01", roundtrip=R)
        row.append(m)
    res22[name] = row
    print(name.ljust(10) + " | " + " | ".join(
        f"R{R*100:.1f}: x{m['final']:5.2f}" for R, m in zip(SWEEP, row)))

print("\n=== BREAKEVEN R2-vs-gb (noi suy tuyen tinh giua cac diem sweep) ===")


def breakeven(a_row, b_row):
    # tim R sao cho NAV_a(R) = NAV_b(R)
    for i in range(len(SWEEP) - 1):
        d0 = a_row[i]["final"] - b_row[i]["final"]
        d1 = a_row[i + 1]["final"] - b_row[i + 1]["final"]
        if d0 > 0 >= d1:
            t = d0 / (d0 - d1)
            return SWEEP[i] + t * (SWEEP[i + 1] - SWEEP[i])
    return None


for a in ["c2 K22", "c2 K23", "c2 K25", "base K20", "base K25"]:
    be = breakeven(res[a], res["gb K25"])
    be22 = breakeven(res22[a], res22["gb K25"])
    print(f"{a:10s} vs gb K25: breakeven full={'%.2f%%' % (be*100) if be else '>1.5% hoac <0.6%'}"
          f" | f22={'%.2f%%' % (be22*100) if be22 else '>1.5% hoac <0.6%'}")
    d06 = res[a][0]["final"] / res["gb K25"][0]["final"] - 1
    d15 = res[a][-1]["final"] / res["gb K25"][-1]["final"] - 1
    print(f"           delta NAV: R0.6 {d06*+100:+.1f}%  ->  R1.5 {d15*100:+.1f}%")
