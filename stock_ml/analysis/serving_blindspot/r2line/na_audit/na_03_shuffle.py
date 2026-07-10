# -*- coding: utf-8 -*-
"""na_03: slot-assignment bias — r2_nav.py sort entry cung ngay theo ALPHABET
symbol roi fill den khi het cash. Xao thu tu 20 seed -> phan phoi NAV.
Kem: kich ban to hop lag2 + phi 0.9% (stress thuc te).
"""
import statistics
import sys

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from na_navlib import run_sim

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
C2 = f"{R2}/r2_c2_pb40snr_s42_trades.csv"
BASE = f"{R2}/r2_base_s42_trades.csv"

for name, csv, k, lo in [("c2 K22 full", C2, 22, "2020-01-01"),
                         ("c2 K25 full", C2, 25, "2020-01-01"),
                         ("c2 K22 f22", C2, 22, "2022-01-01"),
                         ("base K25 full", BASE, 25, "2020-01-01"),
                         ("gb K25 full", GB, 25, "2020-01-01")]:
    alpha = run_sim(csv, K=k, date_lo=lo)["final"]
    navs, dds = [], []
    for seed in range(20):
        m = run_sim(csv, K=k, date_lo=lo, order_seed=seed)
        navs.append(m["final"])
        dds.append(m["maxdd"])
    mean, sd = statistics.mean(navs), statistics.pstdev(navs)
    print(f"{name:14s} alpha=x{alpha:.2f} | shuffle20: mean=x{mean:.2f} sd={sd:.2f}"
          f" min=x{min(navs):.2f} max=x{max(navs):.2f}"
          f" span={(max(navs)-min(navs))/mean*100:.1f}% | alpha-vs-mean {(alpha/mean-1)*100:+.1f}%"
          f" | DD range [{min(dds)*100:.1f}..{max(dds)*100:.1f}]%")

print("\n=== KICH BAN TO HOP (phi + settlement cung luc) ===")
scen = [("goc (R0.6 lag0)", 0.006, 0),
        ("R0.7 lag0 (ung truoc tien ban ~0.08%)", 0.007, 0),
        ("R0.9 lag2", 0.009, 2),
        ("R1.1 lag2", 0.011, 2)]
for label, R, lag in scen:
    c = run_sim(C2, K=22, roundtrip=R, settle_lag=lag)
    g = run_sim(GB, K=25, roundtrip=R, settle_lag=lag)
    c22 = run_sim(C2, K=22, date_lo="2022-01-01", roundtrip=R, settle_lag=lag)
    g22 = run_sim(GB, K=25, date_lo="2022-01-01", roundtrip=R, settle_lag=lag)
    print(f"{label:38s} c2K22 x{c['final']:6.2f}/DD{c['maxdd']*100:6.2f} vs gb x{g['final']:6.2f}/DD{g['maxdd']*100:6.2f}"
          f" -> delta {(c['final']/g['final']-1)*100:+6.1f}% | f22 {(c22['final']/g22['final']-1)*100:+6.1f}%")
