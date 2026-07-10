# -*- coding: utf-8 -*-
"""na_06: kich ban van hanh trung thuc — go het 3 lop bias cung luc:
shuffle-mean (20 seed) x phi {0.6,0.7,0.9} x lag {0,2}; c2 K22/K25 vs gb K25.
"""
import statistics
import sys

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from na_navlib import run_sim

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
C2 = f"{R2}/r2_c2_pb40snr_s42_trades.csv"


def smean(csv, k, lo, R, lag, n=20):
    return statistics.mean(
        run_sim(csv, K=k, date_lo=lo, roundtrip=R, settle_lag=lag, order_seed=s)["final"]
        for s in range(n))


for lo, tag in [("2020-01-01", "full"), ("2022-01-01", "f22 ")]:
    print(f"=== shuffle-mean 20 seed ({tag}) ===")
    for R, lag, lbl in [(0.006, 0, "R0.6 lag0 (khung goc)"),
                        (0.007, 0, "R0.7 lag0 (ung truoc TB)"),
                        (0.007, 2, "R0.7 lag2 (khong ung)  "),
                        (0.009, 0, "R0.9 lag0 (phi day)    "),
                        (0.009, 2, "R0.9 lag2 (xau nhat)   ")]:
        c22 = smean(C2, 22, lo, R, lag)
        c25 = smean(C2, 25, lo, R, lag)
        g = smean(GB, 25, lo, R, lag)
        print(f"{lbl} c2K22 x{c22:6.2f} ({(c22/g-1)*100:+5.1f}%) | c2K25 x{c25:6.2f}"
              f" ({(c25/g-1)*100:+5.1f}%) | gb x{g:6.2f}")
