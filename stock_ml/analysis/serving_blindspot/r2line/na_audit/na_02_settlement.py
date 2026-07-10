# -*- coding: utf-8 -*-
"""na_02: settlement T+2.5 — tien ban ve sau N ngay giao dich moi xai duoc.
r2_nav.py goc: lag=0 (tien ban dung NGAY trong ngay ban). VN thuc te:
ban T, tien ve chieu T+2 (co the ung truoc mat phi). Quet lag {0,1,2,3}.
"""
import sys

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from na_navlib import run_sim

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
C2 = f"{R2}/r2_c2_pb40snr_s42_trades.csv"
BASE = f"{R2}/r2_base_s42_trades.csv"

systems = [("c2 K22", C2, 22), ("c2 K25", C2, 25),
           ("base K20", BASE, 20), ("base K25", BASE, 25), ("gb K25", GB, 25)]

for lo, tag in [("2020-01-01", "full"), ("2022-01-01", "f22")]:
    print(f"=== SETTLEMENT SWEEP ({tag}) ===")
    base_nav = {}
    for name, csv, k in systems:
        outs = []
        for lag in [0, 1, 2, 3]:
            m = run_sim(csv, K=k, date_lo=lo, settle_lag=lag)
            outs.append((lag, m))
        base_nav[name] = outs[0][1]["final"]
        cells = []
        for lag, m in outs:
            d = m["final"] / outs[0][1]["final"] - 1
            cells.append(f"lag{lag}: x{m['final']:6.2f} ({d*100:+5.1f}%) DD{m['maxdd']*100:6.2f}% fill{m['fills']}")
        print(name.ljust(9) + " | " + " | ".join(cells))
    print()

# loi the c2-vs-gb sau khi ca hai cung chiu lag 2
print("=== DELTA c2-vs-gb theo lag (full frame) ===")
for lag in [0, 2]:
    c = run_sim(C2, K=22, settle_lag=lag)["final"]
    g = run_sim(GB, K=25, settle_lag=lag)["final"]
    print(f"lag={lag}: c2K22 x{c:.2f} vs gbK25 x{g:.2f} -> delta {(c/g-1)*100:+.1f}%")
for lag in [0, 2]:
    c = run_sim(C2, K=22, date_lo="2022-01-01", settle_lag=lag)["final"]
    g = run_sim(GB, K=25, date_lo="2022-01-01", settle_lag=lag)["final"]
    print(f"f22 lag={lag}: c2K22 x{c:.2f} vs gbK25 x{g:.2f} -> delta {(c/g-1)*100:+.1f}%")
