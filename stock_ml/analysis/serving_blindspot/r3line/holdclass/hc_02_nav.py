# -*- coding: utf-8 -*-
"""hc_02_nav: cham NAV K25 (nh_nav2, 2 che do chuan v2) cho oracle hold-class
vs baseline mh16. Shuffle-mean 20 permutation.

Che do: (A) settle_lag=2, khong ung truoc; (B) advance_fee=0.0008.
Roundtrip 0.006. Full (2020) va slice >=2022.
"""
import sys
sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats

BASE = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line"
CSVS = [
    ("mh16 base ", f"{BASE}/r3_mh16_s42_trades.csv"),
    ("oracle h40", f"{BASE}/holdclass/hc_oracle_h40_trades.csv"),
    ("oracle h60", f"{BASE}/holdclass/hc_oracle_h60_trades.csv"),
    ("oracle best", f"{BASE}/holdclass/hc_oracle_best_trades.csv"),
]
MODES = [("A lag2", dict(settle_lag=2, advance_fee=None)),
         ("B adv08", dict(settle_lag=2, advance_fee=0.0008))]

for lo in ["2020-01-01", "2022-01-01"]:
    print(f"\n===== date_lo {lo} | K25 R0.006 | shuffle-mean n=20 =====")
    base_res = {}
    for name, csv in CSVS:
        sim = NavSim2(csv, date_lo=lo)
        line = f"{name}:"
        for mname, kw in MODES:
            st = shuffle_stats(sim, K=25, roundtrip=0.006, n=20, **kw)
            key = (mname, lo)
            if name.startswith("mh16"):
                base_res[key] = st["mean"]
            rel = st["mean"] / base_res[key] - 1 if key in base_res else float("nan")
            line += (f"  [{mname}] x{st['mean']:.2f}±{st['sd']:.2f} "
                     f"DD{st['dd_mean']*100:.1f}% ({rel*100:+.1f}%)")
        print(line, f" skipped_db={sim.skipped_db}")
