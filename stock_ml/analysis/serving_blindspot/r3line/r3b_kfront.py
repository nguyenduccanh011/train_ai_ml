# -*- coding: utf-8 -*-
"""r3b_kfront: K-frontier {18,20,22,25,28} cho tam tot nhat (chi NAV sim lai).

Bao NAV 2 che do + DDw moi K; danh dau K matched-DD voi gb (-15.3% full).
Usage: python r3b_kfront.py <trades_csv> [<trades_csv2> ...]
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r2line" / "na_audit"))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

GB_DD = -15.3  # % full, matched-DD target

for csv in sys.argv[1:]:
    name = Path(csv).stem
    for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
        sim = NavSim2(csv, date_lo=lo)
        for K in (18, 20, 22, 25, 28):
            a = shuffle_stats(sim, K=K, roundtrip=0.006, settle_lag=2,
                              advance_fee=0.0008, n=20)
            n = shuffle_stats(sim, K=K, roundtrip=0.006, settle_lag=2,
                              advance_fee=None, n=20)
            mark = ""
            if tag == "full" and abs(a["dd_worst"] * 100 - GB_DD) <= 1.0:
                mark = "  <= matched-DD gb"
            print(f"KF {name} {tag} K{K}: adv x{a['mean']:.2f}±{a['sd']:.2f} "
                  f"DDw {a['dd_worst']*100:.1f}% | noadv x{n['mean']:.2f}±{n['sd']:.2f} "
                  f"DDw {n['dd_worst']*100:.1f}%{mark}", flush=True)
print("KFRONT_DONE")
