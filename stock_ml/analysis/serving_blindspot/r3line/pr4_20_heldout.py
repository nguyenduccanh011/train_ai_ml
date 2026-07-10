# -*- coding: utf-8 -*-
"""pr4_20: T2 selection inflation — cham NAV cac lat HELD-OUT (f21/f24/f25,
chua dung de chon) cho plateau mh12/mh16/mh25 + gb_mh16 + baselines gb/r2c.
Ca hai che do (adv 0.08% / noadv). Delta vs gb kem sd lan truyen."""
import math
import sys

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

R3 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line"
R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"

CSVS = {
    "mh12": f"{R3}/r3_mh12_s42_trades.csv",
    "mh16": f"{R3}/r3_mh16_s42_trades.csv",
    "mh25": f"{R3}/r3_mh25_s42_trades.csv",
    "gbmh16": f"{R3}/r3_gb_mh16_s42_trades.csv",
    "r2c": f"{R2}/r2c_oxt04_p42_s42_trades.csv",
    "gb": GB,
}
SLICES = [("2021-01-01", "f21"), ("2022-01-01", "f22"), ("2023-01-01", "f23"),
          ("2024-01-01", "f24"), ("2025-01-01", "f25")]


def delta(sa, sg):
    d = (sa["mean"] / sg["mean"] - 1) * 100
    sd = math.sqrt((sa["sd"] / sg["mean"]) ** 2
                   + (sg["sd"] * sa["mean"] / sg["mean"] ** 2) ** 2) * 100
    z = d / sd if sd > 0 else float("inf")
    return f"{d:+.1f}%±{sd:.1f} ({z:+.1f}sd)"


for lo, tag in SLICES:
    stats = {}
    for name, csv in CSVS.items():
        sim = NavSim2(csv, date_lo=lo)
        stats[name] = {
            "adv": shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2,
                                 advance_fee=0.0008, n=20),
            "noadv": shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2,
                                   advance_fee=None, n=20)}
    for mode in ("adv", "noadv"):
        g = stats["gb"][mode]
        print(f"\n== {tag} {mode}  (gb x{g['mean']:.2f}±{g['sd']:.2f} "
              f"DDw {g['dd_worst']*100:.1f}%)")
        for name in ("mh12", "mh16", "mh25", "gbmh16", "r2c"):
            s = stats[name][mode]
            print(f"  {name:7s} x{s['mean']:.2f}±{s['sd']:.2f} "
                  f"DDw {s['dd_worst']*100:.1f}%  vs gb {delta(s, g)}", flush=True)
print("PR4_20_DONE")
