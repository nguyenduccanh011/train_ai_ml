# -*- coding: utf-8 -*-
"""hb_21: lat recent f23/f25 cho 3 cot (s42) vs gb re-dump — bo sung hb_20."""
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

CSVS = {
    "t2943": str(HERE / "pr5_dynclean_mh25_s42_trades.csv"),
    "t2936": str(HERE / "ab_noT_s42_trades.csv"),
    "dyn25": str(HERE / "xg_dyn_mh25_s42_trades.csv"),
    "gb": str(HERE / "gb_t2783_s42_trades.csv"),
}


def delta(sa, sg):
    d = (sa["mean"] / sg["mean"] - 1) * 100
    sd = math.sqrt((sa["sd"] / sg["mean"]) ** 2
                   + (sg["sd"] * sa["mean"] / sg["mean"] ** 2) ** 2) * 100
    return f"{d:+.1f}%±{sd:.1f} ({d/sd:+.1f}sd)"


for lo, tag in (("2023-01-01", "f23"), ("2025-01-01", "f25")):
    stats = {}
    for name, csv in CSVS.items():
        sim = NavSim2(csv, date_lo=lo)
        stats[name] = {
            "adv": shuffle_stats(sim, K=25, advance_fee=0.0008, n=20),
            "noadv": shuffle_stats(sim, K=25, advance_fee=None, n=20)}
    for mode in ("adv", "noadv"):
        g = stats[mode] if False else stats["gb"][mode]
        print(f"-- {tag} {mode} (gb x{g['mean']:.2f}±{g['sd']:.2f} "
              f"DDw {g['dd_worst']*100:.1f}%)", flush=True)
        for name in ("t2943", "t2936", "dyn25"):
            s = stats[name][mode]
            print(f"  {name:6s} x{s['mean']:.2f}±{s['sd']:.2f} "
                  f"DDw {s['dd_worst']*100:.1f}%  vs gb {delta(s, g)}", flush=True)
print("HB_21_DONE", flush=True)
