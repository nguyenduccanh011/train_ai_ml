# -*- coding: utf-8 -*-
"""r3b_checks: kiem nhanh combo — LOYO-2021, drop-top-20, so t2907 (mh16) va gb.

Usage: python r3b_checks.py <combo_trades_csv>
"""
import math
import os
import statistics
import sys
import tempfile

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/pr3")
sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from pr3_lib import NavSim2S, yearly_returns  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

R3 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
MH16 = f"{R3}/r3_mh16_s42_trades.csv"
COMBO = sys.argv[1]
N = 20


def prop_delta(ma, sa, mg, sg):
    d = (ma / mg - 1) * 100
    sd = math.sqrt((sa / mg) ** 2 + (sg * ma / mg ** 2) ** 2) * 100
    return d, sd


# --- LOYO-2021 nhanh (bo nam 2021, per-perm compound) ca 2 che do, so gb + mh16
for mode, fee in (("adv", 0.0008), ("noadv", None)):
    sims = {"combo": NavSim2S(COMBO), "mh16": NavSim2S(MH16), "gb": NavSim2S(GB)}
    yr = {}
    for k, sim in sims.items():
        yr[k] = [yearly_returns(sim.run_series(K=25, advance_fee=fee, order_seed=s))
                 for s in range(N)]
    years = sorted(yr["combo"][0].keys())
    navs = {}
    for k in sims:
        vals = []
        for d in yr[k]:
            p = 1.0
            for yy in years:
                if yy != 2021:
                    p *= 1 + d[yy]
            vals.append(p)
        navs[k] = (statistics.mean(vals), statistics.pstdev(vals))
    (mc, sc), (mm, sm), (mg, sg) = navs["combo"], navs["mh16"], navs["gb"]
    dg, sdg = prop_delta(mc, sc, mg, sg)
    dm, sdm = prop_delta(mc, sc, mm, sm)
    print(f"LOYO-2021 {mode}: combo x{mc:.2f}±{sc:.2f} | vs gb {dg:+.1f}%±{sdg:.1f} "
          f"({dg/sdg:+.1f}sd) | vs mh16 {dm:+.1f}%±{sdm:.1f} ({dm/sdm:+.1f}sd)", flush=True)


def drop_top(csv, k):
    t = pd.read_csv(csv)
    t = t.sort_values("pnl_pct", ascending=False).iloc[k:]
    f = os.path.join(tempfile.gettempdir(), f"r3b_drop{k}_" + os.path.basename(csv))
    t.to_csv(f, index=False)
    return f


for mode, fee in (("adv", 0.0008), ("noadv", None)):
    sc_ = shuffle_stats(NavSim2(drop_top(COMBO, 20), "2020-01-01"), K=25, advance_fee=fee, n=20)
    sg_ = shuffle_stats(NavSim2(drop_top(GB, 20), "2020-01-01"), K=25, advance_fee=fee, n=20)
    sm_ = shuffle_stats(NavSim2(drop_top(MH16, 20), "2020-01-01"), K=25, advance_fee=fee, n=20)
    dg, sdg = prop_delta(sc_["mean"], sc_["sd"], sg_["mean"], sg_["sd"])
    dm, sdm = prop_delta(sc_["mean"], sc_["sd"], sm_["mean"], sm_["sd"])
    print(f"drop20 full {mode}: combo x{sc_['mean']:.2f}±{sc_['sd']:.2f} | vs gb {dg:+.1f}% "
          f"({dg/sdg:+.1f}sd) | vs mh16 {dm:+.1f}% ({dm/sdm:+.1f}sd)", flush=True)
print("R3B_CHECKS_DONE")
