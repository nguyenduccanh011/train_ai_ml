# -*- coding: utf-8 -*-
"""pr4_21: T2 — (a) LOYO 7 nam per-perm (phuong phap pr3_40) mh16 vs gb, ca 2 che do;
(b) drop-top-10/20 trade CA HAI ben (full + f22, adv + noadv full)."""
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
N = 20


def prop_delta(ma, sa, mg, sg):
    d = (ma / mg - 1) * 100
    sd = math.sqrt((sa / mg) ** 2 + (sg * ma / mg ** 2) ** 2) * 100
    return d, sd


for mode, fee in (("adv", 0.0008), ("noadv", None)):
    sims = {"mh16": NavSim2S(MH16), "gb": NavSim2S(GB)}
    yr = {k: [] for k in sims}
    for k, sim in sims.items():
        for seed in range(N):
            ns = sim.run_series(K=25, advance_fee=fee, order_seed=seed)
            yr[k].append(yearly_returns(ns))
    years = sorted(yr["mh16"][0].keys())
    print(f"\n=== LOYO {mode} (yearly ret mean mh16|gb|delta diem %) ===")
    for y in years:
        a = statistics.mean(d[y] for d in yr["mh16"]) * 100
        b = statistics.mean(d[y] for d in yr["gb"]) * 100
        print(f"  {y}: {a:+.1f} | {b:+.1f} | {a-b:+.1f}")
    n_pass = 0
    for y in ["none"] + years:
        navs = {}
        for k in sims:
            vals = []
            for d in yr[k]:
                p = 1.0
                for yy in years:
                    if yy != y:
                        p *= 1 + d[yy]
                vals.append(p)
            navs[k] = (statistics.mean(vals), statistics.pstdev(vals))
        (ma, sa), (mg, sg) = navs["mh16"], navs["gb"]
        d, sd = prop_delta(ma, sa, mg, sg)
        tag = "PASS>=2sd" if d / sd >= 2 else ("~1-2sd" if d / sd >= 1 else "FAIL<1sd")
        if y != "none":
            n_pass += d / sd >= 2
        print(f"  bo {y}: mh16 x{ma:.2f}±{sa:.2f} vs gb x{mg:.2f}±{sg:.2f} -> "
              f"{d:+.1f}%±{sd:.1f} ({d/sd:+.1f}sd) {tag}", flush=True)
    print(f"  LOYO >=2sd: {n_pass}/{len(years)}")


def drop_top(csv, k):
    t = pd.read_csv(csv)
    t = t.sort_values("pnl_pct", ascending=False).iloc[k:]
    f = os.path.join(tempfile.gettempdir(), f"pr4_drop{k}_" + os.path.basename(csv))
    t.to_csv(f, index=False)
    return f


print("\n=== drop-top CA HAI ben ===")
for k in (10, 20):
    for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
        for mode, fee in (("adv", 0.0008), ("noadv", None)):
            if tag == "f22" and mode == "noadv":
                continue
            sa = shuffle_stats(NavSim2(drop_top(MH16, k), lo), K=25, advance_fee=fee, n=20)
            sg = shuffle_stats(NavSim2(drop_top(GB, k), lo), K=25, advance_fee=fee, n=20)
            d, sd = prop_delta(sa["mean"], sa["sd"], sg["mean"], sg["sd"])
            print(f"  drop{k} {tag} {mode}: mh16 x{sa['mean']:.2f}±{sa['sd']:.2f} "
                  f"gb x{sg['mean']:.2f}±{sg['sd']:.2f} -> {d:+.1f}%±{sd:.1f} "
                  f"({d/sd:+.1f}sd)", flush=True)
print("PR4_21_DONE")
