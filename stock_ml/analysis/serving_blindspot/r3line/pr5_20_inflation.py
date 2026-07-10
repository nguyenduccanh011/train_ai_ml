# -*- coding: utf-8 -*-
"""pr5_20: CONG TO t2936 (ab_noT) — T1 selection inflation:
(a) held-out f21/f24/f25 (ca 2 che do) cho ab_noT vs gb, kem t2907/gbmh16/dyn25;
(b) f25 tren CA 4 SEED ab_noT (diem yeu mh16 goc — co ke thua khong?);
(c) LOYO 7 nam per-perm ab_noT vs gb ca 2 che do;
(d) drop-top-10/20 ca hai ben (full+f22 adv, full noadv)."""
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
AB = f"{R3}/ab_noT_s42_trades.csv"

CSVS = {
    "ab_noT": AB,
    "t2907": f"{R3}/r3_mh16_s42_trades.csv",
    "gbmh16": f"{R3}/r3_gb_mh16_s42_trades.csv",
    "dyn25": f"{R3}/xg_dyn_mh25_s42_trades.csv",
    "gb": GB,
}


def delta(sa, sg):
    d = (sa["mean"] / sg["mean"] - 1) * 100
    sd = math.sqrt((sa["sd"] / sg["mean"]) ** 2
                   + (sg["sd"] * sa["mean"] / sg["mean"] ** 2) ** 2) * 100
    z = d / sd if sd > 0 else float("inf")
    return f"{d:+.1f}%±{sd:.1f} ({z:+.1f}sd)"


print("=== (a) HELD-OUT f21/f24/f25 (chua dung de chon ab_noT) ===")
for lo, tag in (("2021-01-01", "f21"), ("2024-01-01", "f24"), ("2025-01-01", "f25")):
    stats = {}
    for name, csv in CSVS.items():
        sim = NavSim2(csv, date_lo=lo)
        stats[name] = {
            "adv": shuffle_stats(sim, K=25, advance_fee=0.0008, n=20),
            "noadv": shuffle_stats(sim, K=25, advance_fee=None, n=20)}
    for mode in ("adv", "noadv"):
        g = stats["gb"][mode]
        print(f"\n== {tag} {mode}  (gb x{g['mean']:.2f}±{g['sd']:.2f} "
              f"DDw {g['dd_worst']*100:.1f}%)")
        for name in ("ab_noT", "t2907", "gbmh16", "dyn25"):
            s = stats[name][mode]
            print(f"  {name:7s} x{s['mean']:.2f}±{s['sd']:.2f} "
                  f"DDw {s['dd_worst']*100:.1f}%  vs gb {delta(s, g)}", flush=True)

print("\n=== (b) f25 x 4 SEED ab_noT (vs gb cung frame) ===")
g25 = {m: shuffle_stats(NavSim2(GB, "2025-01-01"), K=25,
                        advance_fee=(0.0008 if m == "adv" else None), n=20)
       for m in ("adv", "noadv")}
for seed in ("s42", "s7", "s99", "s555"):
    csv = AB if seed == "s42" else f"{R3}/ab_noT_{seed}_trades.csv"
    sim = NavSim2(csv, "2025-01-01")
    for m, fee in (("adv", 0.0008), ("noadv", None)):
        s = shuffle_stats(sim, K=25, advance_fee=fee, n=20)
        print(f"  f25 {seed} {m}: x{s['mean']:.2f}±{s['sd']:.2f} DDw {s['dd_worst']*100:.1f}% "
              f"vs gb x{g25[m]['mean']:.2f} (DDw {g25[m]['dd_worst']*100:.1f}%) "
              f"-> {delta(s, g25[m])}", flush=True)


def prop_delta(ma, sa, mg, sg):
    d = (ma / mg - 1) * 100
    sd = math.sqrt((sa / mg) ** 2 + (sg * ma / mg ** 2) ** 2) * 100
    return d, sd


print("\n=== (c) LOYO ab_noT vs gb (per-perm, 2 che do) ===")
for mode, fee in (("adv", 0.0008), ("noadv", None)):
    sims = {"ab": NavSim2S(AB), "gb": NavSim2S(GB)}
    yr = {k: [] for k in sims}
    for k, sim in sims.items():
        for seed in range(20):
            ns = sim.run_series(K=25, advance_fee=fee, order_seed=seed)
            yr[k].append(yearly_returns(ns))
    years = sorted(yr["ab"][0].keys())
    print(f"\n-- LOYO {mode} (yearly ab|gb|delta diem %)")
    for y in years:
        a = statistics.mean(d[y] for d in yr["ab"]) * 100
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
        (ma, sa), (mg, sg) = navs["ab"], navs["gb"]
        d, sd = prop_delta(ma, sa, mg, sg)
        tag = "PASS>=2sd" if d / sd >= 2 else ("~1-2sd" if d / sd >= 1 else "FAIL<1sd")
        if y != "none":
            n_pass += d / sd >= 2
        print(f"  bo {y}: ab x{ma:.2f}±{sa:.2f} vs gb x{mg:.2f}±{sg:.2f} -> "
              f"{d:+.1f}%±{sd:.1f} ({d/sd:+.1f}sd) {tag}", flush=True)
    print(f"  LOYO >=2sd: {n_pass}/{len(years)}")


def drop_top(csv, k):
    t = pd.read_csv(csv)
    t = t.sort_values("pnl_pct", ascending=False).iloc[k:]
    f = os.path.join(tempfile.gettempdir(), f"pr5_drop{k}_" + os.path.basename(csv))
    t.to_csv(f, index=False)
    return f


print("\n=== (d) drop-top CA HAI ben ===")
for k in (10, 20):
    for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
        for mode, fee in (("adv", 0.0008), ("noadv", None)):
            if tag == "f22" and mode == "noadv":
                continue
            sa = shuffle_stats(NavSim2(drop_top(AB, k), lo), K=25, advance_fee=fee, n=20)
            sg = shuffle_stats(NavSim2(drop_top(GB, k), lo), K=25, advance_fee=fee, n=20)
            d, sd = prop_delta(sa["mean"], sa["sd"], sg["mean"], sg["sd"])
            print(f"  drop{k} {tag} {mode}: ab x{sa['mean']:.2f}±{sa['sd']:.2f} "
                  f"gb x{sg['mean']:.2f}±{sg['sd']:.2f} -> {d:+.1f}%±{sd:.1f} "
                  f"({d/sd:+.1f}sd)", flush=True)
print("PR5_20_DONE")
