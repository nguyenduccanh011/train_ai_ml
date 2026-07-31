# -*- coding: utf-8 -*-
"""hb_20: cong to rut gon t2943 — 3 cot t2943 vs t2936 vs dyn25 (deltas vs gb):
(a) held-out f21/f24 (2 che do); (b) LOYO-2021 + LOYO-2022 per-perm (2 che do);
(c) drop-top-20 HAI BEN (full adv/noadv, f22 adv); (d) entry lag+1 HAI BEN
(full/f22 x adv/noadv); (e) fee R0.8/R1.0 adv (full/f22).
BASELINE gb = re-dump run DB t2783 s42 (x14.14 adv full) — enriched2 goc bi xoa
boi refactor 926f4c6b, KHONG tracked; moi cot do cung baseline -> nhat quan noi bo,
deltas se THAP hon con so cong bo cu ~2.6% full / ~1% f22 (baseline manh hon)."""
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402
from pr3_lib import NavSim2S, yearly_returns  # noqa: E402

CSVS = {
    "t2943": str(HERE / "pr5_dynclean_mh25_s42_trades.csv"),
    "t2936": str(HERE / "ab_noT_s42_trades.csv"),
    "dyn25": str(HERE / "xg_dyn_mh25_s42_trades.csv"),
    "gb": str(HERE / "gb_t2783_s42_trades.csv"),
}
CANDS = ("t2943", "t2936", "dyn25")


def delta(sa, sg):
    d = (sa["mean"] / sg["mean"] - 1) * 100
    sd = math.sqrt((sa["sd"] / sg["mean"]) ** 2
                   + (sg["sd"] * sa["mean"] / sg["mean"] ** 2) ** 2) * 100
    z = d / sd if sd > 0 else float("inf")
    return f"{d:+.1f}%±{sd:.1f} ({z:+.1f}sd)"


def prop_delta(ma, sa, mg, sg):
    d = (ma / mg - 1) * 100
    sd = math.sqrt((sa / mg) ** 2 + (sg * ma / mg ** 2) ** 2) * 100
    return d, sd


print("=== (a) HELD-OUT f21/f24 (2 che do, vs gb re-dump) ===", flush=True)
for lo, tag in (("2021-01-01", "f21"), ("2024-01-01", "f24")):
    stats = {}
    for name, csv in CSVS.items():
        sim = NavSim2(csv, date_lo=lo)
        stats[name] = {
            "adv": shuffle_stats(sim, K=25, advance_fee=0.0008, n=20),
            "noadv": shuffle_stats(sim, K=25, advance_fee=None, n=20)}
    for mode in ("adv", "noadv"):
        g = stats["gb"][mode]
        print(f"-- {tag} {mode} (gb x{g['mean']:.2f}±{g['sd']:.2f} "
              f"DDw {g['dd_worst']*100:.1f}%)", flush=True)
        for name in CANDS:
            s = stats[name][mode]
            print(f"  {name:6s} x{s['mean']:.2f}±{s['sd']:.2f} "
                  f"DDw {s['dd_worst']*100:.1f}%  vs gb {delta(s, g)}", flush=True)

print("\n=== (b) LOYO-2021 / LOYO-2022 per-perm (2 che do) ===", flush=True)
for mode, fee in (("adv", 0.0008), ("noadv", None)):
    yr = {}
    for name, csv in CSVS.items():
        sim = NavSim2S(csv)
        yr[name] = [yearly_returns(sim.run_series(K=25, advance_fee=fee, order_seed=sd))
                    for sd in range(20)]
    years = sorted(yr["gb"][0].keys())
    for drop in (2021, 2022):
        navs = {}
        for name in CSVS:
            vals = []
            for d in yr[name]:
                p = 1.0
                for yy in years:
                    if yy != drop:
                        p *= 1 + d.get(yy, 0.0)
                vals.append(p)
            navs[name] = (statistics.mean(vals), statistics.pstdev(vals))
        mg, sg = navs["gb"]
        out = []
        for name in CANDS:
            ma, sa = navs[name]
            d, sd = prop_delta(ma, sa, mg, sg)
            tag = "PASS>=2sd" if d / sd >= 2 else ("~1-2sd" if d / sd >= 1 else "FAIL<1sd")
            out.append(f"{name} x{ma:.2f}±{sa:.2f} {d:+.1f}%({d/sd:+.1f}sd){tag}")
        print(f"  bo{drop} {mode} (gb x{mg:.2f}±{sg:.2f}): " + " | ".join(out), flush=True)

def drop_top(csv, k):
    t = pd.read_csv(csv)
    t = t.sort_values("pnl_pct", ascending=False).iloc[k:]
    f = os.path.join(tempfile.gettempdir(), f"hb20_drop{k}_" + os.path.basename(csv))
    t.to_csv(f, index=False)
    return f


print("\n=== (c) DROP-TOP-20 HAI BEN ===", flush=True)
for lo, tag, mode, fee in (("2020-01-01", "full", "adv", 0.0008),
                           ("2020-01-01", "full", "noadv", None),
                           ("2022-01-01", "f22", "adv", 0.0008)):
    sg = shuffle_stats(NavSim2(drop_top(CSVS["gb"], 20), lo), K=25, advance_fee=fee, n=20)
    out = []
    for name in CANDS:
        sa = shuffle_stats(NavSim2(drop_top(CSVS[name], 20), lo), K=25, advance_fee=fee, n=20)
        out.append(f"{name} x{sa['mean']:.2f}±{sa['sd']:.2f} {delta(sa, sg)}")
    print(f"  drop20 {tag} {mode} (gb x{sg['mean']:.2f}±{sg['sd']:.2f}): "
          + " | ".join(out), flush=True)

print("\n=== (d) ENTRY LAG+1 HAI BEN ===", flush=True)
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
    for mode, fee in (("adv", 0.0008), ("noadv", None)):
        sg_ = NavSim2S(CSVS["gb"], lo)
        dg = sg_.apply_entry_lag(1)
        sg = shuffle_stats(sg_, K=25, advance_fee=fee, n=20)
        out = []
        for name in CANDS:
            sm = NavSim2S(CSVS[name], lo)
            dm = sm.apply_entry_lag(1)
            sa = shuffle_stats(sm, K=25, advance_fee=fee, n=20)
            out.append(f"{name} x{sa['mean']:.2f}±{sa['sd']:.2f} {delta(sa, sg)}")
        print(f"  lag+1 {tag} {mode} (gb x{sg['mean']:.2f} drop{dg}): "
              + " | ".join(out), flush=True)

print("\n=== (e) FEE SWEEP R0.8/R1.0 (adv) ===", flush=True)
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
    sims = {name: NavSim2(csv, lo) for name, csv in CSVS.items()}
    for R in (0.008, 0.010):
        sg = shuffle_stats(sims["gb"], K=25, roundtrip=R, advance_fee=0.0008, n=20)
        out = []
        for name in CANDS:
            sa = shuffle_stats(sims[name], K=25, roundtrip=R, advance_fee=0.0008, n=20)
            out.append(f"{name} x{sa['mean']:.2f}±{sa['sd']:.2f} {delta(sa, sg)}")
        print(f"  {tag} R{R*100:.1f} (gb x{sg['mean']:.2f}±{sg['sd']:.2f}): "
              + " | ".join(out), flush=True)
print("HB_20_DONE", flush=True)
