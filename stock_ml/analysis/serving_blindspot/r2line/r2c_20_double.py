# -*- coding: utf-8 -*-
"""r2c_20: kiem tra kep cho cac diem hua hen (>= +14% full VA f22 >= +6% vs gb).

Moi ung vien (adv, R0.6, K25):
  - NAV-tu-2023 (chong-ghost sau hon) shuffle-mean vs gb f23
  - MaxDD mean±sd + worst permutation (full frame)
  - top-20-trade share (pnl_pct duong lon nhat / tong pnl duong)
  - per-entry-year pnl duong du 7 nam (tu trades csv)
  - yearly NAV mean 20 perm (tu 2022) de soi lat am
Usage: python r2c_20_double.py <name> [<name> ...]   (name = r2c_xxx | r2b_oxtrail04)
"""
import csv as csvmod
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "na_audit"))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

ADV = 0.0008
CSV_GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"

GB = None
with open(HERE / "na_audit" / "nh_frontier_results.csv") as f:
    for r in csvmod.DictReader(f):
        if r["sys"] == "gb" and r["mode"] == "lag2-adv":
            GB = {k: float(r[k]) for k in ("full_mean", "full_sd", "f22_mean", "f22_sd")}

# gb f23 reference: tinh mot lan tai cho
print("tinh gb f23 reference...", flush=True)
_gb23 = shuffle_stats(NavSim2(CSV_GB, date_lo="2023-01-01"), K=25, roundtrip=0.006,
                      settle_lag=2, advance_fee=ADV)
print(f"gb f23: x{_gb23['mean']:.3f}±{_gb23['sd']:.3f}", flush=True)


def delta(mc, sc, mg, sg):
    d = (mc / mg - 1) * 100
    sd = (mc / mg) * math.sqrt((sc / mc) ** 2 + (sg / mg) ** 2) * 100
    return d, sd


for name in sys.argv[1:]:
    csv_path = HERE / f"{name}_s42_trades.csv"
    print(f"\n===== {name} =====", flush=True)
    # f23
    st23 = shuffle_stats(NavSim2(str(csv_path), date_lo="2023-01-01"), K=25,
                         roundtrip=0.006, settle_lag=2, advance_fee=ADV)
    d23, s23 = delta(st23["mean"], st23["sd"], _gb23["mean"], _gb23["sd"])
    print(f"f23: x{st23['mean']:.2f}±{st23['sd']:.2f} (vs gb {d23:+.1f}%±{s23:.1f} "
          f"= {d23/s23:.1f}sd) DD {st23['dd_mean']*100:.1f}%", flush=True)
    # full DD mean±sd + worst perm
    sim_f = NavSim2(str(csv_path), date_lo="2020-01-01")
    import statistics
    navs, dds = [], []
    for seed in range(20):
        m = sim_f.run(K=25, roundtrip=0.006, settle_lag=2, advance_fee=ADV, order_seed=seed)
        navs.append(m["final"]); dds.append(m["maxdd"])
    print(f"full: x{statistics.mean(navs):.2f}±{statistics.pstdev(navs):.2f} "
          f"[min x{min(navs):.2f} / max x{max(navs):.2f}]", flush=True)
    print(f"MaxDD: mean {statistics.mean(dds)*100:.2f}% ± {statistics.pstdev(dds)*100:.2f} "
          f"| worst-perm {min(dds)*100:.2f}% (gb mean -15.28/worst -15.61)", flush=True)
    # trades-level: top-20 share + per-entry-year
    t = pd.read_csv(csv_path)
    t["ey"] = t.entry_date.astype(str).str[:4].astype(int)
    pos = t[t.pnl_pct > 0].pnl_pct
    print(f"top-20 share: {pos.nlargest(20).sum() / pos.sum() * 100:.1f}% "
          f"(cua tong pnl duong; chuan c2 ~7.5%)", flush=True)
    gy = t.groupby("ey").pnl_pct.sum()
    neg = [f"{y}:{v:+.1f}" for y, v in gy.items() if v <= 0]
    print("per-entry-year pnl: " + " ".join(f"{y}:{v:+.1f}" for y, v in gy.items())
          + ("  <-- CO NAM AM: " + ",".join(neg) if neg else "  (du 7 nam duong)"), flush=True)
print("\nDONE", flush=True)
