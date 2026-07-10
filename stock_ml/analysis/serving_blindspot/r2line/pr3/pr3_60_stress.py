# -*- coding: utf-8 -*-
"""pr3_60: tội 6 — (a) per-trade edge p42 vs gb; (b) phí R 0.6/0.7/0.8/0.9;
(c) entry lag +1 phiên (fill close ngày kế) áp CẢ HAI bên. n=20 perm K25 adv."""
import sys
import math

import pandas as pd

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/pr3")
sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402
from pr3_lib import NavSim2S  # noqa: E402

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
CSV_P42 = f"{R2}/r2c_oxt04_p42_s42_trades.csv"
CSV_GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"


def delta(sa, sg):
    d = (sa["mean"] / sg["mean"] - 1) * 100
    sd = math.sqrt((sa["sd"] / sg["mean"]) ** 2
                   + (sg["sd"] * sa["mean"] / sg["mean"] ** 2) ** 2) * 100
    return f"{d:+.1f}%±{sd:.1f} ({d/sd:+.1f}sd)"


# (a) per-trade edge
print("=== (a) per-trade edge ===")
for name, csv in (("p42", CSV_P42), ("gb", CSV_GB)):
    t = pd.read_csv(csv)
    w = t[t.pnl_pct > 0]
    l = t[t.pnl_pct <= 0]
    pf = w.pnl_pct.sum() / abs(l.pnl_pct.sum())
    print(f"{name}: n={len(t)} avg {t.pnl_pct.mean()*100:+.2f}% med {t.pnl_pct.median()*100:+.2f}% "
          f"wr {len(w)/len(t):.3f} PF {pf:.2f} hold_med {t.holding_days.median():.0f}d "
          f"sum {t.pnl_pct.sum():+.1f}")

# (b) phí R
print("\n=== (b) R sweep (K25 adv, full | f22) ===")
for R in (0.006, 0.007, 0.008, 0.009):
    row = {}
    for name, csv in (("p42", CSV_P42), ("gb", CSV_GB)):
        row[name] = (shuffle_stats(NavSim2(csv, "2020-01-01"), K=25, roundtrip=R,
                                   advance_fee=0.0008, n=20),
                     shuffle_stats(NavSim2(csv, "2022-01-01"), K=25, roundtrip=R,
                                   advance_fee=0.0008, n=20))
    print(f"R{R*100:.1f}: p42 x{row['p42'][0]['mean']:.2f} gb x{row['gb'][0]['mean']:.2f} "
          f"-> full {delta(row['p42'][0], row['gb'][0])}; "
          f"f22 {delta(row['p42'][1], row['gb'][1])}", flush=True)

# (c) entry lag +1 (cả hai bên)
print("\n=== (c) entry lag +1 phiên, fill close ngày kế (cả hai bên) ===")
for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22")):
    st = {}
    for name, csv in (("p42", CSV_P42), ("gb", CSV_GB)):
        sim = NavSim2S(csv, date_lo=lo)
        dropped = sim.apply_entry_lag(1)
        navs = []
        import statistics
        for seed in range(20):
            navs.append(float(sim.run_series(K=25, advance_fee=0.0008,
                                             order_seed=seed)["nav"].iloc[-1]))
        st[name] = dict(mean=statistics.mean(navs), sd=statistics.pstdev(navs))
        print(f"  {tag} {name} lag+1: x{st[name]['mean']:.2f}±{st[name]['sd']:.2f} "
              f"(dropped {dropped})", flush=True)
    print(f"  {tag} delta lag+1: {delta(st['p42'], st['gb'])}")
print("DONE")
