# -*- coding: utf-8 -*-
"""xg_20: dao sau cho XGEM_CROSSAPPLY.

Moi label: NAV full/f22/f23 (adv+noadv), LOYO-2021 (adv), per-year pnl, hold median,
overlap trades vs t2429+mh16 (r3_mh16_s42_trades.csv) exact + fuzzy ±5d.

Usage: python xg_20_deep.py <label1> [label2 ...]  (hoac --all)
Labels: dyn_base, v19_base, dyn_mh12/16/25, v19_mh12/16/25, mh16_ref
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r2line" / "na_audit"))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

NAVSCAN = HERE.parent / "r2line" / "navscan"
MH16 = HERE / "r3_mh16_s42_trades.csv"

CSVS = {
    "dyn_base": NAVSCAN / "nv_n2_2429_dyncsr88_s42_trades.csv",
    "v19_base": NAVSCAN / "nv_n2_v19_fullwave_s42_trades.csv",
    "dyn_mh12": HERE / "xg_dyn_mh12_s42_trades.csv",
    "dyn_mh16": HERE / "xg_dyn_mh16_s42_trades.csv",
    "dyn_mh25": HERE / "xg_dyn_mh25_s42_trades.csv",
    "v19_mh12": HERE / "xg_v19_mh12_s42_trades.csv",
    "v19_mh16": HERE / "xg_v19_mh16_s42_trades.csv",
    "v19_mh25": HERE / "xg_v19_mh25_s42_trades.csv",
    "mh16_ref": MH16,
}


def nav(csv, lo, advance=0.0008, drop_year=None):
    sim = NavSim2(str(csv), date_lo=lo)
    if drop_year:
        sim.trades = [t for t in sim.trades if t["entry_date"][:4] != drop_year]
    return shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=advance, n=20)


def overlap(csv, ref_csv):
    a = pd.read_csv(csv)
    b = pd.read_csv(ref_csv)
    a["ed"] = a.entry_date.astype(str).str[:10]
    b["ed"] = b.entry_date.astype(str).str[:10]
    ka = set(zip(a.symbol, a.ed))
    kb = set(zip(b.symbol, b.ed))
    exact = len(ka & kb) / len(ka)
    bd = {}
    for s, d in kb:
        bd.setdefault(s, []).append(pd.Timestamp(d))
    fz = 0
    for s, d in ka:
        dts = bd.get(s)
        if dts is not None:
            d = pd.Timestamp(d)
            if any(abs((d - x).days) <= 5 for x in dts):
                fz += 1
    return exact, fz / len(ka)


def main():
    labels = sys.argv[1:]
    if not labels or labels[0] == "--all":
        labels = list(CSVS)
    for label in labels:
        csv = CSVS[label]
        if not Path(csv).exists():
            print(f"MISSING {label}: {csv}", flush=True)
            continue
        print(f"\n===== {label} =====", flush=True)
        for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"), ("2023-01-01", "f23")):
            a = nav(csv, lo, advance=0.0008)
            n = nav(csv, lo, advance=None)
            print(f"{tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} DDw {a['dd_worst']*100:.1f}% "
                  f"| noadv x{n['mean']:.2f}±{n['sd']:.2f} DDw {n['dd_worst']*100:.1f}%",
                  flush=True)
        st = nav(csv, "2020-01-01", drop_year="2021")
        print(f"LOYO-2021 adv: x{st['mean']:.2f}±{st['sd']:.2f}", flush=True)
        if Path(csv) != MH16 and MH16.exists():
            ex, fz = overlap(csv, MH16)
            print(f"overlap vs t2429+mh16: exact {ex*100:.1f}% fuzzy±5d {fz*100:.1f}%", flush=True)
        t = pd.read_csv(csv)
        t["yr"] = t.entry_date.astype(str).str[:4]
        print("per-year pnl:", t.groupby("yr").pnl_pct.sum().round(1).to_dict(), flush=True)
        print("hold median:", t.holding_days.median(), " trades:", len(t),
              " exit_reason:", t.exit_reason.value_counts().to_dict(), flush=True)
    print("XG20_DONE", flush=True)


if __name__ == "__main__":
    main()
