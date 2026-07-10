# -*- coding: utf-8 -*-
"""P1-E2 buoc 3 — tong hop per-seed per-year + cau truc trades tu cac CSV da dump."""
from __future__ import annotations
import glob, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name: str, seed: int) -> pd.DataFrame:
    p = os.path.join(HERE, f"xr_{name}_s{seed}_trades.csv")
    t = pd.read_csv(p, parse_dates=["entry_date", "exit_date"])
    t["y"] = t.entry_date.dt.year
    return t


def main():
    for name in ["xr_k20", "xr_k10"]:
        rows = []
        for sd in [42, 7, 99, 555]:
            p = os.path.join(HERE, f"xr_{name}_s{sd}_trades.csv")
            if not os.path.exists(p):
                continue
            t = load(name, sd)
            grid = set(t.entry_date.unique())
            off = t[~t.exit_date.isin(grid) & (t.exit_reason == "signal")]
            yr = t.groupby("y").pnl_pct.sum().round(2).to_dict()
            ge22 = t[t.y >= 2022]
            rows.append(dict(seed=sd, n=len(t), sum_pnl=round(t.pnl_pct.sum(), 2),
                             avg=round(t.pnl_pct.mean(), 4),
                             wr=round((t.pnl_pct > 0).mean(), 3),
                             hold=round(t.holding_days.mean(), 1),
                             n_force=len(off), force_avg=round(off.pnl_pct.mean(), 4),
                             ge22_sum=round(ge22.pnl_pct.sum(), 2),
                             ge22_avg=round(ge22.pnl_pct.mean(), 4),
                             per_year=yr))
        print(f"== {name} ==")
        for r in rows:
            py = " ".join(f"{y}:{v:+.2f}" for y, v in sorted(r.pop("per_year").items()))
            print(" ", r, "\n   per-year:", py)


if __name__ == "__main__":
    main()
