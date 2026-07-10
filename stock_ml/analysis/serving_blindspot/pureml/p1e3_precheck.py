# -*- coding: utf-8 -*-
"""P1-E3 PRE-CHECK (offline, 0 run): rank-membership co mang winner TRUC GIAO
ma champion 2646 KHONG co khong? Neu winner rank-only ~0 hoac trung champion het
-> additive = fill-bound/redundant (nhu continuation-h6 corr 0.849) -> DONG truoc khi
viet engine code. Neu pool winner rank-only >=2022 dang ke -> di tiep E3 capacity-aware.

Artefacts: xr_xr_k20_s42_trades.csv (rank membership), xr_xr_smoke_champ2646_s42_trades.csv
(champion canonical 729.6). unit-weight = pnl_pct (giong composite total_pnl thô).
Kill/go do doc: tong pnl winner rank-only >=2022.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
RANK = HERE / "xr_xr_k20_s42_trades.csv"
CHAMP = HERE / "xr_xr_smoke_champ2646_s42_trades.csv"


def load(p):
    df = pd.read_csv(p, parse_dates=["entry_date", "exit_date"])
    df["year"] = df["entry_date"].dt.year
    return df


def champ_intervals(champ):
    """dict symbol -> list of (entry, exit) champion da giu."""
    d = {}
    for sym, g in champ.groupby("symbol"):
        d[sym] = list(zip(g["entry_date"], g["exit_date"]))
    return d


def overlaps(e, x, intervals):
    for (ce, cx) in intervals:
        if e <= cx and ce <= x:   # khoang [e,x] giao [ce,cx]
            return True
    return False


def main():
    rank = load(RANK)
    champ = load(CHAMP)
    ci = champ_intervals(champ)

    rank["champ_held"] = [
        overlaps(r.entry_date, r.exit_date, ci.get(r.symbol, []))
        for r in rank.itertuples()
    ]
    rank["is_win"] = rank["pnl_pct"] > 0
    rank["is_bigwin"] = rank["pnl_pct"] >= 0.15

    print("=== RANK trades (xr_k20 s42):", len(rank), "| champion:", len(champ), "===")
    print(f"champion tong pnl_pct = {champ['pnl_pct'].sum():.2f}u  (canonical 729.6)")
    print(f"rank tong pnl_pct     = {rank['pnl_pct'].sum():.2f}u\n")

    # 1. overlap tong the
    ov = rank["champ_held"].mean()
    print(f"[overlap] {ov:.1%} le rank trung vi the champion cung ma; "
          f"{1-ov:.1%} rank-only (truc giao)\n")

    # 2. phan ra pnl theo class x nam
    for label, sub in [("ALL", rank), (">=2022", rank[rank.year >= 2022])]:
        print(f"--- {label} ({len(sub)} le) ---")
        for held, name in [(False, "rank-ONLY (truc giao)"), (True, "trung champion")]:
            s = sub[sub.champ_held == held]
            win = s[s.is_win]
            print(f"  {name:24s} n={len(s):4d}  pnl={s.pnl_pct.sum():+7.2f}u  "
                  f"| winners n={len(win):4d} pnl={win.pnl_pct.sum():+7.2f}u  "
                  f"bigwin={int(s.is_bigwin.sum()):3d}")
        print()

    # 3. KILL/GO metric: winner rank-only >=2022
    ro = rank[(~rank.champ_held) & (rank.year >= 2022)]
    ro_win = ro[ro.is_win]
    print("=== KILL/GO ===")
    print(f"winner rank-only >=2022: n={len(ro_win)}  pnl={ro_win.pnl_pct.sum():+.2f}u  "
          f"bigwin={int(ro.is_bigwin.sum())}")
    print(f"  (net rank-only >=2022 gom ca loser: {ro.pnl_pct.sum():+.2f}u)")
    # top mã rank-only winner >=2022
    top = (ro_win.groupby("symbol")["pnl_pct"].agg(["sum", "count"])
           .sort_values("sum", ascending=False).head(12))
    print("\n  top mã winner rank-only >=2022 (champion KHONG giu cung luc):")
    for sym, row in top.iterrows():
        print(f"    {sym:6s} +{row['sum']:.2f}u ({int(row['count'])} le)")


if __name__ == "__main__":
    main()
