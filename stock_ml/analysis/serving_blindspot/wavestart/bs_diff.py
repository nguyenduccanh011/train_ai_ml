"""Trade-cohort diff: a bs_* sweep variant vs the champion 2646 seed-42 export.

New-trade cohort = entries (symbol, entry_date) present in the variant but not the champion
(the wave-start shallow fills). Lost = champion entries missing in the variant (occupancy
displacement risk, the F1 lesson). Reports count/WR/mean pnl/median hold/per-year for both.

Usage: python stock_ml/analysis/serving_blindspot/wavestart/bs_diff.py <variant_name>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CHAMP = HERE / "parity" / "before" / "trades_n2_2643_wavestruct_la05_lamp02.csv"


def load(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, parse_dates=["entry_date", "exit_date"])
    df["key"] = df["symbol"] + "|" + df["entry_date"].dt.strftime("%Y-%m-%d")
    return df


def cohort_stats(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"{label}: 0 trades")
        return
    wr = float((df["pnl_pct"] > 0).mean())
    print(f"{label}: n={len(df)} WR={wr:.3f} mean_pnl={df['pnl_pct'].mean():+.4f} "
          f"median_pnl={df['pnl_pct'].median():+.4f} sum_pnl={df['pnl_pct'].sum():+.3f} "
          f"med_hold={df['holding_days'].median():.0f}d")
    yr = df.groupby(df["entry_date"].dt.year)["pnl_pct"].agg(["count", "sum", "mean"])
    for y, r in yr.iterrows():
        print(f"    {y}: n={int(r['count'])} sum={r['sum']:+.3f} mean={r['mean']:+.4f}")
    rs = df["exit_reason"].value_counts()
    print("    exit_reasons: " + ", ".join(f"{k}={v}" for k, v in rs.items()))


def main() -> None:
    name = sys.argv[1]
    vpath = HERE / "runs" / name / f"trades_{name}.csv"
    champ = load(CHAMP)
    var = load(vpath)
    ck, vk = set(champ["key"]), set(var["key"])
    new = var[var["key"].isin(vk - ck)]
    lost = champ[champ["key"].isin(ck - vk)]
    same = champ[champ["key"].isin(ck & vk)]
    print(f"== {name} vs champion: champ={len(champ)} var={len(var)} "
          f"common={len(same)} new={len(new)} lost={len(lost)}")
    cohort_stats(new, "NEW (wave-start fills)")
    cohort_stats(lost, "LOST (displaced core)")
    # entry-price shift on common trades (same signal, shallower fill = higher entry)
    m = champ.merge(var, on="key", suffixes=("_c", "_v"))
    shift = m[m["entry_price_c"] != m["entry_price_v"]]
    if len(shift):
        rel = (shift["entry_price_v"] / shift["entry_price_c"] - 1.0)
        dpnl = shift["pnl_pct_v"] - shift["pnl_pct_c"]
        print(f"COMMON repriced: n={len(shift)} mean_entry_shift={rel.mean():+.4f} "
              f"mean_pnl_delta={dpnl.mean():+.4f} sum_pnl_delta={dpnl.sum():+.3f}")
    else:
        print("COMMON repriced: 0")


if __name__ == "__main__":
    main()
