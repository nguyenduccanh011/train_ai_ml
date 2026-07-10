"""B-entry cohort + displacement diagnostics: a bch_* variant vs the champion 2646 seed-42
export (parity/bch_before). Cohorts:

- B         = variant trades whose (symbol, entry_date) matches the engine's b_entries.csv
              side log (the confirmation-breakout fills).
- NEW-nonB  = variant entries absent from the champion and NOT B (knock-on reshuffle).
- LOST      = champion entries absent from the variant (displacement; a shifted entry date
              shows up as LOST + NEW-nonB under the symbol|entry_date key).
- COMMON    = same entry; report any repriced/changed-exit deltas.

Also: per-year for B (do they land in the 2022/2024/2026 blindspot years?), exit reasons,
displacement net (B pnl + NEW-nonB pnl - LOST pnl), and the idle-slot check (% of B entries
with no champion trade on that symbol entered within +/-10 business days, and % not inside
any champion holding interval = true additivity).

Usage: python stock_ml/analysis/serving_blindspot/wavestart/bch_diff.py <variant_name>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHAMP = HERE / "parity" / "bch_before" / "trades_n2_2643_wavestruct_la05_lamp02.csv"


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
    bents = pd.read_csv(HERE / "runs" / name / "b_entries.csv",
                        parse_dates=["signal_date", "entry_date"])
    bkeys = set(bents["symbol"] + "|" + bents["entry_date"].dt.strftime("%Y-%m-%d")) \
        if len(bents) else set()

    ck, vk = set(champ["key"]), set(var["key"])
    b = var[var["key"].isin(bkeys)]
    new_nonb = var[var["key"].isin(vk - ck - bkeys)]
    lost = champ[champ["key"].isin(ck - vk)]
    common = champ[champ["key"].isin(ck & vk)]
    print(f"== {name} vs champion: champ={len(champ)} var={len(var)} "
          f"B-log={len(bents)} B-trades={len(b)} new_nonB={len(new_nonb)} "
          f"lost={len(lost)} common={len(common)}")
    if len(bents) != len(b):
        # B log entries whose (symbol, entry_date) has no trade row (should not happen) or
        # that coincide with a champion key (B preempted a core same-day at-market entry).
        print(f"   note: {len(bents) - len(b)} B-log entries not matched as B-trades "
              f"(overlap with champion keys: {len(bkeys & ck)})")
    cohort_stats(b, "B-COHORT (breakout fills)")
    cohort_stats(new_nonb, "NEW-nonB (knock-on reshuffle)")
    cohort_stats(lost, "LOST (displaced core)")

    # entry/exit shift on common trades
    m = champ.merge(var, on="key", suffixes=("_c", "_v"))
    shift = m[(m["entry_price_c"] != m["entry_price_v"])
              | (m["pnl_pct_c"] != m["pnl_pct_v"])]
    if len(shift):
        rel = (shift["entry_price_v"] / shift["entry_price_c"] - 1.0)
        dpnl = shift["pnl_pct_v"] - shift["pnl_pct_c"]
        print(f"COMMON changed: n={len(shift)} mean_entry_shift={rel.mean():+.4f} "
              f"mean_pnl_delta={dpnl.mean():+.4f} sum_pnl_delta={dpnl.sum():+.3f}")
    else:
        print("COMMON changed: 0")

    net = b["pnl_pct"].sum() + new_nonb["pnl_pct"].sum() - lost["pnl_pct"].sum()
    print(f"DISPLACEMENT NET (B + new_nonB - lost, pnl units): {net:+.3f} "
          f"(B {b['pnl_pct'].sum():+.3f} | new_nonB {new_nonb['pnl_pct'].sum():+.3f} "
          f"| lost {lost['pnl_pct'].sum():+.3f})")

    # Idle-slot check on the B log (all triggers, matched or not)
    if len(bents):
        champ_by_sym = {s: g for s, g in champ.groupby("symbol")}
        far, outside = 0, 0
        for _, r in bents.iterrows():
            g = champ_by_sym.get(r["symbol"])
            if g is None or g.empty:
                far += 1
                outside += 1
                continue
            ed = np.datetime64(r["entry_date"], "D")
            ents = g["entry_date"].to_numpy().astype("datetime64[D]")
            bd = np.abs([int(np.busday_count(min(e, ed), max(e, ed))) for e in ents])
            if min(bd) > 10:
                far += 1
            inside = ((g["entry_date"] <= r["entry_date"])
                      & (g["exit_date"] >= r["entry_date"])).any()
            if not inside:
                outside += 1
        print(f"IDLE-SLOT: {far}/{len(bents)} ({far / len(bents):.1%}) B-entries have no "
              f"champion entry on the symbol within +/-10 bdays; "
              f"{outside}/{len(bents)} ({outside / len(bents):.1%}) fall outside every "
              f"champion holding interval")


if __name__ == "__main__":
    main()
