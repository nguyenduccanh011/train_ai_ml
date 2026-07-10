# -*- coding: utf-8 -*-
"""pm2 vong 2b: regime slice + tail-check + runner-check tren trades CSV cua cac o pm2.

- Regime (protocol gb_regime.py): pnl theo ENTRY year slice 2020 / >=2022 / >=2023 / >=2024 / >=2025,
  so voi champ 2646 (st_champ2646_s42_trades.csv) va gb_x08 (exitmap/gbx08_s42_trades.csv) — s42.
- Tail-check: con lech <= -20%? min pnl, n<=-0.20, p05.
- Runner-check (DIEU KIEN SONG): LPB 2022-11, VTP 2022-12, BSR 2025 con duoc om khong;
  + tong pnl cac lech hold>200d, top winners.

Usage: python pm2_03_checks.py pm2_hs10_zx25 [seed ...] (mac dinh 42 7 99 555 123)
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SB = HERE.parent
CHAMP_CSV = SB / "signalq" / "st_champ2646_s42_trades.csv"
GB_CSV = SB / "exitmap" / "gbx08_s42_trades.csv"


def slices(df: pd.DataFrame) -> dict:
    y = df.entry_date.dt.year
    return {"2020": df[y == 2020].pnl_pct.sum(), ">=2022": df[y >= 2022].pnl_pct.sum(),
            ">=2023": df[y >= 2023].pnl_pct.sum(), ">=2024": df[y >= 2024].pnl_pct.sum(),
            ">=2025": df[y >= 2025].pnl_pct.sum(), "all": df.pnl_pct.sum(), "n": len(df)}


def load(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, parse_dates=["entry_date", "exit_date"])
    return df


def runner_check(df: pd.DataFrame) -> None:
    probes = [("LPB", "2022-10-01", "2023-01-31"), ("VTP", "2022-11-01", "2023-02-28"),
              ("BSR", "2025-01-01", "2025-12-31")]
    for sym, lo, hi in probes:
        m = df[(df.symbol == sym) & (df.entry_date >= lo) & (df.entry_date <= hi)]
        if m.empty:
            print(f"    runner {sym} [{lo}..{hi}]: MISS (khong co entry)")
        else:
            for _, r in m.iterrows():
                print(f"    runner {sym}: entry {r.entry_date.date()} exit {r.exit_date.date() if pd.notna(r.exit_date) else 'open'} "
                      f"hold {r.holding_days:.0f}d pnl {r.pnl_pct:+.3f} reason {r.exit_reason}")
    long_ = df[df.holding_days > 200]
    print(f"    hold>200d: n={len(long_)} pnl_sum={long_.pnl_pct.sum():+.1f} "
          f"(t1058 s42 goc: 35 lech +109.6)")
    top = df.nlargest(5, "pnl_pct")[["symbol", "entry_date", "holding_days", "pnl_pct"]]
    print("    top5 winners: " + "; ".join(
        f"{r.symbol} {r.entry_date.date()} {r.holding_days:.0f}d {r.pnl_pct:+.2f}" for _, r in top.iterrows()))


def tail_check(df: pd.DataFrame) -> None:
    bad = df[df.pnl_pct <= -0.20]
    print(f"    tail: min={df.pnl_pct.min():+.3f} n<=-20%={len(bad)} p05={df.pnl_pct.quantile(0.05):+.3f} "
          f"| exit_reason mix: {df.exit_reason.value_counts().to_dict()}")
    if len(bad):
        w = bad.nsmallest(5, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "exit_reason"]]
        print("    worst5: " + "; ".join(
            f"{r.symbol} {r.entry_date.date()} {r.pnl_pct:+.2f} ({r.exit_reason})" for _, r in w.iterrows()))


def main():
    name = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2:]] or [42, 7, 99, 555, 123]
    champ = load(CHAMP_CSV); gb = load(GB_CSV)
    rows = {"champ2646_s42": slices(champ), "gb_x08_s42": slices(gb)}
    for sd in seeds:
        p = HERE / f"pm2_{name}_s{sd}_trades.csv"
        if not p.exists():
            print(f"missing {p}"); continue
        df = load(p)
        rows[f"{name}_s{sd}"] = slices(df)
        print(f"== {name} seed {sd} (n={len(df)}) ==")
        tail_check(df)
        runner_check(df)
    print("\n== pnl theo ENTRY-year slice ==")
    print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
