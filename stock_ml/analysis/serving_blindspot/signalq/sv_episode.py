"""SALVAGE 2026-03 episode check: did the crash-brake fire and pull exits before the 03-12 dump?
Usage: python sv_episode.py <variant_csv> <old_stack_csv> [crash_dd]
1) Reconstructs the EW regime index (same idiom as experiment._market_bull_mask) from duckdb closes,
   prints bull-mask (MA60/persist3) vs crash-brake state through 2026-02..2026-03.
2) Compares BCM/FPT/VND/GEX/SAB/TCB 2026-entry trades: variant vs old stack (2760).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import duckdb  # noqa: E402

D = Path(__file__).parent
var = pd.read_csv(sys.argv[1], parse_dates=["entry_date", "exit_date"])
old = pd.read_csv(sys.argv[2], parse_dates=["entry_date", "exit_date"])
CRASH_DD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

uni = sorted(set(var.symbol) | set(old.symbol))
con = duckdb.connect(str(REPO / "market_data" / "market.duckdb"), read_only=True)
px = con.execute(
    "SELECT symbol, date, close FROM ohlcv WHERE symbol IN ({}) AND date >= '2019-06-01'".format(
        ",".join("'" + s + "'" for s in uni))).df()
con.close()
piv = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
mret = piv.pct_change().replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
lvl = (1.0 + mret.fillna(0.0)).cumprod()
ma = lvl.rolling(60, min_periods=60).mean()
above = lvl > ma
bull = (above.rolling(3, min_periods=3).sum() >= 3).where(ma.notna(), False)
peak = lvl.rolling(20, min_periods=1).max()
dd = lvl / peak - 1.0
crash = dd <= -CRASH_DD
bull_v2 = bull & ~crash

w = (lvl.index >= pd.Timestamp("2026-02-15")) & (lvl.index <= pd.Timestamp("2026-03-31"))
print(f"== regime index 2026-02-15..2026-03-31 (crash_dd={CRASH_DD}) ==")
print(f"{'date':12s} {'lvl':>8s} {'dd20':>7s} {'bull_old':>8s} {'crash':>6s} {'bull_v2':>7s}")
for d in lvl.index[w]:
    print(f"{str(pd.Timestamp(d).date()):12s} {lvl[d]:8.4f} {dd[d]*100:6.2f}% "
          f"{str(bool(bull[d])):>8s} {str(bool(crash[d])):>6s} {str(bool(bull_v2[d])):>7s}")
ff = lvl.index[w & crash.to_numpy()]
print(f"\nbrake first fired: {ff.min().date() if len(ff) else 'NEVER in window'}; "
      f"old bull mask first False in window: "
      f"{min([d for d in lvl.index[w] if not bull[d]], default=None)}")

print("\n== 2026-entry episode trades: variant vs old stack (2760) ==")
SYMS = ["BCM", "FPT", "VND", "GEX", "SAB", "TCB"]
for df, nm in [(old, "old_stack"), (var, "variant")]:
    sub = df[(df.entry_date >= "2026-01-01") & df.symbol.isin(SYMS)].sort_values(["symbol", "entry_date"])
    print(f"-- {nm}:")
    for r in sub.itertuples():
        print(f"   {r.symbol:5s} entry {r.entry_date.date()} exit {r.exit_date.date()} "
              f"pnl {r.pnl_pct:+.3f} hold {r.holding_days:.0f} ({r.exit_reason})")

print("\nall 2026 entries pnl: variant "
      f"{var[var.entry_date >= '2026-01-01'].pnl_pct.sum():+.2f} vs old_stack "
      f"{old[old.entry_date >= '2026-01-01'].pnl_pct.sum():+.2f}")
