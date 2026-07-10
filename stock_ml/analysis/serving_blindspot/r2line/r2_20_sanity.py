# -*- coding: utf-8 -*-
"""R2 sanity/chong-ghost cho 1 variant vs r2_base: per-entry-year pnl/n/WR, exit-reason mix,
hold & pnl percentiles. Dung kem: sv_subframe.py (composite >=2022) + r2_nav.py date_lo=2022-01-01.

Usage: python r2_20_sanity.py <variant_csv> [base_csv]
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
var = pd.read_csv(sys.argv[1], parse_dates=["entry_date", "exit_date"])
base_csv = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "r2_base_s42_trades.csv")
base = pd.read_csv(base_csv, parse_dates=["entry_date", "exit_date"])
name = Path(sys.argv[1]).stem.replace("_s42_trades", "")

print(f"== {name} vs r2_base ==")
print(f"n={len(var)} (base {len(base)})  pnl={var.pnl_pct.sum():.1f} (base {base.pnl_pct.sum():.1f})")

print("\nper-entry-year:")
for y in range(2020, 2027):
    v = var[var.entry_date.dt.year == y]
    b = base[base.entry_date.dt.year == y]
    print(f"  {y}: n={len(v):4d}({len(b):4d}) pnl={v.pnl_pct.sum():+7.2f}({b.pnl_pct.sum():+7.2f}) "
          f"WR={((v.pnl_pct > 0).mean() if len(v) else 0):.3f} worst={v.pnl_pct.min() if len(v) else 0:+.3f}")

print("\nexit-reason mix (n, pnl):")
vm = var.groupby("exit_reason").pnl_pct.agg(["count", "sum"])
bm = base.groupby("exit_reason").pnl_pct.agg(["count", "sum"])
for r in sorted(set(vm.index) | set(bm.index)):
    v = vm.loc[r] if r in vm.index else pd.Series({"count": 0, "sum": 0.0})
    b = bm.loc[r] if r in bm.index else pd.Series({"count": 0, "sum": 0.0})
    print(f"  {r:16s}: {int(v['count']):5d} {v['sum']:+8.2f}   (base {int(b['count']):5d} {b['sum']:+8.2f})")

print("\npnl_pct percentiles (var | base):")
for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
    print(f"  p{int(q*100):02d}: {var.pnl_pct.quantile(q):+.4f} | {base.pnl_pct.quantile(q):+.4f}")
print(f"  max: {var.pnl_pct.max():+.4f} | {base.pnl_pct.max():+.4f}")
print(f"hold median/mean: {var.holding_days.median():.0f}/{var.holding_days.mean():.1f} "
      f"(base {base.holding_days.median():.0f}/{base.holding_days.mean():.1f})")

# top-10 winner: ghost concentration check
top = var.nlargest(10, "pnl_pct")[["symbol", "entry_date", "exit_date", "pnl_pct"]]
print("\ntop-10 winners:")
for r in top.itertuples():
    print(f"  {r.symbol:6s} {r.entry_date.date()} -> {r.exit_date.date()}  {r.pnl_pct:+.3f}")
top_share = var.nlargest(20, "pnl_pct").pnl_pct.sum() / var.pnl_pct.sum() if var.pnl_pct.sum() else float("nan")
print(f"top-20 share of total pnl: {top_share*100:.1f}%")
