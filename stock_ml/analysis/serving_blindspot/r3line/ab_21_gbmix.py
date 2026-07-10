# -*- coding: utf-8 -*-
"""ab_21: exit-reason mix cua gb_x08 khong cap (r3_base_2783) — kiem chung kenh
struct-trail/overext o gb truoc khi cap."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "r3_base_2783_s42_trades.csv")
print(f"gb_x08 uncapped: {len(df)} trades, pnl={df.pnl_pct.sum():.1f}u, hold={df.holding_days.mean():.1f}")
g = df.groupby("exit_reason").agg(n=("pnl_pct", "size"), pnl_u=("pnl_pct", "sum"),
                                  mean_pct=("pnl_pct", "mean"), hold=("holding_days", "mean"))
g["mean_pct"] *= 100
print(g.round(2).to_string())
print("\nhold distribution overext/trailing:")
for r in g.index:
    sub = df[df.exit_reason == r]
    print(r, "hold p25/50/75/90:", sub.holding_days.quantile([.25, .5, .75, .9]).tolist())
