# -*- coding: utf-8 -*-
"""ab_23: exit-reason mix cua cac ablation da xong."""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
for v in sys.argv[1:]:
    p = HERE / f"ab_{v}_s42_trades.csv"
    if not p.exists():
        print(f"{v}: chua co")
        continue
    df = pd.read_csv(p)
    g = df.groupby("exit_reason").agg(n=("pnl_pct", "size"), pnl_u=("pnl_pct", "sum"),
                                      mean_pct=("pnl_pct", "mean"), hold=("holding_days", "mean"))
    g["mean_pct"] *= 100
    print(f"\n== ab_{v}: {len(df)} trades pnl={df.pnl_pct.sum():.1f}u hold_med={df.holding_days.median():.0f}")
    print(g.round(2).to_string())
