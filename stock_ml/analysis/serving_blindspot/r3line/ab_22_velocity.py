# -*- coding: utf-8 -*-
"""ab_22: van toc von — pnl/ngay-giu theo kenh exit, gb+mh16 vs t2429+mh16."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
gb = pd.read_csv(HERE / "r3_gb_mh16_s42_trades.csv")
cl = pd.read_csv(HERE / "r3_mh16_s42_trades.csv")
for name, df in (("gb+mh16", gb), ("t2429+mh16", cl)):
    df = df[df.holding_days > 0]
    tot_days = df.holding_days.sum()
    print(f"{name}: pnl_sum={df.pnl_pct.sum():.1f}u tren {tot_days:.0f} symbol-ngay "
          f"=> {df.pnl_pct.sum() / tot_days * 100:.3f}%/ngay-giu; "
          f"pnl/ngay theo kenh:")
    g = df.groupby("exit_reason").apply(
        lambda x: pd.Series({"n": len(x), "pnl_day_pct": x.pnl_pct.sum() / x.holding_days.sum() * 100,
                             "hold": x.holding_days.mean()}))
    print(g.round(3).to_string())
    # winners only (pnl>5%)
    w = df[df.pnl_pct > 0.05]
    print(f"  winners>5%: n={len(w)} mean={w.pnl_pct.mean()*100:.1f}% hold={w.holding_days.mean():.1f} "
          f"=> {w.pnl_pct.sum()/w.holding_days.sum()*100:.3f}%/ngay\n")
