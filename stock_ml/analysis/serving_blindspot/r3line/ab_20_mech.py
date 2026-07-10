# -*- coding: utf-8 -*-
"""ab_20: doc co che +8% o muc trade — so r3_gb_mh16 (t2910, du 16 key + downpress)
vs r3_mh16 (t2907, t2429+mh16 sach). Exit-reason mix, hold, pnl, matched-entry diff."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
GB = HERE / "r3_gb_mh16_s42_trades.csv"     # gb+mh16 (16 key + downpress)
CL = HERE / "r3_mh16_s42_trades.csv"        # t2429+mh16 (clean)

gb = pd.read_csv(GB)
cl = pd.read_csv(CL)
for name, df in (("gb+mh16", gb), ("t2429+mh16", cl)):
    print(f"\n===== {name}: {len(df)} trades, pnl_sum={df.pnl_pct.sum():.1f}u, "
          f"mean={df.pnl_pct.mean()*100:.2f}%, wr={(df.pnl_pct > 0).mean():.3f}, "
          f"hold_mean={df.holding_days.mean():.1f} med={df.holding_days.median():.0f}")
    g = df.groupby("exit_reason").agg(n=("pnl_pct", "size"), pnl_u=("pnl_pct", "sum"),
                                      mean_pct=("pnl_pct", "mean"), hold=("holding_days", "mean"),
                                      wr=("pnl_pct", lambda x: (x > 0).mean()))
    g["mean_pct"] *= 100
    print(g.round(2).to_string())

# matched entries: cung symbol + entry_date
gb["k"] = gb.symbol + "|" + gb.entry_date.astype(str).str[:10]
cl["k"] = cl.symbol + "|" + cl.entry_date.astype(str).str[:10]
m = gb.merge(cl, on="k", suffixes=("_gb", "_cl"))
print(f"\n===== matched entries: {len(m)} (gb-only {len(gb) - len(m)}, cl-only {len(cl) - len(m)})")
same = m[(m.exit_date_gb == m.exit_date_cl)]
diff = m[(m.exit_date_gb != m.exit_date_cl)]
print(f"same exit: {len(same)} pnl_gb={same.pnl_pct_gb.sum():.1f} pnl_cl={same.pnl_pct_cl.sum():.1f}")
print(f"diff exit: {len(diff)} pnl_gb={diff.pnl_pct_gb.sum():.1f} pnl_cl={diff.pnl_pct_cl.sum():.1f} "
      f"delta_cl_minus_gb={diff.pnl_pct_cl.sum() - diff.pnl_pct_gb.sum():.1f}u")
d = diff.copy()
d["dh"] = d.holding_days_gb - d.holding_days_cl
d["dp"] = (d.pnl_pct_gb - d.pnl_pct_cl) * 100
print("\ndiff-exit theo (reason_gb -> reason_cl): n, hold_gb, hold_cl, pnl%gb, pnl%cl, delta_u")
g2 = d.groupby(["exit_reason_gb", "exit_reason_cl"]).agg(
    n=("dp", "size"), hold_gb=("holding_days_gb", "mean"), hold_cl=("holding_days_cl", "mean"),
    pnl_gb=("pnl_pct_gb", "mean"), pnl_cl=("pnl_pct_cl", "mean"),
    delta_u=("dp", lambda x: x.sum() / 100))
g2["pnl_gb"] *= 100
g2["pnl_cl"] *= 100
print(g2.round(2).sort_values("delta_u").to_string())

print("\ntop-12 lech am nhat (gb thua):")
cols = ["k", "exit_reason_gb", "exit_reason_cl", "holding_days_gb", "holding_days_cl",
        "pnl_pct_gb", "pnl_pct_cl"]
print(d.nsmallest(12, "dp")[cols].to_string(index=False))
print("\ntop-8 lech duong nhat (gb hon):")
print(d.nlargest(8, "dp")[cols].to_string(index=False))

# unmatched: lech vong quay (slot bi giu -> mat entry sau)
gbo = gb[~gb.k.isin(cl.k)]
clo = cl[~cl.k.isin(gb.k)]
print(f"\ngb-only trades: {len(gbo)} pnl={gbo.pnl_pct.sum():.1f}u mean={gbo.pnl_pct.mean()*100:.2f}% "
      f"hold={gbo.holding_days.mean():.1f}")
print(f"cl-only trades: {len(clo)} pnl={clo.pnl_pct.sum():.1f}u mean={clo.pnl_pct.mean()*100:.2f}% "
      f"hold={clo.holding_days.mean():.1f}")
print("cl-only theo nam:")
print(clo.assign(y=clo.entry_date.astype(str).str[:4]).groupby("y").pnl_pct.agg(["size", "sum"])
      .round(1).to_string())
print("gb-only theo nam:")
print(gbo.assign(y=gbo.entry_date.astype(str).str[:4]).groupby("y").pnl_pct.agg(["size", "sum"])
      .round(1).to_string())
print("AB20_DONE")
