# -*- coding: utf-8 -*-
"""r2c_02: soi cohort cascade (trade chi co o 1 he) giua oxtrail04 va c2 theo nam.
Exit doi ngay -> reentry_cooldown/occupancy doi -> entry khac nhau (deterministic cascade).
"""
import pandas as pd

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
c2 = pd.read_csv(f"{R2}/r2_c2_pb40snr_s42_trades.csv")
ox = pd.read_csv(f"{R2}/r2b_oxtrail04_s42_trades.csv")
for df in (c2, ox):
    df["entry_date"] = df["entry_date"].astype(str).str[:10]
    df["ey"] = df["entry_date"].str[:4].astype(int)

m = c2.merge(ox, on=["symbol", "entry_date"], suffixes=("_c2", "_ox"), how="outer",
             indicator=True)
lo = m[m._merge == "left_only"]    # chi c2 co
ro = m[m._merge == "right_only"]   # chi ox co

print("cohort chi-c2 (mat o ox): n per year / sum pnl")
print(lo.groupby(lo.ey_c2).agg(n=("pnl_pct_c2", "size"),
                               pnl=("pnl_pct_c2", "sum")).round(2).to_string())
print(f"tong: n={len(lo)} pnl={lo.pnl_pct_c2.sum():+.2f} avg={lo.pnl_pct_c2.mean():+.3f}")
print("\ncohort chi-ox (moi o ox): n per year / sum pnl")
print(ro.groupby(ro.ey_ox).agg(n=("pnl_pct_ox", "size"),
                               pnl=("pnl_pct_ox", "sum")).round(2).to_string())
print(f"tong: n={len(ro)} pnl={ro.pnl_pct_ox.sum():+.2f} avg={ro.pnl_pct_ox.mean():+.3f}")

print("\nnet cascade theo nam (ox_only - c2_only, sum pnl):")
a = ro.groupby(ro.ey_ox).pnl_pct_ox.sum()
b = lo.groupby(lo.ey_c2).pnl_pct_c2.sum()
for y in sorted(set(a.index) | set(b.index)):
    print(f"  {int(y)}: {a.get(y, 0) - b.get(y, 0):+.2f} (ox_only {a.get(y, 0):+.2f}, "
          f"c2_only {b.get(y, 0):+.2f})")

# 2024: overext_trail trades cua ox — thang/thua the nao so voi overext cua c2?
print("\n2024-2026 exit-mix pnl (moi he):")
for name, df in (("c2", c2), ("ox", ox)):
    sub = df[df.ey >= 2024]
    g = sub.groupby("exit_reason").agg(n=("pnl_pct", "size"), sum=("pnl_pct", "sum"),
                                       avg=("pnl_pct", "mean"))
    print(f"-- {name} (>=2024):")
    print(g.round(3).to_string())
