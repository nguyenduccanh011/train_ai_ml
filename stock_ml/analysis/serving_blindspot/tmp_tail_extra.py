import numpy as np
import pandas as pd

AN = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
T = pd.read_csv(AN + r"/trades_metrics.csv", parse_dates=["entry_date", "exit_date"])
P = pd.read_csv(AN + r"/tmp_tail_sim_per_trade.csv", parse_dates=["entry_date", "exit_date"])

bl = T[T.pnl_pct <= -0.15]
print("bl15 mfe: max=", round(bl.mfe_pct.max(), 4), " n mfe>=0.27 (trail arm):", (bl.mfe_pct >= 0.27).sum(),
      " n mfe>=0.12 (bev arm):", (bl.mfe_pct >= 0.12).sum(), " n mfe>=0.08:", (bl.mfe_pct >= 0.08).sum())
b10 = T[T.pnl_pct <= -0.10]
print("bl10 (n=192) mfe: n>=0.27:", (b10.mfe_pct >= 0.27).sum(), " n>=0.12:", (b10.mfe_pct >= 0.12).sum())

# gap severity of stop_8 fills
s8 = P.sim_stop_8.dropna()
print(f"\nstop_8 fills: n={len(s8)} mean={s8.mean():.4f} median={s8.median():.4f} "
      f"p5={s8.quantile(.05):.4f} min={s8.min():.4f}")
print("fills landing <= -12% (stop level -8%, gap >=4pp):", (s8 <= -0.12).sum(),
      "| <= -15%:", (s8 <= -0.15).sum(), "| <= -20%:", (s8 <= -0.20).sum())
s10 = P.sim_stop_10.dropna()
print(f"stop_10 fills: n={len(s10)} mean={s10.mean():.4f}; <=-15%: {(s10<=-0.15).sum()}; <=-20%: {(s10<=-0.20).sum()}")

# where does stop_8's winner-kill come from: winners that dipped below -8% then recovered
w = P[(P.pnl_pct > 0) & P.sim_stop_8.notna()]
print(f"\nstop_8 kills {len(w)} winners: their base sum +{w.pnl_pct.sum():.2f}u -> sim sum {w.sim_stop_8.sum():.2f}u")
bw = w[w.pnl_pct >= 0.15]
print("  big winners among them:", len(bw), "examples:")
print(bw.nlargest(6, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "sim_stop_8", "mae_pct", "mfe_pct"]].to_string(index=False))

# tstop_15 big-winner kills (V-shape recoveries)
w = P[(P.pnl_pct >= 0.15) & P.sim_tstop_15.notna()]
print(f"\ntstop_15 kills {len(w)} big winners, base +{w.pnl_pct.sum():.2f}u -> {w.sim_tstop_15.sum():.2f}u; top:")
print(w.nlargest(5, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "sim_tstop_15", "mae_pct", "holding_days"]].to_string(index=False))

# bev_lock_8 big-winner kills
w = P[(P.pnl_pct >= 0.15) & P.sim_bev_lock_8.notna()]
print(f"\nbev_lock_8 kills {len(w)} big winners, base +{w.pnl_pct.sum():.2f}u -> {w.sim_bev_lock_8.sum():.2f}u")

# share of 59 bl15 in three crash months
bl = bl.assign(xm=bl.exit_date.dt.strftime("%Y-%m"))
crash = bl.xm.isin(["2020-03", "2022-06", "2022-10"])
print(f"\nbl15 in 2020-03/2022-06/2022-10 exits: {crash.sum()}/59 = {crash.mean():.1%}, sum {bl[crash].pnl_pct.sum():.2f}u")
