"""Diagnostics: st_dsb60_snr08 seed-42 trades vs champion 2646 seed-42 trades.
Per-year pnl delta, exit_reason mix shift, big_loss/big_win counts, mdd inputs.
Usage: python stock_ml/analysis/serving_blindspot/signalq/st_diag.py <stack_csv> <champ_csv>
"""
import sys
import pandas as pd

stack = pd.read_csv(sys.argv[1], parse_dates=["entry_date", "exit_date"])
champ = pd.read_csv(sys.argv[2], parse_dates=["entry_date", "exit_date"])

for nm, df in [("stack", stack), ("champ", champ)]:
    print(f"{nm}: trades={len(df)} pnl_sum={df.pnl_pct.sum():.2f}")

print("\n== per-YEAR pnl (by exit year) ==")
sy = stack.groupby(stack.exit_date.dt.year).pnl_pct.agg(["sum", "count"])
cy = champ.groupby(champ.exit_date.dt.year).pnl_pct.agg(["sum", "count"])
yr = sy.join(cy, lsuffix="_stack", rsuffix="_champ", how="outer").fillna(0)
yr["d_pnl"] = yr.sum_stack - yr.sum_champ
for y, r in yr.iterrows():
    print(f"  {y}: stack {r.sum_stack:7.2f} ({int(r.count_stack):4d} tr) | champ {r.sum_champ:7.2f} "
          f"({int(r.count_champ):4d} tr) | d {r.d_pnl:+7.2f}")

print("\n== exit_reason mix ==")
sm = stack.exit_reason.value_counts()
cm = champ.exit_reason.value_counts()
mix = pd.DataFrame({"stack": sm, "champ": cm}).fillna(0).astype(int)
mix["d"] = mix["stack"] - mix["champ"]
mix = mix.sort_values("d")
for reason, r in mix.iterrows():
    print(f"  {reason:35s} stack {r['stack']:5d} champ {r['champ']:5d} d {r['d']:+5d}")

print("\n== tails ==")
for nm, df in [("stack", stack), ("champ", champ)]:
    bl = (df.pnl_pct <= -0.10).sum()
    bw = (df.pnl_pct >= 0.10).sum()
    bw_pnl = df.loc[df.pnl_pct >= 0.10, "pnl_pct"].sum()
    print(f"  {nm}: big_loss(<=-10%)={bl}  big_win(>=+10%)={bw} ({bw/len(df)*100:.1f}%)  "
          f"bigwin_pnl={bw_pnl:.2f}  avg_hold={df.holding_days.mean():.1f}")

print("\n== worst 5 trades (stack) ==")
for _, r in stack.nsmallest(5, "pnl_pct").iterrows():
    print(f"  {r.symbol} {r.entry_date.date()} -> {r.exit_date.date()} {r.pnl_pct:+.3f} {r.exit_reason}")
