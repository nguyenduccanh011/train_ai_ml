"""Regime slices (entries >= 2022/2023/2024/2025) + per-entry-year decomposition
for rescue variants vs champion 2646 and candidate 2730 (all seed 42)."""
import pandas as pd

SQ = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/signalq"
A = SQ + "/autopsy"

sets = {
    "champ2646": pd.read_csv(SQ + "/st_champ2646_s42_trades.csv", parse_dates=["entry_date", "exit_date"]),
    "cand2730": pd.read_csv(SQ + "/sv_snr08_s42_trades.csv", parse_dates=["entry_date", "exit_date"]),
    "sx_g35": pd.read_csv(A + "/sx_g35_s42_trades.csv", parse_dates=["entry_date", "exit_date"]),
    "sx_w40": pd.read_csv(A + "/sx_w40_s42_trades.csv", parse_dates=["entry_date", "exit_date"]),
    "sx_w60": pd.read_csv(A + "/sx_w60_s42_trades.csv", parse_dates=["entry_date", "exit_date"]),
}

def pf(x):
    g = x[x > 0].sum(); l = -x[x <= 0].sum()
    return g / l if l > 0 else float("inf")

print("== per-entry-year pnl sum (n) ==")
years = sorted(set().union(*[set(df.entry_date.dt.year) for df in sets.values()]))
hdr = "year  " + "".join(f"{k:>18s}" for k in sets)
print(hdr)
for y in years:
    row = f"{y}  "
    for k, df in sets.items():
        s = df[df.entry_date.dt.year == y]
        row += f"{s.pnl_pct.sum():+9.2f} ({len(s):4d})"
    print(row)

print("\n== regime slices: entries >= Y ==")
for y in [2022, 2023, 2024, 2025]:
    print(f"\n-- entries >= {y} --")
    for k, df in sets.items():
        s = df[df.entry_date.dt.year >= y]
        print(f"  {k:10s}: n={len(s):4d} pnl={s.pnl_pct.sum():+8.2f} pf={pf(s.pnl_pct):6.3f} "
              f"wr={(s.pnl_pct>0).mean()*100:4.1f}% avg_hold={s.holding_days.mean():5.1f}")

print("\n== w60 vs champ: biggest single-trade deltas (matched entries) ==")
ch = sets["champ2646"]; w = sets["sx_w60"]
m = ch.merge(w, on=["symbol", "entry_date"], how="outer", suffixes=("_c", "_w"), indicator=True)
both = m[m._merge == "both"]
d = both[both.exit_date_c != both.exit_date_w].copy()
d["dpnl"] = d.pnl_pct_w - d.pnl_pct_c
print(f"deferred-diff pairs: {len(d)} sum dpnl={d.dpnl.sum():+.2f} | "
      f"champ-only n={(m._merge=='left_only').sum()} pnl={m[m._merge=='left_only'].pnl_pct_c.sum():+.2f} | "
      f"w60-only n={(m._merge=='right_only').sum()} pnl={m[m._merge=='right_only'].pnl_pct_w.sum():+.2f}")
d["yr"] = d.entry_date.dt.year
print(d.groupby("yr").dpnl.agg(["sum", "count"]).round(2))
print("\ntop 10 w60 gains:")
print(d.nlargest(10, "dpnl")[["symbol", "entry_date", "exit_date_c", "exit_date_w", "pnl_pct_c", "pnl_pct_w", "dpnl"]].to_string(index=False))
print("\nworst 10 w60:")
print(d.nsmallest(10, "dpnl")[["symbol", "entry_date", "exit_date_c", "exit_date_w", "pnl_pct_c", "pnl_pct_w", "dpnl"]].to_string(index=False))
