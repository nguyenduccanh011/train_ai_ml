"""EXIT MAP gb_x08 — fast-fail cohort (hold<=5): would incubation (signal_exit_min_age) pay?"""
import numpy as np
import pandas as pd

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
E = pd.read_csv(EM + "/gbx08_enriched2.csv", parse_dates=["entry_date", "exit_date"])
pd.set_option("display.width", 220)
E["cf_end20_u"] = E.post_end_c * (1 + E.pnl_pct)
E["cf_peak_u"] = E.post_max_c * (1 + E.pnl_pct)

F = E[E.holding_days <= 5]
print(f"fast-fail hold<=5: n={len(F)} pnl={F.pnl_pct.sum():+.1f} "
      f"cf_end20={F.cf_end20_u.sum():+.1f}u cf_peak={F.cf_peak_u.sum():+.1f}u")
print("\nby exit year:")
print(F.groupby("year_exit").apply(lambda g: pd.Series(dict(
    n=len(g), pnl=g.pnl_pct.sum(), wr=(g.pnl_pct > 0).mean(),
    cf_end20_u=g.cf_end20_u.sum()))).round(2).to_string())
print("\nby rule:")
print(F.groupby("label").apply(lambda g: pd.Series(dict(
    n=len(g), pnl=g.pnl_pct.sum(), cf_end20_u=g.cf_end20_u.sum()))).round(2).to_string())
print("\nsplit by pnl sign at exit:")
for nm, G in [("losers", F[F.pnl_pct <= 0]), ("winners", F[F.pnl_pct > 0])]:
    print(f"  {nm}: n={len(G)} pnl={G.pnl_pct.sum():+.1f} cf_end20={G.cf_end20_u.sum():+.1f}u")
    print(G.groupby("year_exit").cf_end20_u.sum().round(2).to_dict())

# also: hold 6-10 for min_age up to 10
F2 = E[(E.holding_days >= 6) & (E.holding_days <= 10)]
print(f"\nhold 6-10: n={len(F2)} pnl={F2.pnl_pct.sum():+.1f} cf_end20={F2.cf_end20_u.sum():+.1f}u")
print(F2.groupby("year_exit").cf_end20_u.sum().round(2).to_dict())
