import pandas as pd
import numpy as np

BASE = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/week0/best_trades/"
cand = pd.read_csv(BASE + "trades_w0_pyr_u10_r02.csv", parse_dates=["entry_date", "exit_date"])
champ = pd.read_csv(BASE + "trades_n2_2643_wavestruct_la05_lamp02.csv", parse_dates=["entry_date", "exit_date"])

print("n cand:", len(cand), " n champ:", len(champ))

key = ["symbol", "entry_date", "exit_date"]
m = cand.merge(champ[key + ["pnl_pct", "entry_price", "exit_price", "holding_days", "exit_reason"]],
               on=key, suffixes=("_cand", "_champ"), how="inner")
print("matched on (symbol,entry,exit):", len(m))

add = m[m["weight"] > 1.0].copy()
noadd = m[m["weight"] <= 1.0].copy()
print("\nadd cohort n =", len(add), " noadd n =", len(noadd))

# sanity: timing identical?
print("entry_price identical:", np.allclose(m["entry_price_cand"], m["entry_price_champ"]))
print("exit_price identical:", np.allclose(m["exit_price_cand"], m["exit_price_champ"]))
print("noadd pnl identical:", np.allclose(noadd["pnl_pct_cand"], noadd["pnl_pct_champ"]))

def stats(df, col):
    wr = (df[col] > 0).mean()
    return f"WR={wr:.3f} mean={df[col].mean():+.4f} median={df[col].median():+.4f} sum={df[col].sum():+.2f}"

print("\n--- CANDIDATE frame (pyramided pnl) ---")
print("add   :", stats(add, "pnl_pct_cand"))
print("noadd :", stats(noadd, "pnl_pct_cand"))
print("\n--- CHAMPION frame (same trades, base-only pnl) ---")
print("add cohort   :", stats(add, "pnl_pct_champ"))
print("noadd cohort :", stats(noadd, "pnl_pct_champ"))

# implied add-leg net return: cand pnl = champ pnl + 1.0 * add_net
add["add_net"] = add["pnl_pct_cand"] - add["pnl_pct_champ"]
print("\n--- implied add-leg net (units=1.0) ---")
print("n =", len(add))
print("add_net > 0:", (add["add_net"] > 0).sum(), f"({(add['add_net']>0).mean():.3f})")
print("add_net mean:", f"{add['add_net'].mean():+.4f}", " sum:", f"{add['add_net'].sum():+.2f}")
print("add_net worst 5:")
print(add.nsmallest(5, "add_net")[["symbol", "entry_date", "exit_date", "pnl_pct_champ", "pnl_pct_cand", "add_net", "exit_reason_cand"]].to_string(index=False))

# losing-add path: base pnl also negative (surged >=2% by bar3 then collapsed)
lose_both = add[(add["pnl_pct_champ"] < 0)]
print("\nadd cohort with NEGATIVE base pnl (surge-then-collapse):", len(lose_both),
      " mean cand pnl:", f"{lose_both['pnl_pct_cand'].mean():+.4f}",
      " mean champ pnl:", f"{lose_both['pnl_pct_champ'].mean():+.4f}")
if len(lose_both):
    ex = lose_both.nsmallest(1, "pnl_pct_cand").iloc[0]
    print("worst example:", ex["symbol"], ex["entry_date"].date(), "->", ex["exit_date"].date(),
          f"champ={ex['pnl_pct_champ']:+.4f} cand={ex['pnl_pct_cand']:+.4f} add_net={ex['add_net']:+.4f} reason={ex['exit_reason_cand']}")

# totals
print("\ntotal pnl cand:", f"{cand['pnl_pct'].sum():+.2f}", " champ:", f"{champ['pnl_pct'].sum():+.2f}",
      " delta:", f"{cand['pnl_pct'].sum() - champ['pnl_pct'].sum():+.2f}")

# holding days of add cohort (must be > 3 by construction)
print("min holding_days in add cohort:", add["holding_days_cand"].min())

# exit reasons of add cohort
print("\nexit reasons add cohort:\n", add["exit_reason_cand"].value_counts().to_string())
