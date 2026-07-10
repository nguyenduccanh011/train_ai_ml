# -*- coding: utf-8 -*-
"""un03: occupancy autopsy — so trades cua bien the un_ voi gb_x08 s42.

Phan loai theo huong bs_/bch_: (a) NEW entries (khong co trong gb, fuzzy +/-5d cung ma);
(b) LOST gb entries (gb co, un khong — bi displacement/threshold); (c) MATCHED (pnl drift
do exit khac / entry lech). Per-year + tong. Usage: python un03_autopsy.py <un_csv> <label>
"""
import sys
import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
un = pd.read_csv(sys.argv[1], parse_dates=["entry_date", "exit_date"])
label = sys.argv[2] if len(sys.argv) > 2 else "un"
gb = pd.read_csv(BASE + r"\exitmap\gbx08_s42_trades.csv", parse_dates=["entry_date", "exit_date"])

for df in (un, gb):
    df["year"] = df["entry_date"].dt.year

def match_map(a, b):
    """for each row in a: has b a trade same symbol entry within 5d?"""
    b_sym = {s: g["entry_date"].values for s, g in b.groupby("symbol")}
    out = []
    for s, d in zip(a["symbol"], a["entry_date"]):
        arr = b_sym.get(s)
        out.append(bool(arr is not None and
                        (np.abs((arr - np.datetime64(d)) / np.timedelta64(1, "D")) <= 5).any()))
    return np.array(out)

un["in_gb"] = match_map(un, gb)
gb["in_un"] = match_map(gb, un)

print(f"== {label} vs gb_x08 s42 ==")
print(f"{label}: {len(un)} trades pnl {un.pnl_pct.sum():.2f} | gb: {len(gb)} trades pnl {gb.pnl_pct.sum():.2f}")
print(f"matched(fuzzy5d): {un.in_gb.sum()} | NEW({label}-only): {(~un.in_gb).sum()} "
      f"pnl {un.loc[~un.in_gb,'pnl_pct'].sum():.2f} | LOST(gb-only): {(~gb.in_un).sum()} "
      f"gb_pnl {gb.loc[~gb.in_un,'pnl_pct'].sum():.2f}")

tbl = pd.DataFrame({
    f"n_{label}": un.groupby("year").size(),
    "n_gb": gb.groupby("year").size(),
    f"pnl_{label}": un.groupby("year")["pnl_pct"].sum().round(2),
    "pnl_gb": gb.groupby("year")["pnl_pct"].sum().round(2),
    "new_n": un[~un.in_gb].groupby("year").size(),
    "new_pnl": un[~un.in_gb].groupby("year")["pnl_pct"].sum().round(2),
    "lost_n": gb[~gb.in_un].groupby("year").size(),
    "lost_gbpnl": gb[~gb.in_un].groupby("year")["pnl_pct"].sum().round(2),
}).fillna(0)
print(tbl.to_string())

# matched pnl drift (exit khac nhau tren cung cohort)
m_un = un[un.in_gb]; m_gb = gb[gb.in_un]
print(f"\nmatched cohort pnl: {label} {m_un.pnl_pct.sum():.2f} (n={len(m_un)}) "
      f"vs gb {m_gb.pnl_pct.sum():.2f} (n={len(m_gb)})")

# top LOST premium entries (gb winners bi mat)
lost = gb[~gb.in_un].nlargest(12, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "exit_reason", "holding_days"]]
print("\ntop-12 LOST gb premium entries:")
print(lost.to_string(index=False))

# top NEW winners/losers
new = un[~un.in_gb]
print("\ntop-8 NEW winners:")
print(new.nlargest(8, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "exit_reason"]].to_string(index=False))
print("top-8 NEW losers:")
print(new.nsmallest(8, "pnl_pct")[["symbol", "entry_date", "pnl_pct", "exit_reason"]].to_string(index=False))

# hold + exit_reason mix shift
print(f"\nhold median: {label} {un.holding_days.median():.0f} vs gb {gb.holding_days.median():.0f}")
print(f"exit_reason {label}: {un.exit_reason.value_counts().to_dict()}")
print(f"exit_reason gb : {gb.exit_reason.value_counts().to_dict()}")
