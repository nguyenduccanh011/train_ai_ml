# -*- coding: utf-8 -*-
"""pureml step 5: autopsy trade-level t1058 (seed 42) vs gb_x08 (seed 42).

- per-year pnl/PF/WR/hold (theo nam ENTRY va nam EXIT)
- exit_reason mix
- join symbol+entry_date voi gb_x08: overlap, cohort khac
- diem chet lon nhat + ngach thang gb_x08
"""
import numpy as np
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml"
RID_1058 = "template/n2_sx_rr_h10_thr2p5-570ad8d5"
GB_CSV = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap\gbx08_s42_trades.csv"

con = psycopg2.connect(**PG)
t = pd.read_sql(f"select symbol, entry_date, exit_date, entry_price, exit_price, holding_days, "
                f"pnl_pct, exit_reason, entry_signal_date from run_trades where run_id='{RID_1058}'", con)
con.close()
t.to_csv(f"{OUT}\\pm_t1058_s42_trades.csv", index=False)

g = pd.read_csv(GB_CSV, parse_dates=["entry_date", "exit_date"])
for df in (t,):
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

def pf(s):
    gains = s[s > 0].sum(); losses = -s[s <= 0].sum()
    return gains / losses if losses > 0 else np.inf

def yearly(df, datecol):
    grp = df.groupby(df[datecol].dt.year)["pnl_pct"]
    out = pd.DataFrame({
        "n": grp.size(), "pnl_sum": grp.sum().round(2), "wr": grp.apply(lambda s: (s > 0).mean()).round(3),
        "pf": grp.apply(pf).round(2), "avg": grp.mean().round(4),
        "med_hold": df.groupby(df[datecol].dt.year)["holding_days"].median(),
        "worst": grp.min().round(3),
    })
    return out

print(f"t1058 n={len(t)} pnl_sum={t.pnl_pct.sum():.2f} | gb_x08 n={len(g)} pnl_sum={g.pnl_pct.sum():.2f}")
print("\n== t1058 per ENTRY-year ==");  print(yearly(t, "entry_date").to_string())
print("\n== t1058 per EXIT-year ==");   print(yearly(t, "exit_date").to_string())
print("\n== gb_x08 per ENTRY-year =="); print(yearly(g, "entry_date").to_string())
print("\n== exit_reason mix t1058 =="); print(t.groupby("exit_reason")["pnl_pct"].agg(["size", "sum", "mean"]).to_string())
print("\n== exit_reason mix gb_x08 =="); print(g.groupby("exit_reason")["pnl_pct"].agg(["size", "sum", "mean"]).to_string())

# hold distribution
print("\n== hold quantiles t1058:", t.holding_days.quantile([.25, .5, .75, .9, .99]).to_dict())
print("== hold quantiles gb_x08:", g.holding_days.quantile([.25, .5, .75, .9, .99]).to_dict())

# ===== join on symbol + entry_date (exact) and ±3 days fuzzy =====
tj = t.copy(); gj = g.copy()
tj["key"] = tj.symbol + "|" + tj.entry_date.dt.strftime("%Y-%m-%d")
gj["key"] = gj.symbol + "|" + gj.entry_date.dt.strftime("%Y-%m-%d")
exact = set(tj.key) & set(gj.key)
print(f"\n== entry overlap exact: {len(exact)} = {len(exact)/len(t):.1%} of t1058, {len(exact)/len(g):.1%} of gb_x08")

# fuzzy ±5d
gsym = {s: df.entry_date.values for s, df in gj.groupby("symbol")}
def fuzzy_match(row):
    arr = gsym.get(row.symbol)
    if arr is None: return False
    return bool((np.abs(arr - np.datetime64(row.entry_date)) <= np.timedelta64(5, "D")).any())
tj["fuzzy"] = tj.apply(fuzzy_match, axis=1)
print(f"== entry overlap fuzzy(±5d): {tj.fuzzy.mean():.1%} of t1058 entries have a gb_x08 entry nearby")

m = tj.merge(gj, on="key", suffixes=("_t", "_g"))
if len(m):
    m["d_pnl"] = m.pnl_pct_t - m.pnl_pct_g
    m["d_hold"] = m.holding_days_t - m.holding_days_g
    print(f"\n== shared trades n={len(m)}: t1058 pnl_sum={m.pnl_pct_t.sum():.2f} vs gb_x08 {m.pnl_pct_g.sum():.2f} "
          f"(d={m.d_pnl.sum():+.2f}); mean d_hold={m.d_hold.mean():+.1f}d")
    print("shared per-year d_pnl:")
    print(m.groupby(m.entry_date_t.dt.year).agg(n=("d_pnl", "size"), t1058=("pnl_pct_t", "sum"),
          gbx08=("pnl_pct_g", "sum"), d=("d_pnl", "sum"), d_hold=("d_hold", "mean")).round(2).to_string())
    print("\ntop-10 shared where t1058 WINS:")
    print(m.nlargest(10, "d_pnl")[["key", "pnl_pct_t", "pnl_pct_g", "holding_days_t", "holding_days_g", "exit_reason_g"]].round(3).to_string(index=False))
    print("\ntop-10 shared where t1058 LOSES:")
    print(m.nsmallest(10, "d_pnl")[["key", "pnl_pct_t", "pnl_pct_g", "holding_days_t", "holding_days_g", "exit_reason_g"]].round(3).to_string(index=False))

only_t = tj[~tj.key.isin(exact)]
only_g = gj[~gj.key.isin(exact)]
print(f"\n== t1058-only entries n={len(only_t)} pnl_sum={only_t.pnl_pct.sum():.2f} wr={(only_t.pnl_pct>0).mean():.3f} pf={pf(only_t.pnl_pct):.2f}")
print(only_t.groupby(only_t.entry_date.dt.year)["pnl_pct"].agg(["size", "sum"]).round(2).to_string())
print(f"\n== gb_x08-only entries n={len(only_g)} pnl_sum={only_g.pnl_pct.sum():.2f} wr={(only_g.pnl_pct>0).mean():.3f} pf={pf(only_g.pnl_pct):.2f}")
print(only_g.groupby(only_g.entry_date.dt.year)["pnl_pct"].agg(["size", "sum"]).round(2).to_string())

# ===== diem chet: top losers t1058 =====
print("\n== top-15 losers t1058 ==")
print(t.nsmallest(15, "pnl_pct")[["symbol", "entry_date", "exit_date", "holding_days", "pnl_pct", "exit_reason"]].round(3).to_string(index=False))

# giveback proxy khong co MFE -> dung hold dai + pnl am
long_losers = t[(t.holding_days > 120) & (t.pnl_pct < -0.15)]
print(f"\n== long-hold (>120d) deep losers (<-15%): n={len(long_losers)} pnl_sum={long_losers.pnl_pct.sum():.2f}")
print(long_losers.groupby(long_losers.entry_date.dt.year)["pnl_pct"].agg(["size", "sum"]).round(2).to_string())

# tail phan phoi pnl
print("\n== pnl quantiles t1058:", t.pnl_pct.quantile([.01, .05, .25, .5, .75, .95, .99]).round(3).to_dict())
print("== pnl quantiles gb_x08:", g.pnl_pct.quantile([.01, .05, .25, .5, .75, .95, .99]).round(3).to_dict())

# ngach t1058 thang: winners lon nhat cua t1058 va lieu gb_x08 co bat duoc khong
tw = t.nlargest(15, "pnl_pct").copy()
def gb_same(row):
    sub = gj[(gj.symbol == row.symbol) & (abs(gj.entry_date - row.entry_date) <= pd.Timedelta(days=10))]
    if len(sub) == 0: return "MISS"
    return f"{sub.pnl_pct.max():.2f}"
tw["gb_pnl_near"] = tw.apply(gb_same, axis=1)
print("\n== top-15 winners t1058 (vs gb_x08 entry cung ma ±10d) ==")
print(tw[["symbol", "entry_date", "exit_date", "holding_days", "pnl_pct", "gb_pnl_near"]].round(3).to_string(index=False))
print("PM05_DONE")
