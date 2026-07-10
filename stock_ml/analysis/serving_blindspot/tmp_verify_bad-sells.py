# -*- coding: utf-8 -*-
"""Adversarial re-verification of 14 bad-sell claims on trades_metrics.csv."""
import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
tm = pd.read_csv(OUT + r"\trades_metrics.csv")
da = pd.read_csv(OUT + r"\daily_activity.csv")
tm["entry_date"] = pd.to_datetime(tm["entry_date"])
tm["exit_date"] = pd.to_datetime(tm["exit_date"])

N = len(tm)
ru = tm["post_max_runup21"]
print(f"total closed trades: {N}")
print(f"exit_reason counts: {tm['exit_reason'].value_counts().to_dict()}")
print(f"realized sum pnl_pct: {tm['pnl_pct'].sum():+.2f}u")

# ---------- CLAIM 1: overall sold-too-early rates ----------
print("\n=== C1 overall ===")
for th in (0.05, 0.10, 0.15):
    m = (ru >= th)
    print(f"  ge{int(th*100)}: n={m.sum()}  rate={m.mean():.4f} (denom all {N})  "
          f"rate_nonnan={(ru.dropna() >= th).mean():.4f} (denom {ru.notna().sum()})")
print(f"  runup21 mean={ru.mean():.4f} median={ru.median():.4f} "
      f"pos_rate={(ru > 0).mean():.4f} (nonnan {(ru.dropna() > 0).mean():.4f})  NaN={ru.isna().sum()}")
print(f"  sold_too_early col sum={tm['sold_too_early'].sum()} rate={tm['sold_too_early'].mean():.4f}")

# ---------- CLAIM 2 / 6: per exit_reason distributions ----------
print("\n=== C2/C6 by exit_reason ===")
for reason, g in tm.groupby("exit_reason"):
    r = g["post_max_runup21"]
    print(f"  {reason}: n={len(g)}")
    for c in ("post_ret5", "post_ret10", "post_ret21", "post_ret42"):
        s = g[c]
        print(f"    {c}: n={s.notna().sum()} mean={s.mean():+.4f} median={s.median():+.4f} "
              f"pos={(s > 0).mean():.4f} sum={s.sum():+.2f}u")
    print(f"    runup21: mean={r.mean():+.4f} median={r.median():+.4f} p90={r.quantile(0.9):+.4f} "
          f"ge5={(r >= 0.05).mean():.4f} ge10={(r >= 0.10).mean():.4f} ge15={(r >= 0.15).mean():.4f}")
    print(f"    sold_too_early={g['sold_too_early'].mean():.4f} ({g['sold_too_early'].sum()}/{len(g)})")

# ---------- CLAIM 3: oracle runup volume by exit_reason ----------
print("\n=== C3 oracle clip0 sums ===")
tot = ru.clip(lower=0).sum()
print(f"  total oracle runup clip0: {tot:.2f}u")
for reason, g in tm.groupby("exit_reason"):
    s = g["post_max_runup21"].clip(lower=0).sum()
    print(f"  {reason}: sum={s:.2f}u share={s/tot:.4f} mean={g['post_max_runup21'].mean():+.4f}")

# ---------- CLAIM 4: blanket hold sums + NaN counts ----------
print("\n=== C4 blanket hold ===")
for c in ("post_ret5", "post_ret10", "post_ret21", "post_ret42"):
    print(f"  {c}: sum={tm[c].sum():+.2f}u mean={tm[c].mean():+.4f} "
          f"pos={(tm[c].dropna() > 0).mean():.4f} NaN={tm[c].isna().sum()}")

# trading-calendar bar index from daily_activity
cal = pd.to_datetime(da["date"]).sort_values().reset_index(drop=True)
cal_idx = {d: i for i, d in enumerate(cal)}
cal_arr = cal.values

def bar_idx(d):
    i = cal_idx.get(d)
    if i is not None:
        return i
    return int(np.searchsorted(cal_arr, np.datetime64(d)))

# ---------- CLAIM 4/9: rebuy analysis ----------
print("\n=== C4/C9 rebuys ===")
s = tm.sort_values(["symbol", "entry_date"]).reset_index(drop=True)
s["next_sym"] = s["symbol"].shift(-1)
s["next_entry_date"] = s["entry_date"].shift(-1)
s["next_entry_price"] = s["entry_price"].shift(-1)
pair = s[s["next_sym"] == s["symbol"]].copy()
pair["gap_bars"] = [bar_idx(ne) - bar_idx(xd) for ne, xd in zip(pair["next_entry_date"], pair["exit_date"])]
pair["premium"] = pair["next_entry_price"] / pair["exit_price"] - 1.0

for gcap in (40, 15):
    r = pair[pair["gap_bars"] <= gcap]
    print(f"  gap<={gcap}: n={len(r)} ({len(r)/N:.4f} of {N} exits)  "
          f"higher={(r['premium'] > 0).mean():.4f}  mean_premium={r['premium'].mean():+.4f}")
r40 = pair[pair["gap_bars"] <= 40]
ste = r40[r40["sold_too_early"]]
print(f"  sold_too_early & gap<=40: n={len(ste)} higher={(ste['premium'] > 0).mean():.4f} "
      f"mean_premium={ste['premium'].mean():+.4f}")
lo = r40[r40["pnl_pct"] <= 0]
print(f"  loss-exit (pnl<=0) & gap<=40: n={len(lo)} higher={(lo['premium'] > 0).mean():.4f} "
      f"mean_premium={lo['premium'].mean():+.4f}")
lo2 = r40[r40["pnl_pct"] < 0]
print(f"  loss-exit (pnl<0)  & gap<=40: n={len(lo2)} higher={(lo2['premium'] > 0).mean():.4f}")

# ---------- CLAIM 5: +10-bar delta by year / bucket / winner ----------
print("\n=== C5 post_ret10 by exit year ===")
tm["year"] = tm["exit_date"].dt.year
for y, g in tm.groupby("year"):
    print(f"  {y}: n={len(g)} sum={g['post_ret10'].sum():+.2f}u mean={g['post_ret10'].mean():+.4f}")
w22 = tm[(tm["year"] == 2022) & (tm["pnl_pct"] > 0)]
print(f"  2022 winners: n={len(w22)} sum={w22['post_ret10'].sum():+.2f}u")
print("  by bucket:")
for b, g in tm.groupby("bucket"):
    print(f"    {b}: n={len(g)} sum={g['post_ret10'].sum():+.2f}u")
wn = tm[tm["pnl_pct"] > 0]; ls_ = tm[tm["pnl_pct"] <= 0]
print(f"  winners all: sum={wn['post_ret10'].sum():+.2f}u | losers all: sum={ls_['post_ret10'].sum():+.2f}u")

# ---------- CLAIM 7: sold_too_early cohort characteristics ----------
print("\n=== C7 cohort characteristics ===")
g = tm.groupby("sold_too_early")
cols = ["pre_ret5", "pre_ret20", "pre_ret60", "holding_days", "mfe_pct", "wait_bars",
        "entry_dist_sma20", "pnl_pct"]
print(g[cols].mean().T.round(4))
print("  hot_run share:", g["hot_run"].mean().round(4).to_dict())
print("  sold-early rate by hot_run:", tm.groupby("hot_run")["sold_too_early"].mean().round(4).to_dict())
print("  sold-early rate by bucket:", tm.groupby("bucket")["sold_too_early"].mean().round(4).to_dict())
print("  hold<=8 share:", tm.groupby("sold_too_early")["holding_days"].apply(lambda x: (x <= 8).mean()).round(4).to_dict())
hc = pd.cut(tm["holding_days"], [-1, 8, 16, 32, 64, 10**9], labels=["<=8", "9-16", "17-32", "33-64", ">=65"])
print("  sold rate by hold bin:", tm.groupby(hc, observed=True)["sold_too_early"].mean().round(4).to_dict())
hc2 = pd.cut(tm["holding_days"], [-1, 7, 15, 31, 63, 10**9], labels=["<=7", "8-15", "16-31", "32-63", ">=64"])
print("  sold rate by hold bin (>=64 variant):", tm.groupby(hc2, observed=True)["sold_too_early"].mean().round(4).to_dict())

# ---------- CLAIM 8: cut-on-dip within sold cohort ----------
print("\n=== C8 cut-on-dip ===")
st = tm[tm["sold_too_early"]]
print(f"  sold cohort n={len(st)}")
seg = {"cut_on_dip(gb<=-5%)": st[st["giveback_pct"] <= -0.05],
       "mid": st[(st["giveback_pct"] > -0.05) & (st["giveback_pct"] <= -0.02)],
       "near_peak(gb>-2%)": st[st["giveback_pct"] > -0.02]}
for k, d in seg.items():
    print(f"  {k}: n={len(d)} ({len(d)/len(st):.4f}) pnl_mean={d['pnl_pct'].mean():+.4f} "
          f"mfe_mean={d['mfe_pct'].mean():+.4f} post_ret10_mean={d['post_ret10'].mean():+.4f} "
          f"runup21_mean={d['post_max_runup21'].mean():+.4f} hold_median={d['holding_days'].median()}")

# ---------- CLAIM 10: signal-exit winners/losers runup thresholds ----------
print("\n=== C10 signal winners/losers ===")
sig = tm[tm["exit_reason"] == "signal"]
for lbl, d in (("winners", sig[sig["pnl_pct"] > 0]), ("losers", sig[sig["pnl_pct"] <= 0])):
    r = d["post_max_runup21"]
    print(f"  {lbl}: n={len(d)} ge2={(r >= 0.02).mean():.4f} ge3={(r >= 0.03).mean():.4f} "
          f"ge5={(r >= 0.05).mean():.4f} mean={r.mean():+.4f}")

# ---------- CLAIM 11: mfe band [10%,27%) ----------
print("\n=== C11 mfe band ===")
band = tm[(tm["mfe_pct"] >= 0.10) & (tm["mfe_pct"] < 0.27)]
print(f"  [10,27): n={len(band)} share={len(band)/N:.4f} pnl_sum={band['pnl_pct'].sum():+.2f}u "
      f"gb_sum={band['giveback_pct'].sum():+.2f}u gb_mean={band['giveback_pct'].mean():+.4f} "
      f"sold={band['sold_too_early'].mean():.4f}")
hi = tm[tm["mfe_pct"] >= 0.10]
print(f"  mfe>=10%: n={len(hi)} gb_sum={hi['giveback_pct'].sum():+.2f}u")

# ---------- CLAIM 12: signal exits mfe>=27% + trailing_stop table ----------
print("\n=== C12 signal mfe>=27% ===")
big = sig[sig["mfe_pct"] >= 0.27]
print(f"  n={len(big)} gb_sum={big['giveback_pct'].sum():+.2f}u gb_mean={big['giveback_pct'].mean():+.4f} "
      f"sold={big['sold_too_early'].mean():.4f} runup21_mean={big['post_max_runup21'].mean():+.4f} "
      f"post_ret10_sum={big['post_ret10'].sum():+.2f}u")
ts = tm[tm["exit_reason"] == "trailing_stop"]
print(f"  trailing_stop n={len(ts)}:")
print(ts[["symbol", "entry_date", "exit_date", "mfe_pct", "giveback_pct",
          "post_max_runup21", "sold_too_early"]].to_string(index=False))

# ---------- CLAIM 13: sold rate by year ----------
print("\n=== C13 sold rate by exit year ===")
print(tm.groupby("year")["sold_too_early"].agg(["mean", "count"]).round(4).to_string())

# ---------- CLAIM 14: hold<=4 churn ----------
print("\n=== C14 hold<=4 ===")
sh = tm[tm["holding_days"] <= 4]
print(f"  n={len(sh)} pnl_sum={sh['pnl_pct'].sum():+.2f}u "
      f"runup_clip0_sum={sh['post_max_runup21'].clip(lower=0).sum():+.2f}u "
      f"sold={sh['sold_too_early'].mean():.4f}")

# ---------- lookahead sanity: post_* columns strictly after exit bar ----------
print("\n=== sanity ===")
same = (tm["post_ret5"] == 0).sum()
print(f"  post_ret5 exactly 0: {same} (should be rare)")
print(f"  sold_too_early == (runup>=5%) mismatch: "
      f"{(tm['sold_too_early'] != (ru.fillna(-9) >= 0.05)).sum()}")
