# -*- coding: utf-8 -*-
"""Adversarial recompute of 14 claims from tmp_goodtrades_01/02 on serving_blindspot data."""
import pandas as pd, numpy as np

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
D = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/"

t = pd.read_csv(D + "trades_metrics.csv", parse_dates=["entry_date", "exit_date", "signal_date"])
n = len(t); tot = t.pnl_pct.sum()
print(f"[BASE] closed trades n={n}  total pnl={tot:.2f}u")

sig = pd.read_csv(D + "signals.csv", parse_dates=["date"],
                  usecols=["symbol", "date", "score", "score3", "entry_csr"])
dup = sig.duplicated(["symbol", "date"]).sum()
t = t.merge(sig.rename(columns={"date": "signal_date"}), on=["symbol", "signal_date"], how="left")
print(f"[JOIN] rows after merge={len(t)} (dup sig keys={dup}), score miss={t.score.isna().sum()}")
t["year"] = t.entry_date.dt.year
bw = t.bucket == "big_win"

def stats(s, label=""):
    return (f"n={len(s)} ({len(s)/n:.1%}) sum={s.pnl_pct.sum():+.2f}u ({s.pnl_pct.sum()/tot:.1%}) "
            f"mean={s.pnl_pct.mean():+.4f} wr={(s.pnl_pct>0).mean():.3f} "
            f"bigwin%={(s.bucket=='big_win').mean():.3f} hold_med={s.holding_days.median()}")

# ---------- CLAIM 1: bucket profile ----------
print("\n===== C1 BUCKET PROFILE =====")
g = t.groupby("bucket").agg(n=("pnl_pct", "size"), sum=("pnl_pct", "sum"),
    hold_med=("holding_days", "median"), wait_med=("wait_bars", "median"),
    mfe_med=("mfe_pct", "median"), mae_med=("mae_pct", "median"),
    below_ma20=("entry_dist_sma20", lambda x: (x < 0).mean()))
print(g.round(4))
nb = t[~bw]
print(f"big_win share of total: {t.pnl_pct[bw].sum()/tot:.1%}; non-bigwin: n={len(nb)} sum={nb.pnl_pct.sum():+.2f}u")

# ---------- CLAIM 2: hold split ----------
print("\n===== C2 HOLD SPLIT =====")
hs = pd.cut(t.holding_days, [0, 8, 20, 10000], labels=["<=8", "9-20", ">20"])
print(t.groupby(hs, observed=True).agg(n=("pnl_pct", "size"), sum=("pnl_pct", "sum"),
      mean=("pnl_pct", "mean"), wr=("pnl_pct", lambda x: (x > 0).mean())).round(4))
h20 = t[t.holding_days > 20]
print(f">20 share of trades {len(h20)/n:.1%}, share of pnl {h20.pnl_pct.sum()/tot:.1%}")
print(f"holding_days==0 rows (excluded by cut left edge): {(t.holding_days==0).sum()}")

# ---------- CLAIMS 3,4,6,13,12: quintiles ----------
def quint(f):
    q = pd.qcut(t[f], 5, duplicates="drop")
    g = t.groupby(q, observed=True).agg(n=("pnl_pct", "size"), mean=("pnl_pct", "mean"),
        wr=("pnl_pct", lambda x: (x > 0).mean()), sum=("pnl_pct", "sum"),
        bigwin=("bucket", lambda x: (x == "big_win").mean()))
    print(f"\n-- {f} quintiles --"); print(g.round(4))
    return g

print("\n===== C3 wait_bars =====");        quint("wait_bars")
print("\n===== C4 entry_dist_sma20 ====="); quint("entry_dist_sma20")
below = t[t.entry_dist_sma20 < 0]
print(f"below-MA20 cohort: n={len(below)} sum={below.pnl_pct.sum():+.2f}u")
print("\n===== C6 signal_to_fill_drop ====="); quint("signal_to_fill_drop")
full_exact = t[np.isclose(t.signal_to_fill_drop, -0.045, atol=1e-9)]
full_le = t[t.signal_to_fill_drop <= -0.045 + 1e-12]
print(f"full-depth ==-4.5%: {stats(full_exact)}")
print(f"full-depth <=-4.5%: {stats(full_le)}")
print("\n===== C13 pre_ret20 =====");       quint("pre_ret20")
hr = t.groupby("hot_run").pnl_pct.agg(["size", "mean", "sum"]); print(hr.round(4))
print("\n===== C12 score / score3 / entry_csr =====")
quint("score"); quint("score3"); quint("entry_csr")

# ---------- CLAIM 5: core edge slice ----------
print("\n===== C5 CORE EDGE SLICE =====")
core = (t.wait_bars <= 4) & (t.entry_dist_sma20 >= 0)
print("wait<=4 & dist>=0:", stats(t[core]))
for y, g_ in t[core].groupby("year"):
    print(f"  {y}: n={len(g_)} sum={g_.pnl_pct.sum():+.2f}u")
prem = (~t.touch_bar_red) & (t.entry_dist_sma20 >= 0)
print("green-touch & dist>=0:", stats(t[prem]))

# ---------- CLAIM 7: churn / friction ----------
print("\n===== C7 CHURN =====")
churn = t[t.pnl_pct.abs() < 0.03]
print(f"churn n={len(churn)} ({len(churn)/n:.1%}) sum={churn.pnl_pct.sum():+.2f}u "
      f"mean={churn.pnl_pct.mean():+.4f} hold_med={churn.holding_days.median()}")
print(f"friction inside churn: fee 0.004*{len(churn)}={0.004*len(churn):.1f}u, full 0.007*{len(churn)}={0.007*len(churn):.2f}u "
      f"=> gross-of-friction churn ~{churn.pnl_pct.sum()+0.007*len(churn):+.1f}u")
print(f"friction ALL: 0.007*{n}={0.007*n:.1f}u = {0.007*n/tot:.1%} of net pnl")
qe = t[t.holding_days <= 8]
print(f"hold<=8: n={len(qe)} sum={qe.pnl_pct.sum():+.2f}u churn_share={(qe.pnl_pct.abs()<0.03).mean():.1%}")

# ---------- CLAIM 8: churn filter efficiency ----------
print("\n===== C8 CHURN FILTER EFFICIENCY =====")
cands = {
    "wait_bars > 20": t.wait_bars > 20,
    "wait_bars > 30": t.wait_bars > 30,
    "wait_bars > 15": t.wait_bars > 15,
    "wait_bars > 10": t.wait_bars > 10,
    "dist < -0.04": t.entry_dist_sma20 < -0.04,
    "entry_csr < 0.3": t.entry_csr < 0.3,
    "score < q20": t.score < t.score.quantile(0.2),
    "pre_ret20 dead-zone": (t.pre_ret20 > -0.0329) & (t.pre_ret20 <= 0.02),
    "wait>15 & dist<-0.02": (t.wait_bars > 15) & (t.entry_dist_sma20 < -0.02),
}
rows = []
for name, m in cands.items():
    s = t[m]; shed = s.pnl_pct.sum(); ch = (s.pnl_pct.abs() < 0.03).sum()
    rows.append({"filter": name, "n_cut": len(s), "churn_cut": ch,
                 "pnl_shed": round(shed, 2), "shed_share": f"{shed/tot:.1%}",
                 "churn_per_u": round(ch / shed, 1) if shed > 0 else np.inf,
                 "bigwin_cut": int((s.bucket == "big_win").sum()),
                 "mean_cut": round(s.pnl_pct.mean(), 4)})
print(pd.DataFrame(rows).to_string(index=False))
w20 = t[t.wait_bars > 20]
print(f"wait>20 churn share of all churn: {(w20.pnl_pct.abs()<0.03).sum()/len(churn):.1%}")

# ---------- CLAIM 9: yearly PF / regime ----------
print("\n===== C9 YEARLY =====")
def yr(g_):
    gp = g_.pnl_pct[g_.pnl_pct > 0].sum(); gl = -g_.pnl_pct[g_.pnl_pct < 0].sum()
    return pd.Series({"n": len(g_), "sum": g_.pnl_pct.sum(),
                      "pf": gp / gl if gl > 0 else np.inf,
                      "bigwin_rate": (g_.bucket == "big_win").mean(),
                      "nonbig_sum": g_.pnl_pct[g_.bucket != "big_win"].sum()})
print(t.groupby("year").apply(yr).round(3))
raw = pd.read_csv(D + "trades_raw.csv", parse_dates=["entry_date"])
op = raw[raw.exit_reason == "end_of_data"]
print(f"open trades={len(op)}, entry-2026={(op.entry_date.dt.year==2026).sum()}, MTM sum={op.pnl_pct.sum():+.2f}u")
t26 = t[(t.year == 2026) & (t.holding_days > 20)]
print(f"2026 closed hold>20: n={len(t26)} sum={t26.pnl_pct.sum():+.2f}u")

# ---------- CLAIM 10: concentration ----------
print("\n===== C10 CONCENTRATION =====")
srt = t.sort_values("pnl_pct", ascending=False)
for k in [10, 100, 717]:
    print(f"top {k} trades: {srt.pnl_pct.head(k).sum():.1f}u = {srt.pnl_pct.head(k).sum()/tot:.1%}")
ps = t.groupby("symbol").pnl_pct.agg(["sum", "size"]).sort_values("sum", ascending=False)
print(f"symbols={len(ps)}; top10 share={ps['sum'].head(10).sum()/tot:.1%}; top20={ps['sum'].head(20).sum()/tot:.1%}")
neg = ps[ps["sum"] < 0]
print(f"negative symbols: {len(neg)} -> {neg.round(3).to_dict('index')}")
print(f"best symbol: {ps.index[0]} {ps.iloc[0]['sum']:+.2f}u / {int(ps.iloc[0]['size'])} trades")

# ---------- CLAIM 11: MAE split ----------
print("\n===== C11 MAE SPLIT =====")
print("mae>=-2%:", stats(t[t.mae_pct >= -0.02]))
print("mae<-8%: ", stats(t[t.mae_pct < -0.08]))

# ---------- CLAIM 14: exit reasons ----------
print("\n===== C14 EXIT REASONS =====")
print(t.exit_reason.value_counts().to_dict())
