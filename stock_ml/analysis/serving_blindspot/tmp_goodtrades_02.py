import pandas as pd, numpy as np

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
D = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/"
t = pd.read_parquet(D + "tmp_goodtrades_joined.parquet")
t["year"] = pd.to_datetime(t.entry_date).dt.year

# ---------- CORE EDGE SLICE ----------
print("===== CORE EDGE SLICES =====")
slices = {
    "wait<=2": t.wait_bars <= 2,
    "wait<=4": t.wait_bars <= 4,
    "dist_sma20>=0 (fill at/above MA20)": t.entry_dist_sma20 >= 0,
    "wait<=4 & dist>=0": (t.wait_bars <= 4) & (t.entry_dist_sma20 >= 0),
    "wait<=4 & dist>=0 & pre20>0": (t.wait_bars <= 4) & (t.entry_dist_sma20 >= 0) & (t.pre_ret20 > 0),
    "hot_run & dist>=0": t.hot_run & (t.entry_dist_sma20 >= 0),
    "green touch (not red)": ~t.touch_bar_red,
    "green & dist>=0": (~t.touch_bar_red) & (t.entry_dist_sma20 >= 0),
}
tot = t.pnl_pct.sum()
for name, m in slices.items():
    s = t[m]
    print(f"{name:38s} n={len(s):5d} ({len(s)/len(t):5.1%}) sum={s.pnl_pct.sum():7.1f}u ({s.pnl_pct.sum()/tot:5.1%} of total) "
          f"mean={s.pnl_pct.mean():+.4f} wr={(s.pnl_pct>0).mean():.3f} bigwin%={(s.bucket=='big_win').mean():.3f}")

# yearly stability of the core slice vs the rest
core = (t.wait_bars <= 4) & (t.entry_dist_sma20 >= 0)
print("\ncore slice (wait<=4 & dist>=0) by year vs rest:")
for y, g in t.groupby("year"):
    c, r = g[core.loc[g.index]], g[~core.loc[g.index]]
    print(f"{y}: core n={len(c):4d} sum={c.pnl_pct.sum():6.1f} mean={c.pnl_pct.mean():+.4f} | rest n={len(r):4d} sum={r.pnl_pct.sum():6.1f} mean={r.pnl_pct.mean():+.4f}")

# ---------- early-path trait (pyramid-relevant): shallow MAE ----------
print("\n===== POST-ENTRY PATH SPLIT (not entry-filterable; pyramid/exit relevant) =====")
for name, m in {"mae>=-2% (never dipped)": t.mae_pct >= -0.02,
                "mae>=-2% & bars_to_peak>=5": (t.mae_pct >= -0.02),
                "mae<-8%": t.mae_pct < -0.08}.items():
    s = t[m]
    print(f"{name:28s} n={len(s):5d} ({len(s)/len(t):5.1%}) sum={s.pnl_pct.sum():7.1f}u mean={s.pnl_pct.mean():+.4f} wr={(s.pnl_pct>0).mean():.3f} bigwin%={(s.bucket=='big_win').mean():.3f}")

hold_split = pd.cut(t.holding_days, [0, 8, 20, 10000], labels=["<=8", "9-20", ">20"])
print("\nhold split:")
print(t.groupby(hold_split, observed=True).agg(n=("pnl_pct","size"), sum=("pnl_pct","sum"),
    mean=("pnl_pct","mean"), wr=("pnl_pct", lambda x:(x>0).mean())).round(4))

# ---------- churn efficiency ranking ----------
print("\n===== CHURN FILTER EFFICIENCY (churn removed per 1u pnl shed) =====")
cands = {
    "wait_bars > 20 (window cap 20)": t.wait_bars > 20,
    "wait_bars > 30": t.wait_bars > 30,
    "wait_bars > 15": t.wait_bars > 15,
    "wait_bars > 10": t.wait_bars > 10,
    "fill below MA20 -4%": t.entry_dist_sma20 < -0.04,
    "entry_csr < 0.3": t.entry_csr < 0.3,
    "score < q20": t.score < t.score.quantile(0.2),
    "pre_ret20 in (-3.3%..2%) (dead-zone q2)": (t.pre_ret20 > -0.0329) & (t.pre_ret20 <= 0.02),
    "wait>15 & dist<-0.02": (t.wait_bars > 15) & (t.entry_dist_sma20 < -0.02),
}
rows = []
for name, m in cands.items():
    s = t[m]
    churn_cut = (s.pnl_pct.abs() < 0.03).sum()
    shed = s.pnl_pct.sum()
    rows.append({"filter": name, "n_cut": len(s), "churn_cut": churn_cut,
                 "pnl_shed": round(shed, 2), "churn_per_u": round(churn_cut / shed, 1) if shed > 0 else np.inf,
                 "bigwin_cut": (s.bucket == "big_win").sum(),
                 "mean_cut": round(s.pnl_pct.mean(), 4)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------- yearly by EXIT year + open-trade caveat ----------
print("\n===== YEARLY by exit year =====")
t["exit_year"] = pd.to_datetime(t.exit_date).dt.year
def yr(g):
    gp = g.pnl_pct[g.pnl_pct > 0].sum(); gl = -g.pnl_pct[g.pnl_pct < 0].sum()
    return pd.Series({"n": len(g), "sum": g.pnl_pct.sum(), "wr": (g.pnl_pct > 0).mean(),
                      "pf": gp/gl if gl else np.inf})
print(t.groupby("exit_year").apply(yr).round(3))

raw = pd.read_csv(D + "trades_raw.csv", parse_dates=["entry_date"])
op = raw[raw.exit_reason == "end_of_data"]
print(f"\nopen trades: {len(op)}; entered 2026: {(op.entry_date.dt.year==2026).sum()}, 2025: {(op.entry_date.dt.year==2025).sum()}")
if "pnl_pct" in op.columns:
    print("open trades mark-to-market pnl sum:", op.pnl_pct.sum().round(2), " mean:", op.pnl_pct.mean().round(4))

# 2026 detail: is it bear-regime or just immature trades?
t26 = t[t.year == 2026]
print(f"\n2026 closed: n={len(t26)} sum={t26.pnl_pct.sum():.2f} hold_med={t26.holding_days.median()} "
      f"| hold>20: n={(t26.holding_days>20).sum()} sum={t26.pnl_pct[t26.holding_days>20].sum():.2f}")

# ---------- concrete examples ----------
print("\n===== EXAMPLES =====")
core_bw = t[core & (t.bucket == "big_win")].sort_values("pnl_pct", ascending=False)
print("core-slice big wins (top 5):")
print(core_bw.head(5)[["symbol","entry_date","exit_date","pnl_pct","wait_bars","entry_dist_sma20","pre_ret20","holding_days","touch_bar_red"]].round(4).to_string(index=False))
ch = t[(t.pnl_pct.abs() < 0.01) & (t.holding_days <= 5)].sort_values("entry_date").tail(5)
print("\ntypical churn trades (|pnl|<1%, hold<=5, recent):")
print(ch[["symbol","entry_date","exit_date","pnl_pct","holding_days","wait_bars","entry_dist_sma20","exit_reason"]].round(4).to_string(index=False))
lw = t[(t.wait_bars > 20)].sort_values("pnl_pct").head(3)
print("\nworst long-wait fills (wait>20):")
print(lw[["symbol","entry_date","signal_date","wait_bars","pnl_pct","entry_dist_sma20"]].round(4).to_string(index=False))
