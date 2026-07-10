import pandas as pd, numpy as np, json

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

D = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/"
t = pd.read_csv(D + "trades_metrics.csv", parse_dates=["entry_date", "exit_date", "signal_date"])
print("closed trades:", len(t))

# join signal-day scores
sig = pd.read_csv(D + "signals.csv", parse_dates=["date"],
                  usecols=["symbol", "date", "score", "score2", "score3", "score4", "score5", "exit_score", "entry_csr"])
t = t.merge(sig.rename(columns={"date": "signal_date"}), on=["symbol", "signal_date"], how="left")
print("score join miss:", t["score"].isna().sum())

t["year"] = t["entry_date"].dt.year
t["gross_pnl"] = np.nan  # placeholder
order = ["big_win", "win", "small", "loss", "big_loss"]

# ---------- 1. BUCKET PROFILE ----------
def prof(g):
    return pd.Series({
        "n": len(g),
        "sum_pnl": g.pnl_pct.sum(),
        "mean_pnl": g.pnl_pct.mean(),
        "wr": (g.pnl_pct > 0).mean(),
        "hold_med": g.holding_days.median(),
        "hold_mean": g.holding_days.mean(),
        "wait_med": g.wait_bars.median(),
        "red_touch": g.touch_bar_red.mean(),
        "hot_run": g.hot_run.mean(),
        "pre5_med": g.pre_ret5.median(),
        "pre20_med": g.pre_ret20.median(),
        "pre60_med": g.pre_ret60.median(),
        "dist_sma20_med": g.entry_dist_sma20.median(),
        "below_ma20": (g.entry_dist_sma20 < 0).mean(),
        "drop_pk20_med": g.entry_drop_from_peak20.median(),
        "mfe_med": g.mfe_pct.median(),
        "mae_med": g.mae_pct.median(),
        "b2peak_med": g.bars_to_peak.median(),
        "score_med": g.score.median(),
        "score3_med": g.score3.median(),
        "csr_med": g.entry_csr.median(),
    })

bp = t.groupby("bucket").apply(prof).reindex(order)
print("\n===== BUCKET PROFILE =====")
print(bp.round(4))

print("\nexit_reason x bucket:")
print(pd.crosstab(t.bucket, t.exit_reason).reindex(order))

# ---------- 2. EXPECTANCY PER SLICE (entry-time features) ----------
feats = ["pre_ret5", "pre_ret20", "pre_ret60", "entry_dist_sma20", "entry_drop_from_peak20",
         "wait_bars", "score", "score3", "score4", "entry_csr", "signal_to_fill_drop", "holding_days"]
print("\n===== QUINTILE EXPECTANCY (mean pnl / WR / sum pnl / n / bigwin%) =====")
for f in feats:
    try:
        q = pd.qcut(t[f], 5, duplicates="drop")
    except Exception:
        continue
    g = t.groupby(q, observed=True).agg(
        n=("pnl_pct", "size"), mean=("pnl_pct", "mean"), wr=("pnl_pct", lambda x: (x > 0).mean()),
        sum=("pnl_pct", "sum"), bigwin=("bucket", lambda x: (x == "big_win").mean()))
    print(f"\n-- {f} --")
    print(g.round(4))

print("\n-- touch_bar_red --")
print(t.groupby("touch_bar_red").agg(n=("pnl_pct", "size"), mean=("pnl_pct", "mean"),
      wr=("pnl_pct", lambda x: (x > 0).mean()), sum=("pnl_pct", "sum"),
      bigwin=("bucket", lambda x: (x == "big_win").mean())).round(4))
print("\n-- hot_run --")
print(t.groupby("hot_run").agg(n=("pnl_pct", "size"), mean=("pnl_pct", "mean"),
      wr=("pnl_pct", lambda x: (x > 0).mean()), sum=("pnl_pct", "sum"),
      bigwin=("bucket", lambda x: (x == "big_win").mean())).round(4))

# where do big_wins live: cross pre_ret20 x score3
t["pre20_q"] = pd.qcut(t.pre_ret20, 4, labels=["q1_low", "q2", "q3", "q4_high"])
t["score3_q"] = pd.qcut(t.score3, 4, labels=["q1_low", "q2", "q3", "q4_high"])
print("\n===== sum_pnl: pre20_q x score3_q =====")
print(t.pivot_table(index="pre20_q", columns="score3_q", values="pnl_pct", aggfunc="sum", observed=True).round(1))
print("\nmean pnl same grid:")
print(t.pivot_table(index="pre20_q", columns="score3_q", values="pnl_pct", aggfunc="mean", observed=True).round(4))
print("\nn same grid:")
print(t.pivot_table(index="pre20_q", columns="score3_q", values="pnl_pct", aggfunc="size", observed=True))

# ---------- 3. CHURN ----------
churn = t[t.pnl_pct.abs() < 0.03]
print("\n===== CHURN (|pnl|<3%) =====")
print(f"n={len(churn)} ({len(churn)/len(t):.1%}), sum_pnl={churn.pnl_pct.sum():.2f}u, "
      f"mean={churn.pnl_pct.mean():.4f}, hold_med={churn.holding_days.median()}")
# fee drag: pnl is net; fees = 0.4% of gross + slippage ~0.3% => explicit fee drag per trade
print(f"fee drag 0.4% x {len(churn)} churn trades = {0.004*len(churn):.1f}u; full friction 0.7% = {0.007*len(churn):.1f}u")
print(f"fee drag ALL {len(t)} trades = {0.004*len(t):.1f}u fees, {0.007*len(t):.1f}u incl slippage")
# quick-exit churn (holding_days<=8)
qe = t[t.holding_days <= 8]
print(f"\nquick exits hold<=8: n={len(qe)} ({len(qe)/len(t):.1%}) sum={qe.pnl_pct.sum():.2f}u mean={qe.pnl_pct.mean():.4f} "
      f"churn_share={(qe.pnl_pct.abs()<0.03).mean():.1%}")

# candidate entry-time filters: churn removed vs profit shed
print("\n===== FILTER CANDIDATES: churn removed vs pnl shed (selection wall check) =====")
cands = {
    "pre_ret20 < q20": t.pre_ret20 < t.pre_ret20.quantile(0.2),
    "pre_ret60 < q20": t.pre_ret60 < t.pre_ret60.quantile(0.2),
    "score3 < q20": t.score3 < t.score3.quantile(0.2),
    "score < q20": t.score < t.score.quantile(0.2),
    "entry_csr < 0.3": t.entry_csr < 0.3,
    "fill below MA20 (dist<0)": t.entry_dist_sma20 < 0,
    "fill below MA20 -4%": t.entry_dist_sma20 < -0.04,
    "wait_bars > 20": t.wait_bars > 20,
    "wait_bars > 30": t.wait_bars > 30,
    "touch_bar_red": t.touch_bar_red == True,
    "hot_run": t.hot_run == True,
    "score3<q20 & below MA20": (t.score3 < t.score3.quantile(0.2)) & (t.entry_dist_sma20 < 0),
    "wait>20 & below MA20": (t.wait_bars > 20) & (t.entry_dist_sma20 < 0),
    "pre20<q20 & score3<q50": (t.pre_ret20 < t.pre_ret20.quantile(0.2)) & (t.score3 < t.score3.median()),
}
rows = []
for name, m in cands.items():
    sub = t[m]
    rows.append({"filter": name, "n_cut": len(sub), "pnl_shed": sub.pnl_pct.sum(),
                 "churn_cut": (sub.pnl_pct.abs() < 0.03).sum(),
                 "bigwin_cut": (sub.bucket == "big_win").sum(),
                 "bigloss_cut": (sub.bucket == "big_loss").sum(),
                 "mean_pnl_cut": sub.pnl_pct.mean(),
                 "fee_saved": 0.007 * len(sub)})
fr = pd.DataFrame(rows)
fr["net_effect"] = -fr.pnl_shed  # pnl is already net of fees; removing trades changes total by -pnl_shed
print(fr.round(3).to_string(index=False))

# ---------- 4. YEARLY STABILITY ----------
print("\n===== YEARLY (by entry year) =====")
def yr(g):
    gp = g.pnl_pct[g.pnl_pct > 0].sum()
    gl = -g.pnl_pct[g.pnl_pct < 0].sum()
    return pd.Series({"n": len(g), "sum_pnl": g.pnl_pct.sum(), "wr": (g.pnl_pct > 0).mean(),
                      "pf": gp / gl if gl > 0 else np.inf, "mean": g.pnl_pct.mean(),
                      "bigwin_n": (g.bucket == "big_win").sum(),
                      "bigwin_pnl": g.pnl_pct[g.bucket == "big_win"].sum(),
                      "nonbig_sum": g.pnl_pct[g.bucket != "big_win"].sum()})
print(t.groupby("year").apply(yr).round(3))

# ---------- 5. CONCENTRATION ----------
print("\n===== CONCENTRATION =====")
srt = t.sort_values("pnl_pct", ascending=False)
tot = t.pnl_pct.sum()
for k in [10, 20, 50, 100, 200, 717]:
    print(f"top {k} trades: {srt.pnl_pct.head(k).sum():.1f}u = {srt.pnl_pct.head(k).sum()/tot:.1%} of total {tot:.1f}u")
print("\ntop-10 trades:")
print(srt.head(10)[["symbol", "entry_date", "exit_date", "pnl_pct", "holding_days", "pre_ret20", "score3", "year"]].round(4).to_string(index=False))

ps = t.groupby("symbol").pnl_pct.agg(["sum", "size", "mean"]).sort_values("sum", ascending=False)
print(f"\nsymbols: {len(ps)}; top10 syms = {ps['sum'].head(10).sum():.1f}u ({ps['sum'].head(10).sum()/tot:.1%}); "
      f"top20 = {ps['sum'].head(20).sum()/tot:.1%}; negative syms: {(ps['sum']<0).sum()} sum {ps.loc[ps['sum']<0,'sum'].sum():.1f}u")
print("\ntop 12 symbols:")
print(ps.head(12).round(3))
print("\nbottom 8 symbols:")
print(ps.tail(8).round(3))

t.to_parquet(D + "tmp_goodtrades_joined.parquet")
