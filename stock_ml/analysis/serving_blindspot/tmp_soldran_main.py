"""Sold-then-ran (ban ho) forensic on serving champion trades. Read-only."""
import pandas as pd, numpy as np, json

D = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/"
tm = pd.read_csv(D + "trades_metrics.csv", parse_dates=["entry_date", "exit_date", "signal_date"])
da = pd.read_csv(D + "daily_activity.csv")
print("closed trades:", len(tm))

POSTS = ["post_ret5", "post_ret10", "post_ret21", "post_ret42", "post_max_runup21"]

def dist(g):
    out = {}
    for c in POSTS:
        s = g[c].dropna()
        out[c] = dict(n=len(s), mean=s.mean(), median=s.median(),
                      p25=s.quantile(.25), p75=s.quantile(.75), p90=s.quantile(.9),
                      pct_pos=(s > 0).mean())
    s = g["post_max_runup21"].dropna()
    out["pct_runup_ge5"] = (s >= .05).mean()
    out["pct_runup_ge10"] = (s >= .10).mean()
    out["pct_runup_ge15"] = (s >= .15).mean()
    return out

print("\n=== A. distributions by exit_reason ===")
for reason, g in tm.groupby("exit_reason"):
    print(f"\n--- {reason} (n={len(g)}) ---")
    print(json.dumps(dist(g), indent=1, default=float))

print("\n=== overall ===")
print(json.dumps(dist(tm), indent=1, default=float))

print("\n=== B. left-on-table aggregates (units, gross) ===")
for reason, g in tm.groupby("exit_reason"):
    print(reason, "n=", len(g),
          "sum_post_ret10=%.2f" % g["post_ret10"].sum(),
          "sum_post_ret21=%.2f" % g["post_ret21"].sum(),
          "sum_post_ret42=%.2f" % g["post_ret42"].sum(),
          "sum_runup21_clip0=%.2f" % g["post_max_runup21"].clip(lower=0).sum(),
          "nan10=%d nan21=%d nan42=%d" % (g["post_ret10"].isna().sum(), g["post_ret21"].isna().sum(), g["post_ret42"].isna().sum()))
print("TOTAL sum_post_ret10=%.2f sum_post_ret21=%.2f sum_post_ret42=%.2f oracle_runup21=%.2f realized_pnl_sum=%.2f" % (
    tm["post_ret10"].sum(), tm["post_ret21"].sum(), tm["post_ret42"].sum(),
    tm["post_max_runup21"].clip(lower=0).sum(), tm["pnl_pct"].sum()))

print("\n=== C. sold_too_early rates ===")
print("overall ge5:", tm["sold_too_early"].mean(), tm["sold_too_early"].sum())
ru = tm["post_max_runup21"]
print("overall ge10:", (ru >= .10).mean(), int((ru >= .10).sum()))
print("by exit_reason ge5:", tm.groupby("exit_reason")["sold_too_early"].agg(["mean", "sum", "count"]).to_dict("index"))
tm["year"] = tm["exit_date"].dt.year
print("by year ge5:", tm.groupby("year")["sold_too_early"].mean().round(3).to_dict())
print("by bucket ge5:", tm.groupby("bucket")["sold_too_early"].agg(["mean", "count"]).round(3).to_dict("index"))
win = tm["pnl_pct"] > 0
print("winners (pnl>0) n=%d ge5=%.3f ge10=%.3f mean_runup=%.4f" % (
    win.sum(), tm.loc[win, "sold_too_early"].mean(), (ru[win] >= .1).mean(), ru[win].mean()))
print("losers (pnl<=0) n=%d ge5=%.3f ge10=%.3f mean_runup=%.4f" % (
    (~win).sum(), tm.loc[~win, "sold_too_early"].mean(), (ru[~win] >= .1).mean(), ru[~win].mean()))

print("\n=== D. traits sold_too_early True vs False ===")
TR = ["pnl_pct", "pre_ret5", "pre_ret20", "pre_ret60", "holding_days", "mfe_pct", "mae_pct",
      "giveback_pct", "bars_to_peak", "wait_bars", "entry_dist_sma20", "signal_to_fill_drop"]
agg = tm.groupby("sold_too_early")[TR].agg(["mean", "median"]).round(4)
print(agg.T.to_string())
for b in ["hot_run", "touch_bar_red"]:
    print(b, tm.groupby("sold_too_early")[b].mean().round(3).to_dict())
tm["peak_at_exit"] = tm["bars_to_peak"] >= tm["holding_days"] - 1
print("peak_at_exit(<=1bar before exit):", tm.groupby("sold_too_early")["peak_at_exit"].mean().round(3).to_dict())
tm["young"] = tm["holding_days"] <= 8
print("young(hold<=8):", tm.groupby("sold_too_early")["young"].mean().round(3).to_dict())
print("sold_too_early rate by hold bucket:",
      tm.groupby(pd.cut(tm["holding_days"], [0, 4, 8, 16, 32, 64, 10000]))["sold_too_early"].agg(["mean", "count"]).round(3).to_string())

print("\n=== D2. inside sold_too_early cohort: cut-on-dip vs sold-near-peak ===")
ste = tm[tm["sold_too_early"]]
cut_dip = ste[ste["giveback_pct"] <= -0.05]
near_peak = ste[ste["giveback_pct"] > -0.02]
mid = ste[(ste["giveback_pct"] > -0.05) & (ste["giveback_pct"] <= -0.02)]
for name, g in [("cut_on_dip(giveback<=-5%)", cut_dip), ("near_peak(giveback>-2%)", near_peak), ("mid", mid)]:
    print(name, "n=%d pnl_mean=%.4f runup21_mean=%.4f post_ret10_mean=%.4f mfe_mean=%.4f hold_med=%.0f" % (
        len(g), g["pnl_pct"].mean(), g["post_max_runup21"].mean(), g["post_ret10"].mean(), g["mfe_pct"].mean(), g["holding_days"].median()))

print("\n=== E. hypothetical +10-bar hold delta ===")
for key, g in [("all", tm), ("winners", tm[win]), ("losers", tm[~win])]:
    p10 = g["post_ret10"].dropna()
    print(key, "n=%d mean=%.4f sum=%.2f pct_pos=%.3f" % (len(p10), p10.mean(), p10.sum(), (p10 > 0).mean()))
print("by exit_reason:")
print(tm.groupby("exit_reason")["post_ret10"].agg(["count", "mean", "sum"]).round(4).to_string())
print("by bucket:")
print(tm.groupby("bucket")["post_ret10"].agg(["count", "mean", "sum"]).round(4).to_string())
print("by year:")
print(tm.groupby("year")["post_ret10"].agg(["count", "mean", "sum"]).round(4).to_string())

print("\n=== F. rebuy-higher (sell then must rebuy) ===")
cal = pd.to_datetime(da["date"] if "date" in da.columns else da.iloc[:, 0]).sort_values().reset_index(drop=True)
cal_idx = {d: i for i, d in enumerate(cal)}
tm2 = tm.sort_values(["symbol", "entry_date"]).reset_index(drop=True)
rows = []
for sym, g in tm2.groupby("symbol"):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        ex, nx = g.loc[i], g.loc[i + 1]
        bi, bj = cal_idx.get(ex["exit_date"]), cal_idx.get(nx["entry_date"])
        if bi is None or bj is None:
            continue
        rows.append(dict(symbol=sym, exit_date=ex["exit_date"], pnl_pct=ex["pnl_pct"],
                         sold_too_early=ex["sold_too_early"],
                         gap_bars=bj - bi, premium=nx["entry_price"] / ex["exit_price"] - 1.0,
                         next_entry=nx["entry_date"]))
rb = pd.DataFrame(rows)
w40 = rb[rb["gap_bars"] <= 40]
print("exits with a next trade same symbol:", len(rb), "; rebuy within 40 bars:", len(w40),
      "(%.1f%% of %d closed exits)" % (100 * len(w40) / len(tm), len(tm)))
print("rebuy<=40b: pct_higher=%.3f mean_premium=%.4f median=%.4f" % (
    (w40["premium"] > 0).mean(), w40["premium"].mean(), w40["premium"].median()))
w40l = w40[w40["pnl_pct"] <= 0]
print("loss-exits rebuy<=40b: n=%d pct_higher=%.3f mean_premium=%.4f" % (
    len(w40l), (w40l["premium"] > 0).mean(), w40l["premium"].mean()))
w40s = w40[w40["sold_too_early"]]
print("sold_too_early exits rebuy<=40b: n=%d pct_higher=%.3f mean_premium=%.4f" % (
    len(w40s), (w40s["premium"] > 0).mean(), w40s["premium"].mean()))
w15 = rb[rb["gap_bars"] <= 15]
print("rebuy<=15b: n=%d pct_higher=%.3f mean_premium=%.4f" % (len(w15), (w15["premium"] > 0).mean(), w15["premium"].mean()))

print("\n=== G. t1831 residual: mfe in [10,27%) band ===")
band = tm[(tm["mfe_pct"] >= .10) & (tm["mfe_pct"] < .27)]
print("n=%d sum_pnl=%.2f mean_pnl=%.4f mean_giveback=%.4f sold_too_early=%.3f sum_giveback_u=%.2f" % (
    len(band), band["pnl_pct"].sum(), band["pnl_pct"].mean(), band["giveback_pct"].mean(),
    band["sold_too_early"].mean(), band["giveback_pct"].sum()))
gb_all = tm[tm["mfe_pct"] >= .10]
print("all mfe>=10%%: n=%d sum_giveback=%.2f" % (len(gb_all), gb_all["giveback_pct"].sum()))

print("\n=== H. examples: top sold-then-ran ===")
cols = ["symbol", "entry_date", "exit_date", "pnl_pct", "holding_days", "mfe_pct", "giveback_pct",
        "post_ret10", "post_ret21", "post_max_runup21", "exit_reason", "pre_ret20"]
top = tm.sort_values("post_max_runup21", ascending=False).head(15)[cols]
print(top.to_string())
print("\n-- big winners sold then ran further (pnl>=15% & runup>=15%) --")
bw = tm[(tm["pnl_pct"] >= .15) & (ru >= .15)].sort_values("post_max_runup21", ascending=False).head(8)[cols]
print(bw.to_string())
print("\n-- loss-exit then big bounce (pnl<=-5% & runup>=15%) --")
lb = tm[(tm["pnl_pct"] <= -.05) & (ru >= .15)].sort_values("post_max_runup21", ascending=False).head(8)[cols]
print(lb.to_string())
