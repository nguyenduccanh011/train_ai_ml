# -*- coding: utf-8 -*-
"""Bad-buy forensic: knife fills & chases on THIS bundle (3,781 closed trades).
Read-only. Prints stats; no files written except stdout capture."""
import os, sys, json
import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
SERVING = r"C:\Users\DUC CANH PC\Desktop\stock-serving"
SLIP = 0.0015
os.chdir(SERVING); sys.path.insert(0, SERVING)

tm = pd.read_csv(os.path.join(OUT, "trades_metrics.csv"))
assert len(tm) == 3781, len(tm)
tm["win"] = tm["pnl_pct"] > 0
tm["big_loss"] = tm["pnl_pct"] <= -0.15

def grp_stats(df, name):
    return {
        "name": name, "n": int(len(df)),
        "wr": round(float(df["win"].mean()), 4) if len(df) else None,
        "avg_pnl": round(float(df["pnl_pct"].mean()), 4) if len(df) else None,
        "med_pnl": round(float(df["pnl_pct"].median()), 4) if len(df) else None,
        "sum_pnl_u": round(float(df["pnl_pct"].sum()), 2),
        "bigloss_rate": round(float(df["big_loss"].mean()), 4) if len(df) else None,
        "n_bigloss": int(df["big_loss"].sum()),
        "avg_mae": round(float(df["mae_pct"].mean()), 4) if len(df) else None,
    }

print("=" * 20, "1. RED vs GREEN touch bar", "=" * 20)
for flag, lbl in [(True, "red_touch"), (False, "green_touch")]:
    print(json.dumps(grp_stats(tm[tm["touch_bar_red"] == flag], lbl)))

print("=" * 20, "2. wait_bars buckets", "=" * 20)
wb = pd.cut(tm["wait_bars"], bins=[0, 1, 2, 5, 10, 20, 40],
            labels=["1", "2", "3-5", "6-10", "11-20", "21-40"])
for lbl, g in tm.groupby(wb, observed=True):
    print(json.dumps(grp_stats(g, f"wait={lbl}")))
print("wait_bars min/max:", int(tm.wait_bars.min()), int(tm.wait_bars.max()))

print("=" * 20, "3. hot_run x fast fill (wait<=2)", "=" * 20)
for hot in [True, False]:
    for fast in [True, False]:
        g = tm[(tm["hot_run"] == hot) & ((tm["wait_bars"] <= 2) == fast)]
        print(json.dumps(grp_stats(g, f"hot={hot} fast={fast}")))
# knife archetype: fast + red
g = tm[(tm["wait_bars"] <= 2) & tm["touch_bar_red"]]
print(json.dumps(grp_stats(g, "fast_red (wait<=2 & red)")))
g = tm[(tm["wait_bars"] >= 6) & tm["touch_bar_red"]]
print(json.dumps(grp_stats(g, "slow_red (wait>=6 & red)")))

print("=" * 20, "4. deciles pre_ret20 / entry_drop_from_peak20", "=" * 20)
for col in ["pre_ret20", "entry_drop_from_peak20"]:
    d = tm.dropna(subset=[col]).copy()
    d["dec"] = pd.qcut(d[col], 10, labels=False, duplicates="drop")
    for dec, g in d.groupby("dec"):
        lo, hi = g[col].min(), g[col].max()
        s = grp_stats(g, f"{col} D{dec} [{lo:.3f},{hi:.3f}]")
        print(json.dumps(s))

print("=" * 20, "5. MAE underwater thresholds", "=" * 20)
for thr in [-0.08, -0.12, -0.20]:
    g = tm[tm["mae_pct"] <= thr]
    rec = g[g["pnl_pct"] > 0]
    print(json.dumps({"mae<=": thr, "n": int(len(g)),
                      "share": round(len(g) / len(tm), 4),
                      "sum_pnl_u": round(float(g["pnl_pct"].sum()), 2),
                      "recovered_to_win_n": int(len(rec)),
                      "recovered_share": round(len(rec) / max(len(g), 1), 4),
                      "avg_pnl": round(float(g["pnl_pct"].mean()), 4)}))

print("=" * 20, "6. cheaper-earlier (needs OHLCV)", "=" * 20)
from serving.ohlcv_store import OhlcvStore  # noqa: E402
ohlcv = OhlcvStore("data/ohlcv.db").load()
ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date.astype(str)
sym_arrays = {}
for sym, g in ohlcv.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    sym_arrays[sym] = {"idx": {d: k for k, d in enumerate(g["date"])},
                       "c": g["close"].to_numpy(float),
                       "l": g["low"].to_numpy(float)}
res = []
for _, t in tm.iterrows():
    A = sym_arrays.get(t["symbol"])
    if A is None:
        res.append((np.nan, np.nan, np.nan)); continue
    eidx = A["idx"].get(str(t["entry_date"]))
    sidx = A["idx"].get(str(t["signal_date"]))
    limit_raw = float(t["entry_price"]) / (1 + SLIP)
    c, l = A["c"], A["l"]
    a = c[eidx - 5] < limit_raw if (eidx is not None and eidx >= 5) else np.nan
    b = c[sidx - 5] < limit_raw if (sidx is not None and sidx >= 5) else np.nan
    d = (l[max(0, sidx - 10):sidx].min() < limit_raw) if (sidx is not None and sidx >= 1) else np.nan
    res.append((a, b, d))
r = pd.DataFrame(res, columns=["c5_before_fill_cheaper", "c5_before_signal_cheaper",
                               "low10_before_signal_cheaper"])
tm = pd.concat([tm.reset_index(drop=True), r], axis=1)
for col in r.columns:
    v = tm[col].dropna()
    yes = tm[tm[col] == True]  # noqa: E712
    print(json.dumps({"metric": col, "n_valid": int(len(v)),
                      "share_true": round(float(v.mean()), 4),
                      "avg_pnl_if_true": round(float(yes["pnl_pct"].mean()), 4),
                      "avg_pnl_if_false": round(float(tm[tm[col] == False]["pnl_pct"].mean()), 4)}))

print("=" * 20, "7. conviction (eff_depth) merge", "=" * 20)
dep = pd.read_parquet(os.path.join(OUT, "depths.parquet"))
dep["date"] = dep["date"].astype(str)
tm = tm.merge(dep.rename(columns={"date": "signal_date"}), on=["symbol", "signal_date"], how="left")
print("eff_depth merged non-null:", int(tm["eff_depth"].notna().sum()))
print(tm["eff_depth"].describe().round(4).to_string())
db = pd.cut(tm["eff_depth"], bins=[0.020, 0.030, 0.040, 0.0449, 0.0451],
            labels=["shallow<=0.030", "0.030-0.040", "0.040-0.0449", "full=0.045"])
for lbl, g in tm.groupby(db, observed=True):
    print(json.dumps(grp_stats(g, f"depth={lbl}")))

print("=" * 20, "8. FILTER TESTS (static cohort removal)", "=" * 20)
total = float(tm["pnl_pct"].sum())
def filt(mask, name):
    rm = tm[mask]
    print(json.dumps({"filter": name, "n_removed": int(len(rm)),
                      "pct_trades": round(len(rm) / len(tm), 4),
                      "pnl_removed_u": round(float(rm["pnl_pct"].sum()), 2),
                      "net_delta_u": round(-float(rm["pnl_pct"].sum()), 2),
                      "wr_removed": round(float(rm["win"].mean()), 4) if len(rm) else None,
                      "bigloss_removed": int(rm["big_loss"].sum()),
                      "bigloss_units_removed": round(float(rm.loc[rm["big_loss"], "pnl_pct"].sum()), 2)}))
filt(tm["touch_bar_red"], "confirm_reversal=green-only (drop red-touch)")
for x in [0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    filt(tm["pre_ret20"] >= x, f"skip pre_ret20>={x}")
for thr in [0.0449, 0.040, 0.035]:
    filt(tm["eff_depth"] >= thr, f"conviction floor: drop eff_depth>={thr}")
filt((tm["wait_bars"] <= 2) & tm["touch_bar_red"] & (tm["pre_ret20"] >= 0.12),
     "combo: hot_run & fast(<=2) & red touch")
filt((tm["wait_bars"] >= 21), "cap window at 20 bars (drop wait>=21)")
filt((tm["wait_bars"] >= 11), "cap window at 10 bars (drop wait>=11)")

print("=" * 20, "9. big_loss cohort profile vs all", "=" * 20)
bl = tm[tm["big_loss"]]
for name, g in [("big_loss(59)", bl), ("all", tm)]:
    print(json.dumps({"cohort": name, "n": int(len(g)),
                      "mean_wait": round(float(g["wait_bars"].mean()), 1),
                      "pct_red": round(float(g["touch_bar_red"].mean()), 3),
                      "pct_hot": round(float(g["hot_run"].mean()), 3),
                      "mean_pre20": round(float(g["pre_ret20"].mean()), 4),
                      "mean_drop_pk20": round(float(g["entry_drop_from_peak20"].mean()), 4),
                      "mean_eff_depth": round(float(g["eff_depth"].mean()), 4),
                      "pct_wait<=2": round(float((g["wait_bars"] <= 2).mean()), 3),
                      "pct_wait>=11": round(float((g["wait_bars"] >= 11).mean()), 3)}))

print("=" * 20, "10. examples", "=" * 20)
cols = ["symbol", "signal_date", "entry_date", "exit_date", "wait_bars", "pnl_pct",
        "pre_ret20", "touch_bar_red", "mae_pct", "mfe_pct", "eff_depth", "hot_run"]
print("-- worst knives (fast red fills, pnl<=-0.15):")
k = tm[(tm["big_loss"]) & (tm["wait_bars"] <= 2) & tm["touch_bar_red"]].nsmallest(6, "pnl_pct")
print(k[cols].to_string(index=False))
print("-- stale-limit knives (wait>=11, pnl<=-0.15):")
k = tm[(tm["big_loss"]) & (tm["wait_bars"] >= 11)].nsmallest(6, "pnl_pct")
print(k[cols].to_string(index=False))
print("-- hot_run chase losers (pre_ret20>=0.25, pnl<=-0.10):")
k = tm[(tm["pre_ret20"] >= 0.25) & (tm["pnl_pct"] <= -0.10)].nsmallest(6, "pnl_pct")
print(k[cols].to_string(index=False))
print("-- red-touch BIG WINNERS (selection-wall counterexamples):")
k = tm[tm["touch_bar_red"]].nlargest(5, "pnl_pct")
print(k[cols].to_string(index=False))
print("-- never-worked entries (mfe<3%, pnl<=-0.12):")
k = tm[(tm["mfe_pct"] < 0.03) & (tm["pnl_pct"] <= -0.12)].nsmallest(6, "pnl_pct")
print(k[cols].to_string(index=False))
print("DONE")
