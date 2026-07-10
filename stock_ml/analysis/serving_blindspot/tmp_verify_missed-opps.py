# -*- coding: utf-8 -*-
"""Adversarial verification of missed-opps claims. Independent recompute from raw CSVs +
ohlcv.db + Postgres. Prints JSON to stdout. Does NOT reuse tmp_missedops_result.json."""
import json
import os
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
WINDOW = 40
COST_RT = 0.007

os.chdir(OUT)
events = pd.read_csv("events.csv")
uf_all = pd.read_csv("unfilled_signals.csv")
tm = pd.read_csv("trades_metrics.csv")
tr = pd.read_csv("trades_raw.csv")
da = pd.read_csv("daily_activity.csv")
sig = pd.read_csv("signals.csv", usecols=["symbol", "date", "score", "score3"])

con = sqlite3.connect(DB)
oh = pd.read_sql("select symbol, date, low, close from ohlcv order by symbol, date", con)
con.close()
A = {}
for s, g in oh.groupby("symbol"):
    g = g.reset_index(drop=True)
    A[s] = {"dates": g["date"].to_numpy(),
            "idx": {d: k for k, d in enumerate(g["date"])},
            "l": g["low"].to_numpy(float), "c": g["close"].to_numpy(float)}

R = {}

# ---------- Claim 1+2: fill rate ----------
ev = events[events["kind"].isin(["filled", "missed"])].copy()
n_f = int((ev["kind"] == "filled").sum())
n_m = int((ev["kind"] == "missed").sum())
R["c1_fill_rate"] = {"filled": n_f, "missed": n_m, "placed": n_f + n_m,
                     "rate": round(n_f / (n_f + n_m), 4),
                     "kinds_all": events["kind"].value_counts().to_dict()}
ev["year"] = ev["date"].str[:4]
byyr = {}
for y, g in ev.groupby("year"):
    f = int((g["kind"] == "filled").sum()); m = int((g["kind"] == "missed").sum())
    byyr[y] = {"filled": f, "missed": m, "rate": round(f / (f + m), 4)}
R["c2_by_year"] = byyr

# ---------- Claim 3+4: unfilled cohort ----------
uf = uf_all[uf_all["drop_reason"].isin(["unfilled", "unfilled_window_open"])].copy()
R["c3_n_unfilled"] = {"total": len(uf),
                      "by_reason": uf["drop_reason"].value_counts().to_dict()}
d = {}
for c in ["atmarket_ret21", "atmarket_ret42"]:
    v = uf[c].dropna()
    d[c] = {"n": int(len(v)), "mean": round(float(v.mean()), 4),
            "median": round(float(v.median()), 4),
            "pct_pos": round(float((v > 0).mean()), 4),
            "sum_gross": round(float(v.sum()), 1),
            "sum_net": round(float(v.sum() - COST_RT * len(v)), 1),
            "mean_net": round(float(v.mean() - COST_RT), 4)}
R["c3_dist"] = d
R["c3_realized"] = {"n_closed": len(tm), "mean_pnl": round(float(tm["pnl_pct"].mean()), 4),
                    "sum_pnl": round(float(tm["pnl_pct"].sum()), 1),
                    "median_hold": float(tm["holding_days"].median())}
R["c4_ratio_net21_vs_realized"] = round(
    (d["atmarket_ret21"]["sum_net"]) / float(tm["pnl_pct"].sum()), 2)

# spot-check atmarket_ret21 definition against ohlcv.db on a random sample
rng = np.random.RandomState(7)
samp = uf.dropna(subset=["atmarket_ret21"]).sample(200, random_state=7)
bad = 0
for r in samp.itertuples():
    a = A.get(r.symbol); i = a["idx"].get(r.signal_date) if a else None
    if i is None:
        bad += 1; continue
    n = len(a["c"]); nb = i + 1
    ref = a["c"][nb + 21] / a["c"][nb] - 1.0 if nb + 21 < n else np.nan
    if not np.isclose(ref, r.atmarket_ret21, atol=1e-9):
        bad += 1
R["c3_defn_spotcheck_mismatches_of_200"] = bad

# LOOKAHEAD diagnostic: at what bar is 'unfilled' knowable? entry is at i+1 but
# non-touch is only known at i+WINDOW. fraction of cohort where the low came within
# 1pp of the limit AFTER bar i+1 (i.e. selection uses future path info) is 100% by
# construction; instead quantify: mean ret21 of ALL placed (filled@limit as-if at-market
# next bar + missed) to show cohort-selection effect.
allret = []
for r in ev.itertuples():
    a = A[r.symbol]; i = int(r.bar_idx); n = len(a["c"]); nb = i + 1
    if nb + 21 < n:
        allret.append(a["c"][nb + 21] / a["c"][nb] - 1.0)
R["c3_lookahead_ctx"] = {
    "atmarket_ret21_ALL_placed_mean": round(float(np.mean(allret)), 4),
    "n": len(allret),
    "note": "cohort mean minus all-placed mean = selection effect of conditioning on future non-touch"}

# ---------- Claim 5+6: chase at window-end ----------
miss = events[events["kind"] == "missed"].copy()
gap = (miss["aux_idx"] - miss["bar_idx"]).value_counts().head(3).to_dict()
R["c5_aux_minus_bar_top"] = {str(k): int(v) for k, v in gap.items()}
rows = []
for r in miss.itertuples():
    a = A[r.symbol]; i, w = int(r.bar_idx), int(r.aux_idx); n = len(a["c"])
    if w >= n - 1 or w - i < WINDOW:
        continue
    ec = a["c"][w]
    prem = ec / a["c"][i] - 1.0
    r21 = a["c"][w + 21] / ec - 1.0 if w + 21 < n else np.nan
    r42 = a["c"][w + 42] / ec - 1.0 if w + 42 < n else np.nan
    rows.append((prem, r21, r42))
ch = pd.DataFrame(rows, columns=["premium", "r21", "r42"])

def coh(dd):
    o = {}
    for c in ["r21", "r42"]:
        v = dd[c].dropna()
        o[c] = {"n": int(len(v)), "mean": round(float(v.mean()), 4),
                "mean_net": round(float(v.mean() - COST_RT), 4),
                "median": round(float(v.median()), 4)}
    o["prem_mean"] = round(float(dd["premium"].mean()), 4)
    o["prem_median"] = round(float(dd["premium"].median()), 4)
    o["n_total"] = int(len(dd))
    return o

R["c5_chase_all"] = coh(ch)
R["c6_le2"] = coh(ch[ch["premium"] <= 0.02])
R["c6_le4"] = coh(ch[ch["premium"] <= 0.04])
R["c6_gt4"] = coh(ch[ch["premium"] > 0.04])

# ---------- Claim 7+8: depth / capture ----------
u = uf.dropna(subset=["max_low_in_window", "limit_price", "signal_close"]).copy()
u["reached"] = 1 - u["max_low_in_window"] / u["signal_close"]
u["ldepth"] = 1 - u["limit_price"] / u["signal_close"]
u["short"] = (u["max_low_in_window"] - u["limit_price"]) / u["signal_close"]
R["c7"] = {"n_denominator": len(u),
           "reached_median": round(float(u["reached"].median()), 4),
           "ldepth_median": round(float(u["ldepth"].median()), 4),
           "short_median": round(float(u["short"].median()), 4),
           "within_05pp": round(float((u["short"] <= 0.005).mean()), 4),
           "within_1pp": round(float((u["short"] <= 0.01).mean()), 4),
           "within_2pp": round(float((u["short"] <= 0.02).mean()), 4)}

def capture(mask, depth_arr, tag):
    cap = u[mask]
    r21l = []
    touch_after_22 = 0; touch_known = 0
    for r in cap.itertuples():
        a = A.get(r.symbol); i = a["idx"].get(r.signal_date) if a else None
        if i is None:
            continue
        n = len(a["c"])
        dep = depth_arr[r.Index] if hasattr(depth_arr, "__getitem__") else depth_arr
        entry = r.signal_close * (1 - dep)
        # first touch bar of the shallower limit
        w_end = min(i + WINDOW, n - 1)
        lows = a["l"][i + 1: w_end + 1]
        hit = np.nonzero(lows <= entry + 1e-12)[0]
        if len(hit):
            touch_known += 1
            if (i + 1 + hit[0]) > i + 22:
                touch_after_22 += 1
        if i + 22 < n:
            r21l.append(a["c"][i + 22] / entry - 1)
    R[tag] = {"n_captured": int(len(cap)),
              "rate_vs_7582": round(len(cap) / 7582, 4),
              "rate_vs_denom": round(len(cap) / len(u), 4),
              "ret21_from_limit_mean": round(float(np.mean(r21l)), 4) if r21l else None,
              "pct_touch_after_bar22": round(touch_after_22 / touch_known, 3) if touch_known else None}

capture(u["max_low_in_window"] <= u["signal_close"] * 0.97, 0.03, "c8_flat03")
capture(u["max_low_in_window"] <= u["signal_close"] * 0.98, 0.02, "c8_flat02")
u["conv03"] = u["ldepth"] * (0.03 / 0.045)
capture(u["max_low_in_window"] <= u["signal_close"] * (1 - u["conv03"]), u["conv03"], "c8_conv03")

# ---------- Claim 9: wave clusters ----------
evm = events[events["kind"] == "missed"][["symbol", "date", "bar_idx"]]
uw = uf.merge(evm, left_on=["symbol", "signal_date"], right_on=["symbol", "date"],
              how="left").dropna(subset=["bar_idx", "max_runup_42"]).sort_values(
    ["symbol", "bar_idx"])
n_cl_start, ge30_start = 0, 0
n_cl_gap, ge30_gap = 0, 0
for s, g in uw.groupby("symbol"):
    # variant A: new cluster when bar - cluster_START > 42 (script's actual code)
    start, best = None, None
    for r in g.itertuples():
        if start is None or r.bar_idx - start > 42:
            if best is not None:
                n_cl_start += 1; ge30_start += int(best >= 0.30)
            start, best = r.bar_idx, r.max_runup_42
        elif r.max_runup_42 > best:
            best = r.max_runup_42
    if best is not None:
        n_cl_start += 1; ge30_start += int(best >= 0.30)
    # variant B: new cluster when gap from PREVIOUS member > 42 (literal reading)
    prev, best = None, None
    for r in g.itertuples():
        if prev is None or r.bar_idx - prev > 42:
            if best is not None:
                n_cl_gap += 1; ge30_gap += int(best >= 0.30)
            best = r.max_runup_42
        else:
            best = max(best, r.max_runup_42)
        prev = r.bar_idx
    if best is not None:
        n_cl_gap += 1; ge30_gap += int(best >= 0.30)
R["c9_clusters"] = {"script_variant_start_anchor": {"n": n_cl_start, "ge30": ge30_start},
                    "literal_gap_variant": {"n": n_cl_gap, "ge30": ge30_gap},
                    "n_rows_clustered": len(uw)}

# ---------- Claim 10: daily activity ----------
q = da["n_live_pending_limits"]
imax = int(q.idxmax())
da["year"] = da["date"].str[:4]
R["c10"] = {"n_days": len(da), "mean": round(float(q.mean()), 1),
            "median": float(q.median()), "p90": float(q.quantile(.9)),
            "max": int(q.max()), "max_date": da.loc[imax, "date"],
            "pct_gt10": round(float((q > 10).mean()), 4),
            "pct_ge30": round(float((q >= 30).mean()), 4),
            "pct_gt30": round(float((q > 30).mean()), 4),
            "mean_2025plus": round(float(da.loc[da["date"] >= "2025-01-01",
                                                "n_live_pending_limits"].mean()), 1),
            "by_year": da.groupby("year")["n_live_pending_limits"].mean().round(1).to_dict()}

# ---------- Claim 11: engine-placed live ----------
pos_mask = {s: np.zeros(len(a["c"]), dtype=bool) for s, a in A.items()}
for t in tr.itertuples():
    a = A.get(t.symbol)
    if a is None:
        continue
    e = a["idx"].get(str(t.entry_date))
    if e is None:
        continue
    if t.exit_reason == "end_of_data":
        pos_mask[t.symbol][e:] = True
    else:
        x = a["idx"].get(str(t.exit_date))
        if x is not None:
            pos_mask[t.symbol][e:x] = True

live_cnt = defaultdict(int); live_syms = defaultdict(set); live_rows = []
for r in ev.itertuples():
    a = A[r.symbol]; i, w = int(r.bar_idx), int(r.aux_idx)
    for k in range(i + 1, w + 1):
        if r.kind == "missed" and pos_mask[r.symbol][k]:
            break
        dd = a["dates"][k]
        live_cnt[dd] += 1
        live_syms[dd].add(r.symbol)
        live_rows.append((dd, r.symbol, r.date, r.kind, float(r.limit_price),
                          1 if (r.kind == "filled" and k == w) else 0,
                          (a["c"][k - 1] - r.limit_price) / a["c"][k - 1]))
days = sorted(dd for dd in live_cnt if dd >= "2020-01-02")
lc = pd.Series([live_cnt[dd] for dd in days], index=days)
ls = pd.Series([len(live_syms[dd]) for dd in days], index=days)
nf_ = da["n_fills"]; nn_ = da["n_new_buy_signals"]
R["c11"] = {"mean": round(float(lc.mean()), 1), "median": float(lc.median()),
            "p90": round(float(lc.quantile(.9)), 1), "max": int(lc.max()),
            "n_live_days": len(days), "n_da_days": len(da),
            "sym_mean": round(float(ls.mean()), 1),
            "per_sym": round(float(lc.mean()) / float(ls.mean()), 2),
            "last_day": days[-1], "last_live": int(lc.iloc[-1]),
            "last_syms": int(ls.iloc[-1]),
            "new_mean": round(float(nn_.mean()), 1), "new_median": float(nn_.median()),
            "new_p90": float(nn_.quantile(.9)), "new_max": int(nn_.max()),
            "fill_mean": round(float(nf_.mean()), 2), "fill_max": int(nf_.max()),
            "fill_max_date": da.loc[int(nf_.idxmax()), "date"],
            "pct_days_with_fill": round(float((nf_ > 0).mean()), 4)}

# ---------- Claim 12+13: top-K ----------
lv = pd.DataFrame(live_rows, columns=["date", "symbol", "sig_date", "kind", "limit",
                                      "fill_day", "prox"])
lv = lv[lv["date"] >= "2020-01-02"]
sc = sig.rename(columns={"date": "sig_date"})
lv = lv.merge(sc, on=["symbol", "sig_date"], how="left")
tmk = tm.set_index(["symbol", "entry_date", "signal_date"])["pnl_pct"]
tmb = tm.set_index(["symbol", "entry_date", "signal_date"])["bucket"]
lv["r_score"] = lv.groupby("date")["score"].rank(ascending=False, method="first")
lv["r_score3"] = lv.groupby("date")["score3"].rank(ascending=False, method="first")
lv["r_prox"] = lv.groupby("date")["prox"].rank(ascending=True, method="first")
fills = lv[lv["fill_day"] == 1].copy()
fills["pnl"] = [tmk.get((s, d2, sd), np.nan)
                for s, d2, sd in zip(fills["symbol"], fills["date"], fills["sig_date"])]
fills["bucket"] = [tmb.get((s, d2, sd), None)
                   for s, d2, sd in zip(fills["symbol"], fills["date"], fills["sig_date"])]
tot_pnl = float(fills["pnl"].sum()); tot_big = int((fills["bucket"] == "big_win").sum())
R["c12_totals"] = {"n_fills": len(fills), "pnl": round(tot_pnl, 1), "big": tot_big,
                   "n_pnl_missing": int(fills["pnl"].isna().sum()),
                   "score_missing_on_fills": int(fills["score"].isna().sum())}
G = {}
for key, rk in [("score", "r_score"), ("score3", "r_score3"), ("prox", "r_prox")]:
    for K in [5, 10, 20]:
        kept = fills[fills[rk] <= K]
        G[f"{key}{K}"] = {"fills_pct": round(len(kept) / len(fills), 3),
                          "pnl_pct": round(float(kept["pnl"].sum()) / tot_pnl, 3),
                          "pnl_u": round(float(kept["pnl"].sum()), 1),
                          "big_pct": round(int((kept["bucket"] == "big_win").sum()) / tot_big, 3),
                          "big_n": int((kept["bucket"] == "big_win").sum())}
R["c12_topK"] = G
R["c13_lift"] = {"score10": round(G["score10"]["pnl_pct"] / G["score10"]["fills_pct"], 2),
                 "prox10": round(G["prox10"]["pnl_pct"] / G["prox10"]["fills_pct"], 2)}

# ---------- Claim 8: Postgres leaderboard ----------
try:
    import psycopg2
    cn = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                          user="stockml", password="stockml_dev")
    cur = cn.cursor()
    cur.execute("""select table_name, column_name from information_schema.columns
                   where table_name in ('leaderboard_runs','strategy_templates')
                   order by table_name, ordinal_position""")
    R["pg_schema"] = defaultdict(list)
    for t, c in cur.fetchall():
        R["pg_schema"][t].append(c)
    R["pg_schema"] = dict(R["pg_schema"])
    cn.close()
except Exception as e:
    R["pg_schema"] = {"error": str(e)}

print(json.dumps(R, indent=1, default=str))
