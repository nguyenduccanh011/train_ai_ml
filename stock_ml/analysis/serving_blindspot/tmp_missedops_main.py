# -*- coding: utf-8 -*-
"""tmp_missedops: complaints #1 (missed momentum) & #2 (operational load).

Sections:
  A fill-rate overall & by year (engine-placed limits: events kind filled|missed)
  B unfilled cohort return distributions + theoretical at-market pnl vs realized
  C fill_if_missed verification (window-end chase entry) incl. premium caps p2/p4
  D near-miss depth analysis + capture at flat 0.03 / 0.02 / conv-rescaled 0.03
  E top-15 biggest missed runs (per-symbol wave clusters)
  F daily operational load stats (daily_activity.csv + engine-placed live count)
  G top-K pending-limit policy simulation (rank by score / score3 / proximity)
"""
import json
import os
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
WINDOW = 40
COST_RT = 0.0015 + 0.0015 + 0.0015 + 0.0015 + 0.001  # buy c+s, sell c+s+tax = 0.7%

os.chdir(OUT)
events = pd.read_csv("events.csv")
uf_all = pd.read_csv("unfilled_signals.csv")
tm = pd.read_csv("trades_metrics.csv")          # 3,781 closed
tr = pd.read_csv("trades_raw.csv")              # 3,813 incl. open
da = pd.read_csv("daily_activity.csv")
sig = pd.read_csv("signals.csv", usecols=["symbol", "date", "score", "score3"])

con = sqlite3.connect(DB)
oh = pd.read_sql("select symbol, date, low, close from ohlcv order by symbol, date", con)
con.close()
A = {}
for s, g in oh.groupby("symbol"):
    g = g.reset_index(drop=True)
    A[s] = {"dates": g["date"].to_numpy(), "idx": {d: k for k, d in enumerate(g["date"])},
            "l": g["low"].to_numpy(float), "c": g["close"].to_numpy(float)}

res = {}

# ---------- A. fill rate ----------
ev = events[events["kind"].isin(["filled", "missed"])].copy()
ev["year"] = ev["date"].str[:4]
byyr = ev.pivot_table(index="year", columns="kind", values="symbol", aggfunc="count").fillna(0)
byyr["placed"] = byyr["filled"] + byyr["missed"]
byyr["fill_rate"] = (byyr["filled"] / byyr["placed"]).round(3)
res["A_fill_rate_overall"] = {"filled": int(byyr["filled"].sum()), "missed": int(byyr["missed"].sum()),
                              "placed": int(byyr["placed"].sum()),
                              "fill_rate": round(byyr["filled"].sum() / byyr["placed"].sum(), 4)}
res["A_fill_rate_by_year"] = byyr[["filled", "missed", "placed", "fill_rate"]].astype(object).to_dict("index")

# ---------- B. unfilled cohort ----------
uf = uf_all[uf_all["drop_reason"].isin(["unfilled", "unfilled_window_open"])].copy()
res["B_n_unfilled"] = len(uf)
dist = {}
for c in ["atmarket_ret5", "atmarket_ret10", "atmarket_ret21", "atmarket_ret42", "max_runup_42"]:
    v = uf[c].dropna()
    dist[c] = {"n": len(v), "mean": round(v.mean(), 4), "median": round(v.median(), 4),
               "p25": round(v.quantile(.25), 4), "p75": round(v.quantile(.75), 4),
               "p90": round(v.quantile(.90), 4), "pct_pos": round((v > 0).mean(), 3),
               "sum_gross": round(v.sum(), 1)}
res["B_unfilled_distributions"] = dist
for h in ["atmarket_ret21", "atmarket_ret42"]:
    v = uf[h].dropna()
    res[f"B_theoretical_{h}"] = {"sum_gross_u": round(v.sum(), 1),
                                 "sum_net_u_after_0.7pct_rt": round(v.sum() - COST_RT * len(v), 1),
                                 "mean_net": round(v.mean() - COST_RT, 4)}
res["B_realized_closed"] = {"n": len(tm), "sum_pnl_u": round(tm["pnl_pct"].sum(), 1),
                            "mean_pnl": round(tm["pnl_pct"].mean(), 4),
                            "median_hold_bars": float(tm["holding_days"].median())}

# ---------- C. fill_if_missed chase (window-end close entry) ----------
miss = events[events["kind"] == "missed"].copy()
rows = []
for r in miss.itertuples():
    a = A[r.symbol]
    i, w = int(r.bar_idx), int(r.aux_idx)
    n = len(a["c"])
    if w >= n - 1 or w - i < WINDOW:      # window truncated by end of data -> skip
        continue
    ec = a["c"][w]
    prem = ec / a["c"][i] - 1.0
    r21 = a["c"][w + 21] / ec - 1.0 if w + 21 < n else np.nan
    r42 = a["c"][w + 42] / ec - 1.0 if w + 42 < n else np.nan
    rows.append((r.symbol, r.date, prem, r21, r42))
ch = pd.DataFrame(rows, columns=["symbol", "date", "premium", "chase_ret21", "chase_ret42"])
def coh(d, tag):
    o = {}
    for c in ["chase_ret21", "chase_ret42"]:
        v = d[c].dropna()
        o[c] = {"n": len(v), "mean": round(v.mean(), 4), "median": round(v.median(), 4),
                "pct_pos": round((v > 0).mean(), 3), "sum": round(v.sum(), 1),
                "mean_net": round(v.mean() - COST_RT, 4)}
    o["premium_mean"] = round(d["premium"].mean(), 4)
    o["premium_median"] = round(d["premium"].median(), 4)
    res[tag] = o
coh(ch, "C_chase_all")
coh(ch[ch["premium"] <= 0.02], "C_chase_prem_le2pct_fm_p2")
coh(ch[ch["premium"] <= 0.04], "C_chase_prem_le4pct_fm_p4")
coh(ch[ch["premium"] > 0.04], "C_chase_prem_gt4pct")
res["C_n_by_prem"] = {"all": len(ch), "le2": int((ch["premium"] <= .02).sum()),
                      "le4": int((ch["premium"] <= .04).sum()), "gt4": int((ch["premium"] > .04).sum())}

# ---------- D. near-miss & shallower depth capture ----------
u = uf.dropna(subset=["max_low_in_window", "limit_price", "signal_close"]).copy()
u["reached_depth"] = 1 - u["max_low_in_window"] / u["signal_close"]
u["limit_depth"] = 1 - u["limit_price"] / u["signal_close"]
u["shortfall"] = (u["max_low_in_window"] - u["limit_price"]) / u["signal_close"]
res["D_limit_depth_actual"] = {"mean": round(u["limit_depth"].mean(), 4),
                               "median": round(u["limit_depth"].median(), 4),
                               "p10": round(u["limit_depth"].quantile(.1), 4),
                               "p90": round(u["limit_depth"].quantile(.9), 4)}
res["D_reached_depth"] = {"mean": round(u["reached_depth"].mean(), 4),
                          "median": round(u["reached_depth"].median(), 4),
                          "p75": round(u["reached_depth"].quantile(.75), 4),
                          "p90": round(u["reached_depth"].quantile(.9), 4)}
res["D_shortfall_dist"] = {"median": round(u["shortfall"].median(), 4),
                           "within_0.5pp": round((u["shortfall"] <= .005).mean(), 3),
                           "within_1pp": round((u["shortfall"] <= .01).mean(), 3),
                           "within_2pp": round((u["shortfall"] <= .02).mean(), 3)}

def capture(mask, entry_depth_series, tag):
    cap = u[mask].copy()
    o = {"n_captured": len(cap), "capture_rate_of_unfilled": round(len(cap) / len(u), 3)}
    r21l, r42l, ru = [], [], []
    for r in cap.itertuples():
        a = A.get(r.symbol)
        i = a["idx"].get(r.signal_date) if a else None
        if i is None:
            continue
        n = len(a["c"])
        entry = r.signal_close * (1 - (entry_depth_series[r.Index] if hasattr(entry_depth_series, "__getitem__") else entry_depth_series))
        if i + 22 < n:
            r21l.append(a["c"][i + 22] / entry - 1)
        if i + 43 < n:
            r42l.append(a["c"][i + 43] / entry - 1)
        ru.append(r.max_runup_42)
    o["ret21_from_limit_mean"] = round(float(np.mean(r21l)), 4) if r21l else None
    o["ret21_from_limit_sum_gross"] = round(float(np.sum(r21l)), 1) if r21l else None
    o["ret42_from_limit_mean"] = round(float(np.mean(r42l)), 4) if r42l else None
    o["ret42_from_limit_sum_gross"] = round(float(np.sum(r42l)), 1) if r42l else None
    o["max_runup_42_mean"] = round(float(np.nanmean(ru)), 4) if ru else None
    res[tag] = o

capture(u["max_low_in_window"] <= u["signal_close"] * 0.97, 0.03, "D_capture_flat_0.03")
capture(u["max_low_in_window"] <= u["signal_close"] * 0.98, 0.02, "D_capture_flat_0.02")
u["conv_limit_03"] = u["signal_close"] * (1 - u["limit_depth"] * (0.03 / 0.045))
capture(u["max_low_in_window"] <= u["conv_limit_03"], u["limit_depth"] * (0.03 / 0.045), "D_capture_conv_rescaled_0.03")

# ---------- E. top-15 missed runs (wave clusters) ----------
evm = events[events["kind"] == "missed"][["symbol", "date", "bar_idx"]]
uw = uf.merge(evm, left_on=["symbol", "signal_date"], right_on=["symbol", "date"], how="left")
uw = uw.dropna(subset=["bar_idx", "max_runup_42"]).sort_values(["symbol", "bar_idx"])
clusters = []
for s, g in uw.groupby("symbol"):
    start, best = None, None
    for r in g.itertuples():
        if start is None or r.bar_idx - start > 42:
            if best is not None:
                clusters.append(best)
            start, best = r.bar_idx, r
        elif r.max_runup_42 > best.max_runup_42:
            best = r
    if best is not None:
        clusters.append(best)
cl = pd.DataFrame([{"symbol": c.symbol, "signal_date": c.signal_date,
                    "max_runup_42": round(c.max_runup_42, 3),
                    "atmarket_ret42": round(c.atmarket_ret42, 3) if pd.notna(c.atmarket_ret42) else None,
                    "drop_reason": c.drop_reason} for c in clusters])
res["E_top15_missed_runs"] = cl.sort_values("max_runup_42", ascending=False).head(15).to_dict("records")
res["E_n_wave_clusters"] = len(cl)
res["E_clusters_runup_ge30"] = int((cl["max_runup_42"] >= 0.30).sum())

# ---------- F. operational load ----------
q = da["n_live_pending_limits"]
res["F_daily_activity_live_limits"] = {
    "n_days": len(da), "mean": round(q.mean(), 1), "median": float(q.median()),
    "p90": float(q.quantile(.9)), "max": int(q.max()),
    "days_gt10": int((q > 10).sum()), "pct_days_gt10": round((q > 10).mean(), 3),
    "days_gt50": int((q > 50).sum()), "days_gt100": int((q > 100).sum()),
    "mean_2025plus": round(da.loc[da["date"] >= "2025-01-01", "n_live_pending_limits"].mean(), 1)}
res["F_daily_fills"] = {"mean": round(da["n_fills"].mean(), 2), "median": float(da["n_fills"].median()),
                        "p90": float(da["n_fills"].quantile(.9)), "max": int(da["n_fills"].max()),
                        "pct_days_with_fill": round((da["n_fills"] > 0).mean(), 3)}
res["F_daily_new_limits"] = {"mean": round(da["n_new_buy_signals"].mean(), 1),
                             "median": float(da["n_new_buy_signals"].median()),
                             "p90": float(da["n_new_buy_signals"].quantile(.9)),
                             "max": int(da["n_new_buy_signals"].max())}
da["year"] = da["date"].str[:4]
res["F_live_by_year"] = da.groupby("year")["n_live_pending_limits"].mean().round(1).to_dict()

# engine-placed live count (only the 11,395 real placements, stacked-missed cancelled on position)
pos_mask = {s: np.zeros(len(a["c"]), dtype=bool) for s, a in A.items()}
for t in tr.itertuples():
    a = A.get(t.symbol)
    if a is None:
        continue
    e = a["idx"].get(str(t.entry_date))
    if e is None:
        continue
    x = len(a["c"]) - 1 if t.exit_reason == "end_of_data" else a["idx"].get(str(t.exit_date))
    if x is not None:
        pos_mask[t.symbol][e:x + (1 if t.exit_reason == "end_of_data" else 0)] = True

live_cnt = defaultdict(int)
live_syms = defaultdict(set)
live_rows = []           # for top-K sim: (date, symbol, sig_date, kind, limit, fill_day)
for r in ev.itertuples():
    a = A[r.symbol]
    i, w = int(r.bar_idx), int(r.aux_idx)
    for k in range(i + 1, w + 1):
        if r.kind == "missed" and pos_mask[r.symbol][k]:
            break
        d = a["dates"][k]
        live_cnt[d] += 1
        live_syms[d].add(r.symbol)
        live_rows.append((d, r.symbol, r.date, r.kind, float(r.limit_price),
                          1 if (r.kind == "filled" and k == w) else 0,
                          (a["c"][k - 1] - r.limit_price) / a["c"][k - 1]))
days = sorted(d for d in live_cnt if d >= "2020-01-02")
lc = pd.Series([live_cnt[d] for d in days], index=days)
ls = pd.Series([len(live_syms[d]) for d in days], index=days)
res["F_engine_placed_live"] = {"mean": round(lc.mean(), 1), "median": float(lc.median()),
                               "p90": float(lc.quantile(.9)), "max": int(lc.max()),
                               "mean_2025plus": round(lc[lc.index >= "2025-01-01"].mean(), 1),
                               "distinct_symbols_mean": round(ls.mean(), 1),
                               "distinct_symbols_2025plus": round(ls[ls.index >= "2025-01-01"].mean(), 1),
                               "last_day": days[-1], "last_day_live": int(lc.iloc[-1]),
                               "last_day_symbols": int(ls.iloc[-1])}

# ---------- G. top-K policy simulation ----------
lv = pd.DataFrame(live_rows, columns=["date", "symbol", "sig_date", "kind", "limit", "fill_day", "prox"])
lv = lv[lv["date"] >= "2020-01-02"]
sc = sig.rename(columns={"date": "sig_date"})
lv = lv.merge(sc, on=["symbol", "sig_date"], how="left")
tmk = tm.set_index(["symbol", "entry_date", "signal_date"])["pnl_pct"]
tmb = tm.set_index(["symbol", "entry_date", "signal_date"])["bucket"]

lv["r_score"] = lv.groupby("date")["score"].rank(ascending=False, method="first")
lv["r_score3"] = lv.groupby("date")["score3"].rank(ascending=False, method="first")
lv["r_prox"] = lv.groupby("date")["prox"].rank(ascending=True, method="first")

fills = lv[lv["fill_day"] == 1].copy()
fills["pnl"] = [tmk.get((s, d, sd), np.nan) for s, d, sd in zip(fills["symbol"], fills["date"], fills["sig_date"])]
fills["bucket"] = [tmb.get((s, d, sd), None) for s, d, sd in zip(fills["symbol"], fills["date"], fills["sig_date"])]
tot_pnl = fills["pnl"].sum()
tot_big = (fills["bucket"] == "big_win").sum()
res["G_totals"] = {"n_fills": len(fills), "closed_pnl_sum": round(tot_pnl, 1), "n_big_win": int(tot_big)}
G = {}
for key, rk in [("score", "r_score"), ("score3", "r_score3"), ("proximity", "r_prox")]:
    for K in [3, 5, 10, 20]:
        kept = fills[fills[rk] <= K]
        G[f"{key}_top{K}"] = {
            "fills_kept": len(kept), "fills_kept_pct": round(len(kept) / len(fills), 3),
            "pnl_kept_u": round(kept["pnl"].sum(), 1),
            "pnl_kept_pct": round(kept["pnl"].sum() / tot_pnl, 3),
            "big_wins_kept": int((kept["bucket"] == "big_win").sum()),
            "big_wins_kept_pct": round((kept["bucket"] == "big_win").sum() / tot_big, 3)}
res["G_topK"] = G

with open("tmp_missedops_result.json", "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
