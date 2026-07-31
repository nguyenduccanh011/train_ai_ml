# -*- coding: utf-8 -*-
"""PORTFOLIO-STRUCTURE research (user: fewer slots? more weight-tiers? different construction?).
On champion book (preempt meta-prio + conviction, causal, total<=1), test 3 structural axes:
  (1) K-sweep: 8/10/12/14/16/20/25 slots (concentration via count) at conv-k1.5
  (2) tiered/barbell: discrete weight tiers by conviction quantile (vs continuous conv)
  (3) concentration-via-weight: high conviction-k (2.5/3.0) = dump capital into few effective names @K16
3-seed [42,21,123]. Report NAV/CAGR/DD/Calmar. WIN = beats champion K16 conv-k1.5 on CAGR-per-DD."""
from __future__ import annotations
import os, sys, statistics
from collections import defaultdict
from pathlib import Path
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
SEEDS = [42, 21, 123]; MARGIN = 0.01; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun(sim, pm, cm, K, mode, kc, margin=MARGIN, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["meta"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
        t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; cmu = statistics.mean(cv); csd = statistics.pstdev(cv) or 1.0
    # conviction percentile across all trades (for tiered/barbell)
    ss = sorted(cv); n = len(ss)
    def pct(x):
        import bisect
        return bisect.bisect_left(ss, x) / max(1, n)
    raw = []
    for t in sim.trades:
        zc = (t["conv"] - cmu) / csd; p = pct(t["conv"])
        if mode == "conv":
            w = 1.0 + kc * zc
        elif mode == "tier3":                     # 3 discrete tiers by conviction quantile
            w = 1.6 if p > 0.66 else (1.0 if p > 0.33 else 0.5)
        elif mode == "barbell":                   # top gets big, rest small (concentrate)
            w = 1.8 if p > 0.5 else 0.5
        elif mode == "tier5":
            w = [0.5, 0.75, 1.0, 1.3, 1.7][min(4, int(p * 5))]
        t["_w"] = min(max(w, 0.4), 1.8); raw.append(t["_w"]); t["prio"] = t["meta"]
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=True)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    def mk(t, size):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"])

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' "
                "ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    parts.append(g[["symbol", "date"] + CS4])
PANEL = pd.concat(parts, ignore_index=True)
for col in CS4:
    PANEL[col + "_r"] = PANEL.groupby("date")[col].rank(pct=True)
PANEL["cs4"] = PANEL[[c + "_r" for c in CS4]].mean(axis=1)
CS = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in PANEL.itertuples()}

con = psycopg2.connect(**PG); feat = None
cv_c, cm_c, tr_c = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                       "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm = {}
    for r in cvtr.itertuples():
        cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
    cm_c[sd] = cm
    cv = HERE / f"_ps_s{sd}.csv"
    cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False); cv_c[sd] = str(cv)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    tr_c[sd] = M.build_tr(con, rid, feat)
con.close()
pm = {sd: M.meta_preds(tr_c[sd], tgt='t_pnl') for sd in SEEDS}


def run(name, K, mode, kc):
    nv, cg, dd = [], [], []
    for sd in SEEDS:
        f, c, d = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cm_c[sd], K, mode, kc)
        nv.append(f); cg.append(c); dd.append(d)
    mn = statistics.mean(nv); mcg = statistics.mean(cg); mdd = statistics.mean(dd)
    print(f"  {name:26s} | x{mn:6.2f}  {mcg*100:5.1f}  {mdd*100:5.1f}  | {mcg/abs(mdd):5.2f}", flush=True)
    return mn, mcg, mdd


print("PORTFOLIO STRUCTURE on champion book (preempt meta-prio, 3-seed):", flush=True)
print(f"  {'variant':26s} | NAV      CAGR%   DD%   | Calmar", flush=True)
print("  -- (1) K-sweep (conv k1.5) --", flush=True)
for K in (8, 10, 12, 14, 16, 20, 25):
    run(f"K{K} conv-k1.5", K, "conv", 1.5)
print("  -- (2) weight-tiers @K16 --", flush=True)
for nm, md in [("tier3 (1.6/1.0/0.5)", "tier3"), ("tier5", "tier5"), ("barbell (1.8/0.5)", "barbell")]:
    run(nm, 16, md, 0)
print("  -- (3) concentration-via-weight @K16 --", flush=True)
for k in (2.0, 2.5, 3.0):
    run(f"conv-k{k}", 16, "conv", k)
print("STRUCT_DONE", flush=True)
