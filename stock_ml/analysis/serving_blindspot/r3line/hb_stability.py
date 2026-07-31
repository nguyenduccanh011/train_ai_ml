# -*- coding: utf-8 -*-
"""Which board model is the STABLE elite (not overfit)? For top candidates compute 3-seed CAGR mean +
DISPERSION (std/CV across seeds — high CV = seed-fragile = overfit smell), DD, and per-year MIN (worst
year, 3-seed mean). Stable elite = high CAGR + low seed-CV + high worst-year + acceptable DD + fewest
tuned/searched knobs. K10 low-slot = higher variance; cs5_ma50 = searched. Champion K16/cs4 = most-validated."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, duckdb
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]; YEARS = list(range(2020, 2027))
# (label, K, margin, kconv, sig, n_tuned_knobs)
CANDS = [
    ("K16/cs4 (CHAMPION)", 16, 0.01, 1.5, "cs4", 0),
    ("K12/cs4", 12, 0.01, 2.0, "cs4", 1),
    ("K10/cs4", 10, 0.005, 2.0, "cs4", 3),
    ("K12/cs5ma50", 12, 0.01, 2.0, "cs5_ma50", 3),
    ("K10/cs5 ", 10, 0.005, 2.0, "cs5", 4),
    ("K10/cs5ma50 (TOP)", 10, 0.005, 2.0, "cs5_ma50", 5),
]


def prun(sim, pm, cmap, K, margin, kconv):
    s_new = (0.006 - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cmap.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + kconv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] >= SKIP:
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
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - 0.0008); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - 0.0008); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    py = {}
    dd_ = d.set_index("date")["nav"]
    for y in YEARS:
        g = dd_[dd_.index.year == y]
        py[y] = (g.iloc[-1] / g.iloc[0] - 1) if len(g) > 2 else None
    return float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min()), py


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
b4 = [c + "_r" for c in CS4]
P["cs4"] = P[b4].mean(axis=1); P["cs5"] = P[b4 + ["atrpct_r"]].mean(axis=1); P["cs5_ma50"] = P[b4 + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CS = {sig: {(r.symbol, str(r.date.date())): (getattr(r, sig) if pd.notna(getattr(r, sig)) else 0.5) for r in P.itertuples()} for sig in ("cs4", "cs5", "cs5_ma50")}

con = psycopg2.connect(**PG); feat = None; cv_c, key_c, pm = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    key_c[sd] = [(r.symbol, str(pd.to_datetime(r.entry_date).date()), str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_st_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def mapfor(sd, sig):
    return {(sym, ed): CS[sig].get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]}


print("=== STABILITY of top candidates (3-seed) ===", flush=True)
print(f"  {'config':20s} | CAGR mean | seed-CV | DD | worst-yr | knobs", flush=True)
rows = []
for lab, K, m, kc, sig, nk in CANDS:
    cgs, dds, pys = [], [], defaultdict(list)
    for sd in SEEDS:
        cg, dd, py = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], mapfor(sd, sig), K, m, kc)
        cgs.append(cg); dds.append(dd)
        for y in YEARS:
            if py[y] is not None:
                pys[y].append(py[y])
    mean_cg = statistics.mean(cgs); cv = statistics.pstdev(cgs) / mean_cg
    worst = min(statistics.mean(pys[y]) for y in YEARS if pys[y])
    rows.append((lab, mean_cg, cv, statistics.mean(dds), worst, nk))
    print(f"  {lab:20s} | {100*mean_cg:6.1f}% | {100*cv:5.1f}% | {100*statistics.mean(dds):5.1f} | {100*worst:+5.0f}% | {nk}", flush=True)
print("\n(seed-CV low = robust; worst-yr high = consistent; knobs low = less overfit-prone)", flush=True)
print("STABILITY_DONE", flush=True)
