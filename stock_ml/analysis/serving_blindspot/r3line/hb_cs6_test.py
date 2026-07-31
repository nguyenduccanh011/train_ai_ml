# -*- coding: utf-8 -*-
"""Extend the forensic selection-feature win: add more forensic-B separators on top of cs5(+atrpct) at K10.
Candidates: +ret5 (short-mom), +dist63hi (breakout proximity), +gap, and double-atrpct weight. Strict:
3/3 seed + per-year >=5/7 to be a real win (guard against feature-piling overfit). K10/m005/convk2.0."""
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
SEEDS = [42, 21, 123]; K = 10; MARGIN = 0.005; KCONV = 2.0; SKIP = 0.40
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]; YEARS = list(range(2020, 2027))


def prun(sim, pm, cmap):
    s_new = (0.006 - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cmap.get((t["symbol"], t["entry_date"]), 0.5)
    cvv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cvv); sd = statistics.pstdev(cvv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + KCONV * z, 0.4), 1.8); raw.append(t["_w"])
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
                if t["prio"] - c["prio"] > MARGIN:
                    vnow = lv(c, dt); cash += vnow * (1.0 - 0.0008); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"])
    return d


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,open,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h, o = g["close"], g["low"], g["high"], g["open"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["ret5"] = c.pct_change(5); g["dist63hi"] = c / h.rolling(63).max() - 1
    g["gap"] = o / c.shift() - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "ret5", "dist63hi", "gap"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "ret5", "dist63hi", "gap"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
base4 = [c + "_r" for c in CS4]
P["cs4"] = P[base4].mean(axis=1)
P["cs5"] = P[base4 + ["atrpct_r"]].mean(axis=1)
P["cs6_ret5"] = P[base4 + ["atrpct_r", "ret5_r"]].mean(axis=1)
P["cs6_63hi"] = P[base4 + ["atrpct_r", "dist63hi_r"]].mean(axis=1)
P["cs6_gap"] = P[base4 + ["atrpct_r", "gap_r"]].mean(axis=1)
P["cs5_atr2"] = P[base4 + ["atrpct_r", "atrpct_r"]].mean(axis=1)   # double atrpct weight
P["cs6_all"] = P[base4 + ["atrpct_r", "ret5_r", "dist63hi_r"]].mean(axis=1)
SIGS = ["cs4", "cs5", "cs6_ret5", "cs6_63hi", "cs6_gap", "cs5_atr2", "cs6_all"]
CS = {sig: {(r.symbol, str(r.date.date())): (getattr(r, sig) if pd.notna(getattr(r, sig)) else 0.5) for r in P.itertuples()} for sig in SIGS}

con = psycopg2.connect(**PG); feat = None; cv_c, key_c, pm = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    key_c[sd] = [(r.symbol, str(pd.to_datetime(r.entry_date).date()), str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_c6_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def mapfor(sd, sig):
    return {(sym, ed): CS[sig].get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]}


def evalsig(sig):
    navs = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], mapfor(sd, sig)) for sd in SEEDS]
    fin = [float(d["nav"].iloc[-1]) for d in navs]
    yrs = (navs[0]["date"].iloc[-1] - navs[0]["date"].iloc[0]).days / 365.25
    cg = statistics.mean(f ** (1 / yrs) - 1 for f in fin); dd = statistics.mean(float((d["nav"] / d["nav"].cummax() - 1).min()) for d in navs)
    py = {}
    for y in YEARS:
        vs = []
        for d in navs:
            ss = d.set_index("date")["nav"]; g = ss[ss.index.year == y]
            if len(g) > 2:
                vs.append(g.iloc[-1] / g.iloc[0] - 1)
        py[y] = statistics.mean(vs) if vs else 0.0
    return fin, cg, dd, py


R = {sig: evalsig(sig) for sig in SIGS}
f5, c5, d5, py5 = R["cs5"]
print("=== extend cs5 with more forensic features (K10/m005/convk2.0), 3-seed ===", flush=True)
print(f"  {'signal':10s} | NAV(3seed)        | CAGR%  DD%   | vs cs5(3/3?) | yrs>=cs5", flush=True)
for sig in SIGS:
    fin, cg, dd, py = R[sig]
    rob = all(fin[i] > f5[i] for i in range(len(SEEDS)))
    wins = sum(1 for y in YEARS if py[y] >= py5[y] - 0.005)
    tag = "(baseline)" if sig == "cs5" else ("3/3 " + ("✓" if rob else "x") + f" {wins}/7")
    print(f"  {sig:10s} | {[f'{x:.0f}' for x in fin]} | {100*cg:5.1f} {100*dd:5.1f} | {tag}", flush=True)
print("CS6_TEST_DONE", flush=True)
