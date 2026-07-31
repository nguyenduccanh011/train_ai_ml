# -*- coding: utf-8 -*-
"""Disciplined FORWARD-selection on top of cs5_ma50 (cs4+atrpct+dist_ma50). Each candidate must beat
cs5_ma50 3/3 seed at BOTH K10 AND K12 (stricter multi-config gate -> controls FDR). Candidates (NEW,
not yet tested): dist_ma100, ret60, hi252prox, volz, maspread(ma20/ma50), ret10. Only cross-config-robust
additions are real selection alpha."""
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
SEEDS = [42, 21, 123]; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
CFGS = [("K10", 10, 0.005, 2.0), ("K12", 12, 0.01, 2.0)]


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
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]
    d["date"] = pd.to_datetime(d["date"]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return float(nav.iloc[-1]), float(nav.iloc[-1]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,high,low,close,volume FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h, v = g["close"], g["low"], g["high"], g["volume"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1
    g["dist_ma100"] = c / c.rolling(100).mean() - 1; g["ret60"] = c.pct_change(60); g["ret10"] = c.pct_change(10)
    g["hi252prox"] = c / h.rolling(252).max() - 1; g["volz"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    g["maspread"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "dist_ma100", "ret60", "ret10", "hi252prox", "volz", "maspread"]])
P = pd.concat(parts, ignore_index=True)
CANDS = ["dist_ma100", "ret60", "ret10", "hi252prox", "volz", "maspread"]
for col in CS4 + ["atrpct", "dist_ma50"] + CANDS:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
BASE = [c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]     # cs5_ma50
P["cs5_ma50"] = P[BASE].mean(axis=1)
for cand in CANDS:
    P["add_" + cand] = P[BASE + [cand + "_r"]].mean(axis=1)
SIGS = ["cs5_ma50"] + ["add_" + c for c in CANDS]
CS = {sig: {(r.symbol, str(r.date.date())): (getattr(r, sig) if pd.notna(getattr(r, sig)) else 0.5) for r in P.itertuples()} for sig in SIGS}

con = psycopg2.connect(**PG); feat = None; cv_c, key_c, pm = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    key_c[sd] = [(r.symbol, str(pd.to_datetime(r.entry_date).date()), str(r.sigd.date())) for r in cvtr.itertuples()]
    cvf = HERE / f"_fs_s{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cvf, index=False); cv_c[sd] = str(cvf)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def mapfor(sd, sig):
    return {(sym, ed): CS[sig].get((sym, sgd), 0.5) for sym, ed, sgd in key_c[sd]}


def evalc(sig, K, m, kc):
    r = [prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], mapfor(sd, sig), K, m, kc) for sd in SEEDS]
    return [x[0] for x in r], statistics.mean(x[1] for x in r), statistics.mean(x[2] for x in r)


base = {lab: evalc("cs5_ma50", K, m, kc) for lab, K, m, kc in CFGS}
print("=== forward-select on cs5_ma50: candidate must beat 3/3 at BOTH K10 & K12 ===", flush=True)
print(f"  base cs5_ma50: K10 {100*base['K10'][1]:.1f}% / K12 {100*base['K12'][1]:.1f}%", flush=True)
print(f"  {'candidate':12s} | K10 Δpp 3/3 | K12 Δpp 3/3 | PASS-both?", flush=True)
for cand in CANDS:
    sig = "add_" + cand; line = ""; passes = []
    for lab, K, m, kc in CFGS:
        fin, cg, dd = evalc(sig, K, m, kc); bfin, bcg, _ = base[lab]
        rob = all(fin[i] > bfin[i] for i in range(len(SEEDS)))
        passes.append(rob and cg > bcg)
        line += f" {100*(cg-bcg):+5.1f} {'3/3' if rob else 'x  '} |"
    print(f"  {cand:12s} |{line} {'✓ PASS' if all(passes) else 'no'}", flush=True)
print("FORWARD_SELECT_DONE", flush=True)
