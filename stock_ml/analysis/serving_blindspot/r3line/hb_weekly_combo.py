# -*- coding: utf-8 -*-
"""BREAKTHROUGH probe: multi-timeframe (weekly) gate on the combo. Weekly-UP entries = 67% winners vs
weekly-down 52% (diag_mtf_rs orthogonal +0.05). Test as an OFFLINE overlay on the champion (preempt R2m01
+ cs k1.5 + conv-skip 0.40): skip or down-size trades whose SIGNAL-date weekly trend is DOWN. 3-seed
[42,21,123]. WIN = beats champion 3/3 (weekly adds orthogonal alpha)."""
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
SEEDS = [42, 21, 123]; K = 16; KCONV = 1.5; MARGIN = 0.01; SKIP = 0.40; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun(sim, pm, cm, wk, mode="base", k_conv=KCONV, margin=MARGIN, skip=SKIP, advance_fee=0.0008, roundtrip=0.006):
    """mode: base | wskip (skip weekly-down) | wsize (down-size weekly-down x0.5)."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
        t["wup"] = wk.get((t["symbol"], t["entry_date"]), 1.0)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; w = min(max(1.0 + k_conv * z, 0.4), 1.8)
        if mode == "wsize" and t["wup"] < 0.5:
            w *= 0.5
        t["_w"] = w; raw.append(w)
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] < skip:
            continue
        if mode == "wskip" and t["wup"] < 0.5:
            continue
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


# cs4 + weekly-up panels
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-01-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []; wparts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    parts.append(g[["symbol", "date"] + CS4])
    gi = g.set_index("date"); wk = gi["close"].resample("W-FRI").last()
    wup = ((wk > wk.rolling(10).mean()) & (wk.rolling(10).mean().diff() > 0)).astype(float).reindex(gi.index, method="ffill")
    wparts.append(pd.DataFrame({"symbol": s, "date": gi.index, "wup": wup.values}))
P = pd.concat(parts, ignore_index=True)
for col in CS4:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs4"] = P[[c + "_r" for c in CS4]].mean(axis=1)
CS = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in P.itertuples()}
WK = {(r.symbol, str(r.date.date())): (r.wup if pd.notna(r.wup) else 1.0) for r in pd.concat(wparts, ignore_index=True).itertuples()}

con = psycopg2.connect(**PG); feat = None
MODES = ["base", "wskip", "wsize"]; res = {m: {} for m in MODES}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades "
                       "where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm, wkm = {}, {}
    for r in cvtr.itertuples():
        key = (r.symbol, str(pd.to_datetime(r.entry_date).date())); sk = str(r.sigd.date())
        cm[key] = CS.get((r.symbol, sk), 0.5); wkm[key] = WK.get((r.symbol, sk), 1.0)
    cv = HERE / f"_wk_{sd}.csv"; cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')

    def mk():
        sim = NavSim2(str(cv), date_lo="2020-01-01")
        for t in sim.trades:
            key = (t["symbol"], str(pd.to_datetime(t["entry_date"]).date()))
            t["score"] = 0.0
        return sim
    for m in MODES:
        res[m][sd] = prun(NavSim2(str(cv), date_lo="2020-01-01"), pm, cm, wkm, mode=m)
    print(f"seed{sd} done", flush=True)
con.close()

print("\n=== WEEKLY-GATE on combo, 3-seed, vs base ===", flush=True)
base = res["base"]
for m in MODES:
    navs = [res[m][s][0] for s in SEEDS]; cg = statistics.mean([res[m][s][1] for s in SEEDS]); dd = statistics.mean([res[m][s][2] for s in SEEDS])
    d = [res[m][s][0] - base[s][0] for s in SEEDS]; signs = "".join("+" if x > 0 else "-" for x in d)
    print(f"  {m:6s}: NAV x{statistics.mean(navs):5.2f}  CAGR {cg*100:5.1f}%  DD {dd*100:5.1f}%  | Δ={[f'{x:+.1f}' for x in d]} {signs}", flush=True)
print("WEEKLY_COMBO_DONE")
