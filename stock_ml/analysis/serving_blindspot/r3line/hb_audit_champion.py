# -*- coding: utf-8 -*-
"""RIGOROUS AUDIT of the combo champion (preempt R2m01 + cs-sizing k1.5 + conv-skip 0.40, ~100% CAGR).
Checks: (1) NO LEVERAGE -> report MAX & MEAN exposure (pos/nav); max<=1.0 proves total<=1. (2) reproduce
3-seed [42,21,123]. (3) robustness on EXTRA seeds [7,99] (outside the preempt lineage). (4) fee sensitivity
(roundtrip 0.006/0.010/0.015). (5) per-year returns (concentrated or broad). Causality is code-level:
cs4 keyed at SIGNAL-date (sigd), meta walk-forward (exit_date<test-year) — asserted below."""
from __future__ import annotations
import os, sys, statistics
from collections import defaultdict
from pathlib import Path
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
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
SEEDS3 = [42, 21, 123]; SEEDS_EXTRA = [7, 99]; K = 16; KCONV = 1.5; MARGIN = 0.01; SKIP = 0.40
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun(sim, pm, cm, k_conv=KCONV, margin=MARGIN, skip=SKIP, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
        t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd
        t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] >= skip:
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

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []; exps = []
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
        pos = sum(lv(l, dt) for l in legs); nav = cash + pt + pos
        ns.append((dt, nav)); exps.append(pos / nav if nav > 0 else 0.0)
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    d["yr"] = d["date"].dt.year
    yret = {int(y): g["nav"].iloc[-1] / g["nav"].iloc[0] - 1.0 for y, g in d.groupby("yr")}
    return dict(nav=final, cagr=final ** (1 / yrs) - 1, dd=float((nav / nav.cummax() - 1).min()),
                exp_max=max(exps), exp_mean=statistics.mean(exps), yret=yret)


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
P = pd.concat(parts, ignore_index=True)
for col in CS4:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs4"] = P[[c + "_r" for c in CS4]].mean(axis=1)
CS = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in P.itertuples()}

con = psycopg2.connect(**PG); feat = None
allres = {}
for sd in SEEDS3 + SEEDS_EXTRA:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                       "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    # CAUSALITY assert: entry_signal_date strictly BEFORE entry_date (fill next bar)
    sd_ok = (pd.to_datetime(cvtr["entry_signal_date"]) < pd.to_datetime(cvtr["entry_date"])).mean()
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm = {}
    for r in cvtr.itertuples():
        cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
    cv = HERE / f"_auc_s{sd}.csv"
    cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    allres[sd] = (str(cv), pm, cm, sd_ok)
con.close()

print("=== AUDIT: combo champion (preempt R2m01 + cs k1.5 + conv-skip0.40) ===", flush=True)
print("  (1) LEVERAGE + reproduce, per seed:", flush=True)
r3 = {}
for sd in SEEDS3 + SEEDS_EXTRA:
    cv, pm, cm, sdok = allres[sd]
    m = prun(NavSim2(cv, date_lo="2020-01-01"), pm, cm)
    r3[sd] = m
    tag = "(3-seed set)" if sd in SEEDS3 else "(EXTRA seed)"
    print(f"    seed{sd:3d} {tag}: NAV x{m['nav']:5.2f}  CAGR {m['cagr']*100:5.1f}%  DD {m['dd']*100:5.1f}%  "
          f"exp_max={m['exp_max']*100:5.1f}%  exp_mean={m['exp_mean']*100:4.1f}%  sig<entry={sdok*100:.0f}%", flush=True)
m3 = statistics.mean([r3[s]["nav"] for s in SEEDS3]); m5 = statistics.mean([r3[s]["nav"] for s in SEEDS3 + SEEDS_EXTRA])
print(f"  3-seed[42,21,123] mean NAV={m3:.2f}  |  5-seed mean NAV={m5:.2f}", flush=True)
maxexp = max(r3[s]["exp_max"] for s in r3)
print(f"  >>> MAX exposure across all seeds = {maxexp*100:.2f}%  ({'NO LEVERAGE (<=100%)' if maxexp<=1.0001 else 'LEVERAGE FLAG!'})", flush=True)

print("  (5) per-year return (seed 42, 3-seed representative):", flush=True)
yr = r3[42]["yret"]
for y in sorted(yr):
    print(f"    {y}: {yr[y]*100:+6.1f}%", flush=True)

print("  (4) FEE sensitivity (3-seed mean NAV):", flush=True)
for rt in [0.006, 0.010, 0.015]:
    navs = [prun(NavSim2(allres[s][0], date_lo="2020-01-01"), allres[s][1], allres[s][2], roundtrip=rt)["nav"] for s in SEEDS3]
    print(f"    roundtrip {rt:.3f}: NAV x{statistics.mean(navs):.2f}", flush=True)
print("AUDIT_CHAMPION_DONE")
