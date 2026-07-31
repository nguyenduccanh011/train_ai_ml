# -*- coding: utf-8 -*-
"""Answer user: tried even-lower-K and different EVICTION (duoi/khong duoi) rules?
On champion combo (cs4-conv k2.0 + conv-skip0.40): sweep (A) very-low K {4,5,6,8} and (B) eviction modes:
  none   = never evict (idle-fill only, momentum cannot preempt)
  meta   = champion R2 prio-swap by meta-pred (evict lowest meta-prio)
  conv   = evict lowest conviction (cs4) instead of meta
  loser  = evict worst unrealized-P&L leg (cut losers)
  oldest = evict longest-held leg
  + margin sweep {0.005, 0.01, 0.03}. 3-seed [42,21,123]. Report CAGR/DD."""
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


def prun(sim, pm, cm, K, k_conv, evict, margin, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
    cv = [t["conv"] for t in sim.trades]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean(raw) if raw else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["conv"] >= skip_g:
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

    def mk(t, size, di):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"],
                    prio=t["prio"], conv=t["conv"], di0=di)

    def victim(legs, dt, di):
        if evict == "meta":
            return min(legs, key=lambda l: l["prio"])
        if evict == "conv":
            return min(legs, key=lambda l: l["conv"])
        if evict == "loser":
            return min(legs, key=lambda l: lv(l, dt) / l["invested"] - 1.0)   # worst unrealized
        if evict == "oldest":
            return min(legs, key=lambda l: l["di0"])
        return None

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
                cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs and evict != "none":
                c = victim(legs, dt, di)
                # eviction gate: meta/conv use prio margin; loser/oldest evict if new trade's prio beats victim's
                ok = (t["prio"] - c["prio"] > margin) if evict in ("meta",) else \
                     (t["conv"] - c["conv"] > margin) if evict == "conv" else \
                     (t["prio"] > c["prio"])          # loser/oldest: only if incoming is better meta-prio
                if ok:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, di); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


skip_g = SKIP
cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
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

con = psycopg2.connect(**PG); feat = None; cv_c, cm_c, pm = {}, {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm = {}
    for r in cvtr.itertuples():
        cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
    cm_c[sd] = cm
    cv = HERE / f"_pv_s{sd}.csv"
    cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False); cv_c[sd] = str(cv)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()


def ev(K, kc, evict, margin):
    nv, cg, dd = [], [], []
    for sd in SEEDS:
        f, c, d = prun(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], cm_c[sd], K, kc, evict, margin)
        nv.append(f); cg.append(c); dd.append(d)
    return statistics.mean(cg), statistics.mean(dd), nv


print("=== (A) EVEN-LOWER K (conv-k2.0, champion evict=meta m0.01) ===", flush=True)
print(f"  {'K':4s} | CAGR%  DD%   | 3seed-NAV", flush=True)
for K in (4, 5, 6, 8, 10):
    cg, dd, nv = ev(K, 2.0, "meta", 0.01)
    print(f"  K{K:<3}| {cg*100:5.1f} {dd*100:5.1f} | {[f'{x:.0f}' for x in nv]}", flush=True)

print("\n=== (B) EVICTION MODE (K10, conv-k2.0) ===", flush=True)
print(f"  {'evict':8s} | CAGR%  DD%   | 3seed-NAV", flush=True)
for evict in ("none", "meta", "conv", "loser", "oldest"):
    cg, dd, nv = ev(10, 2.0, evict, 0.01)
    print(f"  {evict:8s} | {cg*100:5.1f} {dd*100:5.1f} | {[f'{x:.0f}' for x in nv]}", flush=True)

print("\n=== (C) MARGIN sweep (K10, conv-k2.0, evict=meta) ===", flush=True)
for m in (0.005, 0.01, 0.03, 0.06):
    cg, dd, nv = ev(10, 2.0, "meta", m)
    print(f"  margin{m:<6}| CAGR {cg*100:5.1f}%  DD {dd*100:5.1f}% | {[f'{x:.0f}' for x in nv]}", flush=True)
print("PREEMPT_VARIANTS_DONE", flush=True)
