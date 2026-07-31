# -*- coding: utf-8 -*-
"""FORENSIC: instrument the combo champion (preempt R2m01 + cs-sizing k1.5) and dissect the PROFIT
STRUCTURE. Logs every filled leg (invested fraction, realized return, conviction cs4, meta-prio, hold,
whether preempted/evicted) -> concentration (Pareto), by-year, by-conviction-bucket, by-hold, and
EVICTION quality (did we cut winners or losers?). Seed 42 detailed; profit-share = invested_frac*ret."""
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
SEED = 42; K = 16; KCONV = 1.5; MARGIN = 0.01; CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]


def prun_log(sim, pm, cm, k_conv=KCONV, margin=MARGIN, advance_fee=0.0008, roundtrip=0.006):
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

    def mk(t, size, nav_at):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"],
                    conv=t["conv"], entry_dt=t["entry_date"], nav_at=nav_at, w=t["w"])

    log = []
    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                val = leg["invested"] * (1.0 + leg["net"])
                cash += val * (1.0 - advance_fee); legs.remove(leg)
                log.append({**{k: leg[k] for k in ("symbol", "entry_dt", "conv", "prio", "w", "invested", "nav_at")},
                            "exit_dt": dt, "ret": (1.0 + leg["net"]) - 1.0, "evicted": 0})
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size, nav_now); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()):
                        exits[c["exit_date"]].remove(c)
                    log.append({**{k: c[k] for k in ("symbol", "entry_dt", "conv", "prio", "w", "invested", "nav_at")},
                                "exit_dt": dt, "ret": vnow / c["invested"] - 1.0, "evicted": 1})
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K * t["w"]
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size, cash + pt + sum(lv(l, dt) for l in legs)); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    # residual open legs at end
    for leg in legs:
        log.append({**{k: leg[k] for k in ("symbol", "entry_dt", "conv", "prio", "w", "invested", "nav_at")},
                    "exit_dt": cal[-1], "ret": leg["last_val"] / leg["invested"] - 1.0, "evicted": 0})
    d = pd.DataFrame(ns, columns=["date", "nav"]); nav = d["nav"]
    return pd.DataFrame(log), float(nav.iloc[-1])


# cs4 + meta setup (seed 42)
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

con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=SEED).get("run_id")
cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                   "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
cm = {}
for r in cvtr.itertuples():
    cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
cv = HERE / f"_an_s{SEED}.csv"
cvtr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False)
feat = M.features(cvtr.symbol.unique().tolist())
pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
con.close()

L, navf = prun_log(NavSim2(str(cv), date_lo="2020-01-01"), pm, cm)
L["entry_dt"] = pd.to_datetime(L["entry_dt"]); L["exit_dt"] = pd.to_datetime(L["exit_dt"])
L["yr"] = L["entry_dt"].dt.year
L["hold"] = (L["exit_dt"] - L["entry_dt"]).dt.days.clip(lower=1)
L["contrib"] = (L["invested"] / L["nav_at"]) * L["ret"]   # approx P&L share (fraction of NAV at entry)

print(f"=== COMBO CHAMPION profit structure (seed {SEED}, final NAV x{navf:.1f}) ===", flush=True)
print(f"filled legs={len(L)}  win%={100*(L.ret>0).mean():.1f}  avg_ret={100*L.ret.mean():.2f}%  "
      f"med_hold={L.hold.median():.0f}d  evicted={int(L.evicted.sum())} ({100*L.evicted.mean():.1f}%)", flush=True)
# concentration (Pareto by contrib)
Ls = L.sort_values("contrib", ascending=False).reset_index(drop=True); tot = L.contrib.sum()
for topn in [5, 10, 20, 50]:
    print(f"  top {topn:3d} legs = {100*Ls.contrib[:topn].sum()/tot:5.1f}% of total contrib", flush=True)
pos = L[L.contrib > 0].contrib.sum(); neg = L[L.contrib < 0].contrib.sum()
print(f"  gross +{pos:.2f} / {neg:.2f} = net {tot:.2f}  (winners {100*(L.contrib>0).mean():.0f}% of legs)", flush=True)
# by year
print("  --- by entry-year: legs | win% | sum_contrib | avg_ret", flush=True)
for y, g in L.groupby("yr"):
    print(f"    {y}: {len(g):3d} | {100*(g.ret>0).mean():4.0f}% | {g.contrib.sum():+.2f} | {100*g.ret.mean():+5.1f}%", flush=True)
# by conviction quartile
L["cq"] = pd.qcut(L["conv"], 4, labels=["Q1lo", "Q2", "Q3", "Q4hi"])
print("  --- by conviction (cs4) quartile: legs | win% | avg_ret | sum_contrib", flush=True)
for q, g in L.groupby("cq", observed=True):
    print(f"    {q}: {len(g):3d} | {100*(g.ret>0).mean():4.0f}% | {100*g.ret.mean():+5.1f}% | {g.contrib.sum():+.2f}", flush=True)
# eviction quality
ev = L[L.evicted == 1]; nat = L[L.evicted == 0]
print(f"  --- EVICTION: {len(ev)} evicted, avg_ret={100*ev.ret.mean():+.1f}% (win {100*(ev.ret>0).mean():.0f}%) "
      f"vs natural avg_ret={100*nat.ret.mean():+.1f}%", flush=True)
# hold-time buckets
L["hb"] = pd.cut(L["hold"], [0, 5, 10, 20, 40, 9999], labels=["<=5", "6-10", "11-20", "21-40", ">40"])
print("  --- by hold bucket: legs | avg_ret | sum_contrib", flush=True)
for h, g in L.groupby("hb", observed=True):
    print(f"    {h:>6s}: {len(g):3d} | {100*g.ret.mean():+5.1f}% | {g.contrib.sum():+.2f}", flush=True)
print("ANALYZE_DONE")
