# -*- coding: utf-8 -*-
"""BREAKTHROUGH #2: PYRAMID into winners (multi-lot). When a held trade is up +T%, ADD a tranche at that
price (cash-only, no new slot), exiting with the base leg -> amplify the winner-tail (top-10% = 62% pnl).
Causal (add when confirmed up). Combo preempt R2m01 + cs k1.5 + conv-skip 0.40. Sweep T + add-fraction.
3-seed [42,21,123]. WIN = beats champion 3/3. (Risk: capture 0.29 giveback -> add may buy pre-reversal.)"""
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


def prun(sim, pm, cm, pyr, pfrac, k_conv=KCONV, margin=MARGIN, skip=SKIP, advance_fee=0.0008, roundtrip=0.006):
    """pyr = set of (symbol, entry_date_str) legs that are PYRAMID adds (cash-only, no slot). pfrac scales
    their size vs a normal unit."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9); t["conv"] = cm.get((t["symbol"], t["entry_date"]), 0.5)
        t["pyr"] = (t["symbol"], str(pd.to_datetime(t["entry_date"]).date())) in pyr
    cv = [t["conv"] for t in sim.trades if not t["pyr"]]; mu = statistics.mean(cv); sd = statistics.pstdev(cv) or 1.0
    raw = []
    for t in sim.trades:
        z = (t["conv"] - mu) / sd; t["_w"] = min(max(1.0 + k_conv * z, 0.4), 1.8); raw.append(t["_w"])
    off = 1.0 - (statistics.mean([t["_w"] for t in sim.trades if not t["pyr"]]) if cv else 1.0)
    for t in sim.trades:
        t["w"] = max(0.3, t["_w"] + off)
    entries = defaultdict(list)
    for t in sim.trades:
        if t["pyr"]:
            entries[t["entry_date"]].append(t); continue
        if t["conv"] >= skip:
            entries[t["entry_date"]].append(t)
    for d in entries:                                   # regular first (higher prio), pyramids last
        entries[d].sort(key=lambda t: (t["pyr"], -t["prio"]))
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
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos; nopen = len(legs)
        for t in entries.get(dt, ()):
            if t["pyr"]:                                 # pyramid add: cash-only, no slot, size *= pfrac
                size = (nav_now / K) * t["w"] * pfrac
                if cash + 1e-12 >= size:
                    cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
                continue
            size = (nav_now / K) * t["w"]
            if cash + 1e-12 >= size and nopen < K:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg); nopen += 1
            elif legs and nopen >= K:
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
px = cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
CLA, BOA = {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l = g["close"], g["low"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    parts.append(g[["symbol", "date"] + CS4])
    gg = g.sort_values("date").reset_index(drop=True); CLA[s] = gg["close"].to_numpy(); BOA[s] = {d: i for i, d in enumerate(gg["date"])}
P = pd.concat(parts, ignore_index=True)
for col in CS4:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs4"] = P[[c + "_r" for c in CS4]].mean(axis=1)
CS = {(r.symbol, str(r.date.date())): (r.cs4 if pd.notna(r.cs4) else 0.5) for r in P.itertuples()}


def build_pyr(cvtr, T):
    """For each base trade reaching +T before exit, a pyramid row entering at the trigger close."""
    rows = []; pyrset = set(); cm2 = {}
    for r in cvtr.itertuples():
        be = BOA.get(r.symbol, {}).get(pd.to_datetime(r.entry_date)); bx = BOA.get(r.symbol, {}).get(pd.to_datetime(r.exit_date))
        if be is None or bx is None or bx <= be or r.entry_price <= 0:
            continue
        ca = CLA[r.symbol]; seg = ca[be:bx]           # bars strictly before exit
        trig = np.where(seg / r.entry_price - 1.0 >= T)[0]
        if not trig.size:
            continue
        tb = be + int(trig[0]); td = None
        for dkey, iv in BOA[r.symbol].items():
            if iv == tb:
                td = dkey; break
        if td is None or tb >= bx:
            continue
        rows.append({"symbol": r.symbol, "entry_date": str(td.date()), "exit_date": str(pd.to_datetime(r.exit_date).date()),
                     "entry_price": float(ca[tb]), "exit_price": float(r.exit_price)})
        pyrset.add((r.symbol, str(td.date()))); cm2[(r.symbol, str(td.date()))] = CS.get((r.symbol, str(pd.to_datetime(r.entry_signal_date).date())), 0.5)
    return pd.DataFrame(rows), pyrset, cm2


con = psycopg2.connect(**PG); feat = None
CONF = [("base", None, 0.0), ("pyr08_f05", 0.08, 0.5), ("pyr08_f10", 0.08, 1.0), ("pyr15_f10", 0.15, 1.0)]
res = {nm: {} for nm, _, _ in CONF}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades "
                       "where run_id=%s and exit_date is not null", con, params=(rid,))
    cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"])
    cm = {}
    for r in cvtr.itertuples():
        cm[(r.symbol, str(pd.to_datetime(r.entry_date).date()))] = CS.get((r.symbol, str(r.sigd.date())), 0.5)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    for nm, T, pf in CONF:
        base_csv = HERE / f"_py_{nm}_{sd}.csv"
        cols = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]
        if T is None:
            cvtr[cols].to_csv(base_csv, index=False); pyrset = set(); cmv = dict(cm)
        else:
            pdf, pyrset, cm2 = build_pyr(cvtr, T)
            pd.concat([cvtr[cols], pdf[cols]], ignore_index=True).to_csv(base_csv, index=False)
            cmv = {**cm, **cm2}
        res[nm][sd] = prun(NavSim2(str(base_csv), date_lo="2020-01-01"), pm, cmv, pyrset, pf)
    print(f"seed{sd} done", flush=True)
con.close()

print("\n=== PYRAMID-into-winners on combo, 3-seed, vs base ===", flush=True)
base = res["base"]
for nm, _, _ in CONF:
    navs = [res[nm][s][0] for s in SEEDS]; cg = statistics.mean([res[nm][s][1] for s in SEEDS]); dd = statistics.mean([res[nm][s][2] for s in SEEDS])
    d = [res[nm][s][0] - base[s][0] for s in SEEDS]; signs = "".join("+" if x > 0 else "-" for x in d)
    print(f"  {nm:10s}: NAV x{statistics.mean(navs):5.2f}  CAGR {cg*100:5.1f}%  DD {dd*100:5.1f}%  | Δ={[f'{x:+.1f}' for x in d]} {signs}", flush=True)
print("PYRAMID_DONE")
