# -*- coding: utf-8 -*-
"""hb_114: SLOT-PREEMPTION oracle ceiling @ K16. Currently book full -> new signal DROPPED.
Preempt: khi full va new entry den, tim held leg co REMAINING-return thap nhat; neu new trade
return > remaining cua worst held -> evict (realize held @ today value -> cash) + take new.
ORACLE = biet truoc net (lookahead) -> do TRAN cua lever preemption. Neu >> base -> build causal
version (meta du bao remaining-potential). Neu nho -> lever preemption dong. So base-shuffle & meta."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np
from nh_nav2 import NavSim2, shuffle_stats, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]; K = 16


def prun_preempt(sim, pm, preempt=False, margin=0.0, advance_fee=0.0008, roundtrip=0.006):
    """pm = priority map (meta pred). preempt=True -> allow evicting worst held (ORACLE remaining
    return via net lookahead) when new trade beats it by margin. Returns (final, cagr, dd, n_evict)."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
    entries = defaultdict(list)
    for t in sim.trades: entries[t["entry_date"]].append(t)
    for d in entries: entries[d].sort(key=lambda t: t["prio"], reverse=True)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []; n_ev = 0
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs:
                cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = nav_now / K
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
            elif preempt and legs:
                # ORACLE remaining return of each held leg from today to its exit
                worst = None; worst_rem = 1e9
                for l in legs:
                    vnow = lv(l, dt)
                    if vnow <= 0: continue
                    rem = (l["invested"] * (1.0 + l["net"])) / vnow - 1.0   # frac gain now->exit
                    if rem < worst_rem: worst_rem = rem; worst = l
                if worst is not None and t["net"] > worst_rem + margin:
                    # evict worst: realize at today's value, free slot/cash
                    vnow = lv(worst, dt)
                    cash += vnow * (1.0 - advance_fee); legs.remove(worst)
                    if worst in exits.get(worst["exit_date"], ()): exits[worst["exit_date"]].remove(worst)
                    n_ev += 1
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K
                    if cash + 1e-12 >= size:
                        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
                        leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                                   ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"])
                        cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min()), n_ev


def main():
    con = psycopg2.connect(**PG); feat = None
    seed_tr, seed_cv = {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k114_s{sd}.csv"; cvtr.to_csv(cv, index=False)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        seed_tr[sd] = M.build_tr(con, rid, feat); seed_cv[sd] = str(cv)
    con.close()
    # meta priority map per seed
    seed_pm = {sd: M.meta_preds(seed_tr[sd], tgt='t_pnl') for sd in SEEDS}
    print("SLOT-PREEMPTION oracle ceiling @ K16 (3-seed mean):", flush=True)
    print("  variant                    | NAV     CAGR%   DD%    evict", flush=True)
    configs = [("meta no-preempt", False, 0.0), ("meta +preempt oracle m0", True, 0.0),
               ("meta +preempt oracle m.05", True, 0.05), ("meta +preempt oracle m.10", True, 0.10)]
    for name, pre, mg in configs:
        nv, cg, dd, ev = [], [], [], []
        for sd in SEEDS:
            f, c, d, n = prun_preempt(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], preempt=pre, margin=mg)
            nv.append(f); cg.append(c); dd.append(d); ev.append(n)
        print(f"  {name:26s} | x{statistics.mean(nv):5.2f}  {statistics.mean(cg)*100:5.1f}  {statistics.mean(dd)*100:5.1f}  {statistics.mean(ev):5.0f}", flush=True)
    print("HB_114_DONE", flush=True)


if __name__ == "__main__":
    main()
