# -*- coding: utf-8 -*-
"""hb_133: PYRAMIDING (add to winners) — orthogonal sizing-alpha memory flagged +63.7u/all-years,
deferred pending multi-lot; NOW unlocked (total<=1). Differs from conviction-sizing (entry-time
PREDICTION) — pyramid uses CONFIRMED momentum (position already up). On top of champion sizing stack
(K25, meta-prio conviction-size + preempt): when held leg up>=thr & not-yet-pyramided & cash avail,
add child tranche (buys at today close, rides to parent exit). Total<=1 (cash floor). Sweep thr/delta.
Compare champion (no pyramid)."""
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
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]


def prun_pyr(sim, pm, K=25, alpha=0.6, pscale=0.03, cap=(0.4, 2.5), margin=0.01,
             pyr_thr=None, pyr_delta=0.5, pyr_max=1, advance_fee=0.0008, roundtrip=0.006):
    """champion sizing+preempt, + optional pyramiding (pyr_thr=None disables). pyr_delta = tranche size
    as frac of slot; pyr_max = max adds per leg."""
    s_new = (roundtrip - FEE) / 2.0; lo, hi = cap
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
    def mult(pr): return float(np.clip(1.0 + alpha * (pr / pscale), lo, hi)) if pr > -9 else 1.0
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

    def mk(t, size):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"], npyr=0)

    def mk_child(par, dt, delta):
        s = par["symbol"]; j = si[s][dt]; i1 = par["i1"]
        cj = sc[s][j]; c1 = sc[s][i1]; xe = par["p0"] * (1.0 + par["net"])   # parent exit price
        net_c = (xe * (1.0 - s_new)) / (cj * (1.0 + s_new)) - 1.0 - FEE
        return dict(symbol=s, i0=j, i1=i1, invested=delta, net=net_c, p0=cj,
                    ratio0=1.0, ratio1=xe / c1, last_val=delta, exit_date=par["exit_date"], prio=par["prio"], npyr=9)

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs: cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            slot = nav_now / K
            if cash + 1e-12 >= slot:
                size = min(slot * mult(t["prio"]), cash)
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()): exits[c["exit_date"]].remove(c)
                    nav2 = cash + pt + sum(lv(l, dt) for l in legs); slot2 = nav2 / K
                    if cash + 1e-12 >= slot2:
                        size = min(slot2 * mult(t["prio"]), cash)
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        # PYRAMID: add to confirmed winners using idle cash
        if pyr_thr is not None and cash > 1e-9:
            slot = (cash + pt + sum(lv(l, dt) for l in legs)) / K
            for leg in list(legs):
                if leg["npyr"] >= pyr_max: continue
                si_s = si[leg["symbol"]].get(dt)
                if si_s is None or si_s >= leg["i1"]: continue         # need room before exit
                up = lv(leg, dt) / leg["invested"] - 1.0
                if up >= pyr_thr:
                    delta = min(slot * pyr_delta, cash)
                    if delta > 1e-9:
                        child = mk_child(leg, dt, delta); cash -= delta
                        legs.append(child); exits[child["exit_date"]].append(child); leg["npyr"] += 1
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


def main():
    con = psycopg2.connect(**PG); feat = None
    seed = {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k133_s{sd}.csv"; cvtr.to_csv(cv, index=False)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        seed[sd] = (str(cv), M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl'))
    con.close()
    print("PYRAMID on top champion sizing (K25, 3-seed mean CAGR%/DD% [win vs no-pyr]):", flush=True)
    base = {sd: prun_pyr(NavSim2(seed[sd][0], date_lo="2020-01-01"), seed[sd][1], K=25, pyr_thr=None)[0] for sd in SEEDS}
    print(f"  champion (no pyramid)      | CAGR {statistics.mean([prun_pyr(NavSim2(seed[sd][0],date_lo='2020-01-01'),seed[sd][1],K=25,pyr_thr=None)[1] for sd in SEEDS])*100:5.1f}", flush=True)
    for thr in (0.05, 0.10, 0.15):
        for dl in (0.5, 1.0):
            cg, dd, w = [], [], 0
            for sd in SEEDS:
                f, c, d = prun_pyr(NavSim2(seed[sd][0], date_lo="2020-01-01"), seed[sd][1], K=25, pyr_thr=thr, pyr_delta=dl)
                cg.append(c); dd.append(d); w += (f > base[sd])
            print(f"  pyr thr={thr:.2f} delta={dl:.1f}     | CAGR {statistics.mean(cg)*100:5.1f}  DD {statistics.mean(dd)*100:5.1f}  [{w}/3]", flush=True)
    print("HB_133_DONE", flush=True)


if __name__ == "__main__":
    main()
