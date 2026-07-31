# -*- coding: utf-8 -*-
"""hb_135: PROSECUTE conviction-sizing = EXPOSURE artifact? (memory warns fair-null-exposure-artifact).
Conviction-size deploys MORE idle-cash (mult>1) -> higher avg exposure vs equal-weight (holds cash).
Gain from CONVICTION (better trades) or just EXPOSURE (leverage-to-1 in bull)? Control = FLAT-oversize
(all trades x const c, NO conviction tilt) matched to conviction's avg exposure. If flat~conviction @
same exposure -> ARTIFACT. Track avg invested fraction. K25, 3-seed."""
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


def prun(sim, pm, K=25, mode="equal", alpha=0.6, pscale=0.03, flat_c=1.0, cap=(0.4, 2.5),
         margin=0.01, advance_fee=0.0008, roundtrip=0.006):
    """mode: 'equal'(mult=1) | 'conviction'(prio tilt) | 'flat'(mult=flat_c const, NO tilt).
    Returns (final, cagr, dd, avg_exposure=mean invested/nav over days)."""
    s_new = (roundtrip - FEE) / 2.0; lo, hi = cap
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
    def mult(pr):
        if mode == "equal": return 1.0
        if mode == "flat": return flat_c
        return float(np.clip(1.0 + alpha * (pr / pscale), lo, hi)) if pr > -9 else 1.0
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
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"])

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []; expo = []
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
        pos = sum(lv(l, dt) for l in legs); nav_e = cash + pt + pos
        ns.append((dt, nav_e)); expo.append(pos / nav_e if nav_e > 0 else 0.0)
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min()), float(np.mean(expo))


def main():
    con = psycopg2.connect(**PG); feat = None; seed = {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cv = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                         "where run_id=%s and exit_date is not null", con, params=(rid,))
        p = HERE / f"_k135_s{sd}.csv"; cv.to_csv(p, index=False)
        if feat is None: feat = M.features(cv.symbol.unique().tolist())
        seed[sd] = (str(p), M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl'))
    con.close()
    def run(mode, **kw):
        cg, dd, ex = [], [], []
        for sd in SEEDS:
            f, c, d, e = prun(NavSim2(seed[sd][0], date_lo="2020-01-01"), seed[sd][1], K=25, mode=mode, **kw)
            cg.append(c); dd.append(d); ex.append(e)
        return statistics.mean(cg) * 100, statistics.mean(dd) * 100, statistics.mean(ex)
    print("K25 EXPOSURE prosecution (3-seed): CAGR% / DD% / avg-exposure", flush=True)
    c, d, e = run("equal"); print(f"  equal-weight (alpha0)      | {c:5.1f}  {d:5.1f}  expo {e:.2f}", flush=True)
    c, d, e = run("conviction", alpha=0.6); print(f"  CONVICTION alpha0.6        | {c:5.1f}  {d:5.1f}  expo {e:.2f}", flush=True)
    print("  --- FLAT-oversize controls (no conviction tilt, matched exposure) ---", flush=True)
    for fc in (1.3, 1.5, 1.7, 2.0):
        c, d, e = run("flat", flat_c=fc); print(f"  flat c={fc:.1f}               | {c:5.1f}  {d:5.1f}  expo {e:.2f}", flush=True)
    print("HB_135_DONE", flush=True)


if __name__ == "__main__":
    main()
