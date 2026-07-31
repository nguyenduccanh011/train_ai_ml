# -*- coding: utf-8 -*-
"""hb_120: combine entry-prio (R2, good ranker) + hold-quality (fresh but noisy). R4 alone THUA R2.
R5 veto : candidate = lowest entry-prio held (R2); evict only if hold_rem_hat[cand] < new_prio too.
R6 blend: evict held with lowest (0.5*entry_prio + 0.5*hold_rem_hat). See if fresh info sharpens R2."""
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
import psycopg2, pandas as pd
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M
import hb_115_preempt_causal as P
import hb_119_hold_quality_preempt as H

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]; K = 16


def prun_combo(sim, pm, hm, mode="R5", margin=0.01, blend=0.5, advance_fee=0.0008, roundtrip=0.006):
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

    def mk(t, size):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"])

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []; n_ev = 0
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs: cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        dtk = dt if isinstance(dt, str) else pd.Timestamp(dt).strftime('%Y-%m-%d')
        for t in entries.get(dt, ()):
            size = nav_now / K
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                cand = None
                if mode == "R5":       # R2 select + hold veto
                    c = min(legs, key=lambda l: l["prio"])
                    rh = hm.get((c["symbol"], dtk), c["prio"])
                    if t["prio"] - c["prio"] > margin and rh < t["prio"]: cand = c
                elif mode == "R6":     # blend score
                    def bl(l): return blend * l["prio"] + (1 - blend) * hm.get((l["symbol"], dtk), l["prio"])
                    c = min(legs, key=bl)
                    if t["prio"] - bl(c) > margin: cand = c
                if cand is not None:
                    vnow = lv(cand, dt); cash += vnow * (1.0 - advance_fee); legs.remove(cand)
                    if cand in exits.get(cand["exit_date"], ()): exits[cand["exit_date"]].remove(cand)
                    n_ev += 1
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min()), n_ev


def main():
    con = psycopg2.connect(**PG); feat = None; clo = None
    seed_pm, seed_hm, seed_cv = {}, {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k120_s{sd}.csv"; cvtr.to_csv(cv, index=False); seed_cv[sd] = str(cv)
        if feat is None:
            feat = M.features(cvtr.symbol.unique().tolist()); clo = H.close_panel(cvtr.symbol.unique().tolist())
        seed_pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
        seed_hm[sd] = H.hold_preds(H.build_hold(con, rid, feat, clo))
    con.close()
    base = {sd: P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], rule=None)[0] for sd in SEEDS}
    print(f"K16 preempt combo (3-seed mean); base x{statistics.mean(base.values()):.2f}, R2 m0.01 = x64.0/89.3%:", flush=True)
    print("  variant             | NAV     CAGR%   DD%    evict  win", flush=True)
    def show(name, fn):
        rs = [fn(sd) for sd in SEEDS]; w = sum(rs[i][0] > list(base.values())[i] for i in range(3))
        print(f"  {name:19s} | x{statistics.mean([x[0] for x in rs]):5.2f}  {statistics.mean([x[1] for x in rs])*100:5.1f}  {statistics.mean([x[2] for x in rs])*100:5.1f}  {statistics.mean([x[3] for x in rs]):5.0f}  {w}/3", flush=True)
    for mg in (0.0, 0.01):
        show(f"R2 base m{mg:.2f}", lambda sd, mg=mg: P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], rule="R2", margin=mg))
        show(f"R5 veto m{mg:.2f}", lambda sd, mg=mg: prun_combo(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], seed_hm[sd], mode="R5", margin=mg))
        show(f"R6 blend m{mg:.2f}", lambda sd, mg=mg: prun_combo(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], seed_hm[sd], mode="R6", margin=mg))
    print("HB_120_DONE", flush=True)


if __name__ == "__main__":
    main()
