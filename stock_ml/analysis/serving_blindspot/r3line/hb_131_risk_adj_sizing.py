# -*- coding: utf-8 -*-
"""hb_131: RISK-ADJUSTED sizing (Kelly/vol-target). Champion sizes by meta-prio only (prio already
down-weights high-vol via atr feature). Test if EXPLICIT inverse-vol tilt adds value beyond that:
(a) prio-only (champion a0.6), (b) prio x inverse-vol, (c) pure inverse-vol risk-parity (ignore prio),
(d) prio / vol Sharpe-like. K25, 3-seed. size_map precomputed per-trade, preempt prio=meta."""
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


def prun_map(sim, pm, szmap, K=25, cap=(0.4, 2.5), margin=0.01, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0; lo, hi = cap
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
        t["msz"] = float(np.clip(szmap.get((t["symbol"], t["entry_date"]), 1.0), lo, hi))
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

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs: cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            slot = nav_now / K
            if cash + 1e-12 >= slot:
                size = min(slot * t["msz"], cash)
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()): exits[c["exit_date"]].remove(c)
                    nav2 = cash + pt + sum(lv(l, dt) for l in legs); slot2 = nav2 / K
                    if cash + 1e-12 >= slot2:
                        size = min(slot2 * t["msz"], cash)
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
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
        cv = HERE / f"_k131_s{sd}.csv"; cvtr.to_csv(cv, index=False)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        tr = M.build_tr(con, rid, feat); pm = M.meta_preds(tr, tgt='t_pnl')
        # per-trade risk (atr_pct), prio
        A, P0, S0 = 0.6, 0.0, 0.03
        rmed = tr['atr_pct'].median()
        maps = {'base': {}, 'prio_x_ivol': {}, 'ivol_only': {}, 'prio_div_vol': {}}
        for _, r in tr.iterrows():
            k = (r.symbol, r.edkey); p = pm.get(k, np.nan); atr = r.atr_pct
            base = np.clip(1.0 + A * (p / S0), 0.4, 2.5) if pd.notna(p) else 1.0
            ivol = (rmed / atr) if (pd.notna(atr) and atr > 0) else 1.0
            maps['base'][k] = base
            maps['prio_x_ivol'][k] = base * np.clip(ivol ** 0.5, 0.6, 1.6)
            maps['ivol_only'][k] = np.clip(ivol, 0.4, 2.5)
            sharpe = (p / atr) if (pd.notna(p) and pd.notna(atr) and atr > 0) else 0.0
            maps['prio_div_vol'][k] = np.clip(1.0 + A * sharpe * 0.03 / S0, 0.4, 2.5)
        seed[sd] = (str(cv), pm, maps)
    con.close()
    print("K25 RISK-ADJUSTED sizing schemes (3-seed mean CAGR%/DD%):", flush=True)
    for scheme in ['base', 'prio_x_ivol', 'ivol_only', 'prio_div_vol']:
        cg, dd = [], []
        for sd in SEEDS:
            cv, pm, maps = seed[sd]
            f, c, d = prun_map(NavSim2(cv, date_lo="2020-01-01"), pm, maps[scheme], K=25)
            cg.append(c); dd.append(d)
        print(f"  {scheme:14s} | CAGR {statistics.mean(cg)*100:5.1f}  DD {statistics.mean(dd)*100:5.1f}", flush=True)
    print("HB_131_DONE", flush=True)


if __name__ == "__main__":
    main()
