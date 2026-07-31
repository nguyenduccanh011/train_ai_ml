# -*- coding: utf-8 -*-
"""hb_127: CONVICTION-SIZING (user cho phep sizing voi rang buoc TONG VON <=1, khong don bay, cong
bang). Size vi the = (nav/K)*mult, mult theo meta-prio (cap chan DD), tong <= nav (cash constraint).
Naive-tilt cu chet 2023-26 (variance-drag + slot-lockup) — gio co PREEMPTION (xoa slot-lockup) +
META-prio. Sweep tilt alpha, K25/K16, 3-seed. alpha=0 = equal-weight champion. Watch DD (variance-drag)."""
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


def prun_sized(sim, pm, K=16, alpha=0.0, cap=(0.4, 2.5), pscale=None, margin=0.01,
               advance_fee=0.0008, roundtrip=0.006):
    """Preempt R2 (m=margin) + conviction-sizing: size=(nav/K)*mult, mult=clip(1+alpha*prio/pscale).
    Total invested <= nav (cash floor). alpha=0 -> equal-weight."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
    if pscale is None:
        pv = [t["prio"] for t in sim.trades if t["prio"] > -9]; pscale = statistics.pstdev(pv) if len(pv) > 5 else 1.0
    lo, hi = cap
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
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"])

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs: cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            slot = nav_now / K                                 # a standard slot's worth of cash
            if cash + 1e-12 >= slot:                           # slot free (== prun_causal test) -> enter sized
                size = min(slot * mult(t["prio"]), cash)       # tilt size, cap by cash (total<=1)
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:                                         # full -> preempt lowest entry-prio held
                c = min(legs, key=lambda l: l["prio"])
                if t["prio"] - c["prio"] > margin:
                    vnow = lv(c, dt); cash += vnow * (1.0 - advance_fee); legs.remove(c)
                    if c in exits.get(c["exit_date"], ()): exits[c["exit_date"]].remove(c)
                    nav2 = cash + pt + sum(lv(l, dt) for l in legs); slot2 = nav2 / K
                    if cash + 1e-12 >= slot2:
                        size = min(slot2 * mult(t["prio"]), cash)
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())


def main():
    con = psycopg2.connect(**PG); feat = None
    seed_pm, seed_cv = {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k127_s{sd}.csv"; cvtr.to_csv(cv, index=False); seed_cv[sd] = str(cv)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        seed_pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    con.close()
    print("CONVICTION-SIZING (total<=1) + preempt, meta-prio tilt. 3-seed mean CAGR%/DD% [win vs a0]:", flush=True)
    for Kv in (25, 16):
        print(f"  --- K={Kv} ---  (alpha=0 = equal-weight champion)", flush=True)
        base = {sd: prun_sized(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], K=Kv, alpha=0.0)[0] for sd in SEEDS}
        for al in (0.0, 0.3, 0.6, 1.0, 1.5):
            cg, dd, w = [], [], 0
            for sd in SEEDS:
                f, c, d = prun_sized(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], K=Kv, alpha=al)
                cg.append(c); dd.append(d); w += (f > base[sd])
            print(f"    alpha {al:.1f} | CAGR {statistics.mean(cg)*100:5.1f}  DD {statistics.mean(dd)*100:5.1f}  [{w}/3 win]", flush=True)
    print("HB_127_DONE", flush=True)


if __name__ == "__main__":
    main()
