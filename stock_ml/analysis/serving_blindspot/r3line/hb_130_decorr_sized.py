# -*- coding: utf-8 -*-
"""hb_130: decorr-source (dip t110) RISK-SCALED via sizing (now enabled, total<=1). hb_126 dip failed
(DD -34%) under equal-weight — couldn't scale high-variance dip DOWN. Now: dip sized at FIXED small
beta*(nav/K), idle-fill only (prio=-5, cannot preempt momentum, momentum preempts dip); momentum
conviction-sized (meta-prio). Sweep beta. K25. Does small dip fill dead-year w/o DD blowout vs
struct-only-sized (105%)?"""
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
import hb_112_meta_target as M
import hb_125_decorr_source_probe as D

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent


def prun_mixed(sim, pm, dipkeys, K=25, alpha=0.6, pscale=0.03, beta=0.5, cap=(0.4, 2.5),
               margin=0.01, advance_fee=0.0008, roundtrip=0.006):
    """momentum: conviction size + normal preempt prio. dip (in dipkeys): fixed beta size, prio=-5
    (idle-fill, no preempt momentum). Total<=1 (cash floor)."""
    s_new = (roundtrip - FEE) / 2.0
    lo, hi = cap
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        isdip = (t["symbol"], t["entry_date"]) in dipkeys
        raw = pm.get((t["symbol"], t["entry_date"]), -9.9)
        t["isdip"] = isdip
        t["prio"] = -5.0 if isdip else raw                    # dip = idle-fill (low preempt prio)
        t["msz"] = beta if isdip else (float(np.clip(1.0 + alpha * (raw / pscale), lo, hi)) if raw > -9 else 1.0)
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
            elif legs and not t["isdip"]:                     # only momentum preempts
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
    con = psycopg2.connect(**PG)
    st = D.pull(con, D.STRUCT, 0); dp = D.pull(con, D.DIP, 1)
    con.close()
    syms = sorted(set(st.symbol) | set(dp.symbol)); feat = M.features(syms)
    tr_c = D.build(pd.concat([st, dp], ignore_index=True), feat); pm = D.meta(tr_c)
    comb = pd.concat([st, dp], ignore_index=True).sort_values('entry_date')
    cv_c = HERE / "_k130_comb.csv"; comb[['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price']].to_csv(cv_c, index=False)
    cv_s = HERE / "_k130_struct.csv"; st[['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price']].to_csv(cv_s, index=False)
    dipkeys = set(zip(dp.symbol, pd.to_datetime(dp.entry_date).dt.strftime('%Y-%m-%d')))
    K = 25
    print(f"K25 conviction-sized. struct={len(st)} dip={len(dp)}. dip=idle-fill fixed-beta size:", flush=True)
    # struct-only sized ref (no dip)
    fs, cs, ds = prun_mixed(NavSim2(str(cv_s), date_lo="2020-01-01"), pm, set(), K=K, beta=0.5)
    print(f"  struct-only sized      | CAGR {cs*100:5.1f}  DD {ds*100:5.1f}", flush=True)
    for beta in (0.3, 0.5, 0.8, 1.0):
        f, c, d = prun_mixed(NavSim2(str(cv_c), date_lo="2020-01-01"), pm, dipkeys, K=K, beta=beta)
        print(f"  struct+dip beta={beta:.1f}    | CAGR {c*100:5.1f}  DD {d*100:5.1f}", flush=True)
    print("HB_130_DONE", flush=True)


if __name__ == "__main__":
    main()
