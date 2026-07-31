# -*- coding: utf-8 -*-
"""hb_100: validate conviction-priority x K multi-seed (C). hb_98 seed42: PRIO-hi K20 +4.3%, K25 -2.2%
(K-flip). Re-run struct_to 3185 seeds 42/21/123 (bat trades+signals), priority-fill (conviction desc)
vs shuffle-mean tai K25/K20/K16. Neu PRIO-hi > mean ROBUST moi seed o K<=20 -> lever THAT (distinctive
strength model: concentration + conviction execution). Cung do CAGR/DD cua config priority-K."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats, FEE

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]


def priority_run(sim, prio_map, K, desc=True, roundtrip=0.006, settle_lag=2, advance_fee=0.0008):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = prio_map.get((t["symbol"], t["entry_date"]), 0.0)
    entries = defaultdict(list)
    for t in sim.trades: entries[t["entry_date"]].append(t)
    for d in entries: entries[d].sort(key=lambda t: t["prio"], reverse=desc)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    cash = 1.0; pending = defaultdict(float); legs = []; exits = defaultdict(list); nav_series = []
    for di, dt in enumerate(cal):
        cash += pending.pop(dt, 0.0); pt = sum(pending.values())
        for leg in exits.get(dt, ()):
            pr = leg["invested"] * (1.0 + leg["net"])
            if advance_fee is not None: cash += pr * (1.0 - advance_fee)
            else:
                ri = di + settle_lag
                if ri < len(cal): pending[cal[ri]] += pr
                else: pending["NEVER"] += pr
            legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        for t in entries.get(dt, ()):
            size = nav_now / K
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"]/c0, ratio1=xe/c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); nav_series.append((dt, cash + pt + pos))
    ns = pd.DataFrame(nav_series, columns=["date", "nav"]); ns["date"] = pd.to_datetime(ns["date"])
    nav = ns["nav"]; final = float(nav.iloc[-1]); yrs = (ns["date"].iloc[-1]-ns["date"].iloc[0]).days/365.25
    dd = float((nav/nav.cummax()-1).min())
    return final, final**(1/yrs)-1, dd


def build_prio(con, rid, tr):
    sig = pd.read_sql("select symbol,date,score from run_signals where run_id=%s and signal=1 and score is not null", con, params=(rid,))
    sig["dt"] = pd.to_datetime(sig["date"]); tr = tr.copy(); tr["dt"] = pd.to_datetime(tr["entry_date"]); tr["edkey"] = tr["dt"].dt.strftime("%Y-%m-%d")
    pm = {}
    for s, g in sig.groupby("symbol"):
        g = g.sort_values("dt"); tg = tr[tr.symbol == s].sort_values("dt")
        if not len(tg): continue
        m = pd.merge_asof(tg[["dt", "edkey"]], g[["dt", "score"]], on="dt", direction="backward")
        for ed, x in zip(m["edkey"], m["score"]): pm[(s, ed)] = float(x) if pd.notna(x) else 0.0
    return pm


def main():
    con = psycopg2.connect(**PG)
    data = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=3185, seed=sd); rid = r.get("run_id")
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k100_s{sd}.csv"; tr.to_csv(cv, index=False)
        pm = build_prio(con, rid, tr)
        data[sd] = (str(cv), pm)
        print(f"seed {sd}: {len(tr)} trades, prio {len(pm)}", flush=True)
    print("\n  K  | shuffle-mean(3seed) | PRIO-hi(3seed) delta | PRIO-lo | prio-hi CAGR/DD", flush=True)
    for K in (25, 20, 16):
        means, his, los, cagrs, dds = [], [], [], [], []
        for sd in SEEDS:
            cv, pm = data[sd]
            m = shuffle_stats(NavSim2(cv, date_lo="2020-01-01"), K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=30)["mean"]
            hi, cg, dd = priority_run(NavSim2(cv, date_lo="2020-01-01"), pm, K, desc=True)
            lo, _, _ = priority_run(NavSim2(cv, date_lo="2020-01-01"), pm, K, desc=False)
            means.append(m); his.append(hi); los.append(lo); cagrs.append(cg); dds.append(dd)
        mm, hh, ll = statistics.mean(means), statistics.mean(his), statistics.mean(los)
        print(f"  {K:2d} | mean x{mm:5.2f}          | hi x{hh:5.2f} ({(hh/mm-1)*100:+.1f}%) | lo x{ll:5.2f} ({(ll/mm-1)*100:+.1f}%) | CAGR {statistics.mean(cagrs)*100:.1f}% DD {statistics.mean(dds)*100:.1f}%", flush=True)
    con.close(); print("HB_100_DONE", flush=True)


if __name__ == "__main__":
    main()
