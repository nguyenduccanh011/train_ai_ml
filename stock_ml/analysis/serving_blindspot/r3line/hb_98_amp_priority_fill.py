# -*- coding: utf-8 -*-
"""hb_98: AMPLITUDE-PRIORITY fill-order (clue A). Fill-order quyet dinh +-5-8% NAV (hb: K25 mean 27.01
max 28.46). Sim hien fill theo symbol/shuffle (ngau nhien). Uu tien fill entry CONVICTION cao (blended
entry score, AMP IC +0.206 = tin hieu manh nhat he) truoc khi het slot -> bat s* nhieu song hon.
KHONG phai sizing (van equal-weight nav/K); chi la THU TU chon khi slot khan. So: priority-DESC vs
shuffle-mean vs priority-ASC (control). Neu DESC>mean>ASC -> amplitude-priority THANG."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import psycopg2, pandas as pd, numpy as np
from nh_nav2 import NavSim2, shuffle_stats, FEE, S0, DATE_HI

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent
RID = "template/x2_struct_to-69338138"


def priority_run(sim, prio_map, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, desc=True):
    """Replica NavSim2.run nhung fill entries[d] theo prio (desc=cao truoc). prio_map=(sym,entry_date)->score."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = prio_map.get((t["symbol"], t["entry_date"]), 0.0)
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: t["prio"], reverse=desc)  # PRIORITY fill order
    sym_close, sym_idx, calendar = sim.sym_close, sim.sym_idx, sim.calendar

    def leg_value(leg, dt):
        s = leg["symbol"]; j = sym_idx[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        ratio = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sym_close[s][j] * ratio) / leg["p0"]; leg["last_val"] = v; return v

    cash = 1.0; pending = defaultdict(float); pend_total = 0.0; legs = []; exits = defaultdict(list); nav_series = []
    for di, dt in enumerate(calendar):
        cash += pending.pop(dt, 0.0); pend_total = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            if advance_fee is not None: cash += proceeds * (1.0 - advance_fee)
            elif settle_lag <= 0: cash += proceeds
            else:
                ri = di + settle_lag
                if ri < len(calendar): pending[calendar[ri]] += proceeds
                else: pending["NEVER"] += proceeds
            legs.remove(leg)
        pos = sum(leg_value(l, dt) for l in legs); nav_now = cash + pend_total + pos
        for t in entries.get(dt, ()):
            size = nav_now / K
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
                exit_eff = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"] / c0, ratio1=exit_eff / c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(leg_value(l, dt) for l in legs); nav_series.append((dt, cash + pend_total + pos))
    ns = pd.DataFrame(nav_series, columns=["date", "nav"]); ns["date"] = pd.to_datetime(ns["date"])
    return float(ns["nav"].iloc[-1])


def main():
    con = psycopg2.connect(**PG)
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(RID,))
    sig = pd.read_sql("select symbol,date,score from run_signals where run_id=%s and signal=1 and score is not null", con, params=(RID,))
    con.close()
    cv = HERE / "_k98_trades.csv"; tr.to_csv(cv, index=False)
    # prio = blended entry score at the buy-signal just <= entry_date (fill is next-bar)
    sig["dt"] = pd.to_datetime(sig["date"]); tr["dt"] = pd.to_datetime(tr["entry_date"])
    tr["edkey"] = tr["dt"].dt.strftime("%Y-%m-%d")
    prio_map = {}
    for s, g in sig.groupby("symbol"):
        g = g.sort_values("dt")
        tg = tr[tr.symbol == s].sort_values("dt")
        m = pd.merge_asof(tg[["dt", "edkey"]], g[["dt", "score"]], on="dt", direction="backward")
        for ed, sc in zip(m["edkey"], m["score"]):
            prio_map[(s, ed)] = float(sc) if pd.notna(sc) else 0.0
    cov = sum(1 for v in prio_map.values() if v != 0.0)
    print(f"trades={len(tr)} prio-mapped={cov}/{len(prio_map)}", flush=True)
    for K in (25, 20):
        sh = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=40)
        sim = NavSim2(str(cv), date_lo="2020-01-01")
        hi = priority_run(sim, prio_map, K=K, desc=True)
        lo = priority_run(sim, prio_map, K=K, desc=False)
        print(f"K={K}: shuffle mean=x{sh['mean']:.2f} [min {sh['min']:.2f}, max {sh['max']:.2f}] | "
              f"PRIO-hi=x{hi:.2f} ({(hi/sh['mean']-1)*100:+.1f}% vs mean) | PRIO-lo=x{lo:.2f} ({(lo/sh['mean']-1)*100:+.1f}%)", flush=True)
    print("HB_98_DONE", flush=True)


if __name__ == "__main__":
    main()
