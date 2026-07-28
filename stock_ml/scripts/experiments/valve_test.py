"""DOWNSTREAM VALVE test (user challenge): NAV is insensitive to the entry score because the NAV sim
fills oversubscribed days by RANDOM SHUFFLE (nh_nav2.run line 81-83), wasting the score at the
portfolio level. Test SCORE-PRIORITY fill (sort entries by score DESC when slots scarce) vs the
default random fill, for the frontier and a DIFFERENT-ranking variant (cst_h10, rank-corr 0.24):
  - if priority > random => the score carries value the random sim wastes (valve exists),
  - if frontier & cst_h10 DIVERGE under priority (tied under random) => ranking QUALITY monetizes.
K=25 and K=16 (concentration amplifies the valve).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "F:/PROJECTS/hb2943_work")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from collections import defaultdict
import pandas as pd, psycopg2
import nh_nav2
from nh_nav2 import NavSim2, shuffle_stats, FEE

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
WORK = Path("F:/PROJECTS/hb2943_work/navboard"); WORK.mkdir(exist_ok=True)
RUNS = {"frontier": "template/x2_struct_to-69338138", "cst_h10": "template/cst_h10-aaf47721"}


def run_priority(sim, K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008):
    """NavSim2.run replica but entries filled by SCORE DESC (priority) instead of shuffle."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
    entries = defaultdict(list)
    for t in sim.trades:
        entries[t["entry_date"]].append(t)
    for d in entries:
        entries[d].sort(key=lambda t: -t.get("score", 0.0))  # PRIORITY: high score first
    sym_close, sym_idx, calendar = sim.sym_close, sim.sym_idx, sim.calendar

    def leg_value(leg, dt):
        s = leg["symbol"]; j = sym_idx[s].get(dt)
        if j is None:
            return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        ratio = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sym_close[s][j] * ratio) / leg["p0"]; leg["last_val"] = v; return v

    cash = 1.0; pending = defaultdict(float); pend_total = 0.0; legs = []; exits = defaultdict(list)
    nav_series = []
    for di, dt in enumerate(calendar):
        cash += pending.pop(dt, 0.0); pend_total = sum(pending.values())
        for leg in exits.get(dt, ()):
            proceeds = leg["invested"] * (1.0 + leg["net"])
            if advance_fee is not None:
                cash += proceeds * (1.0 - advance_fee)
            elif settle_lag <= 0:
                cash += proceeds
            else:
                ri = di + settle_lag
                if ri < len(calendar):
                    pending[calendar[ri]] += proceeds; pend_total += proceeds
                else:
                    pend_total += proceeds; pending["NEVER"] += proceeds
            legs.remove(leg)
        pos = sum(leg_value(l, dt) for l in legs); nav_now = cash + pend_total + pos
        for t in entries.get(dt, ()):
            size = nav_now / K
            if cash + 1e-12 >= size:
                s = t["symbol"]; c0, c1 = sym_close[s][t["i0"]], sym_close[s][t["i1"]]
                exit_eff = t["p0"] * (1.0 + t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"] / c0, ratio1=exit_eff / c1, last_val=size,
                           entry_date=dt, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(leg_value(l, dt) for l in legs); nav = cash + pend_total + pos
        nav_series.append((dt, nav))
    ns = pd.DataFrame(nav_series, columns=["date", "nav"]); ns["date"] = pd.to_datetime(ns["date"])
    nav = ns["nav"]; final = float(nav.iloc[-1])
    years = (ns["date"].iloc[-1] - ns["date"].iloc[0]).days / 365.25
    return dict(final=final, cagr=final ** (1 / years) - 1)


con = psycopg2.connect(**PG)
print("run       | K  | random-fill | priority-fill | Δ(prio-rand)")
for nm, rid in RUNS.items():
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
                     "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(rid,))
    sg = pd.read_sql("SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,))
    tr["esd"] = pd.to_datetime(tr["entry_signal_date"])
    sg["d"] = pd.to_datetime(sg["date"])
    sc = tr.merge(sg[["symbol", "d", "score"]], left_on=["symbol", "esd"], right_on=["symbol", "d"], how="left")
    look = {(r.symbol, str(pd.to_datetime(r.entry_date).date())): (r.score if pd.notna(r.score) else 0.0)
            for r in sc.itertuples()}
    csv = WORK / f"_valve_{nm}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)
    for K in (25, 16):
        sim = NavSim2(str(csv), date_lo="2020-01-01")
        for t in sim.trades:
            t["score"] = look.get((t["symbol"], str(pd.to_datetime(t["entry_date"]).date())), 0.0)
        rand = shuffle_stats(sim, K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
        rnav = rand["mean"]
        prio = run_priority(sim, K=K)
        print(f"{nm:9s} | {K} | {rnav:11.2f} | {prio['final']:13.2f} | {prio['final']-rnav:+.2f}", flush=True)
con.close()
print("VALVE_DONE")
