"""FAIR-BENCH re-evaluation under the OPEN VALVE (priority-fill K16). The whole session measured
signal variants under random-fill K25 (valve CLOSED) -> all null. But the valve reveals ranking
QUALITY (frontier +4.27, cst_h10 -1.96 under priority@K16). Re-score the 'null' signal variants under
priority-fill K16: a genuinely-better ranking would monetize MORE than the frontier (beat 35.93 /
beat frontier's +4.27 priority-gain). Re-runs each at seed42 (fresh trades+signals), then valve-scores.
Bar: frontier priority@K16 = 35.93 (rand 31.66, +4.27) ; cst_h10 = 26.37 (rand 28.33, -1.96).
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
from stock_ml.scripts.run_template import run_template_experiment
from stock_ml.scripts.experiments.valve_test import run_priority

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
WORK = Path("F:/PROJECTS/hb2943_work/navboard")
WORK.mkdir(exist_ok=True)
# variants whose config is still valid (feature sets di_ef/capasym were reverted -> excluded)
TPLS = {"frontier": 3185, "obj_q70": 3238, "e5_csent": 3255, "cap_l31": 3245}


def valve_score(con, rid, nm, K=16):
    tr = pd.read_sql(
        "SELECT symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date "
        "FROM run_trades WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
        "AND exit_price IS NOT NULL",
        con,
        params=(rid,),
    )
    sg = pd.read_sql(
        "SELECT symbol,date,score FROM run_signals WHERE run_id=%s", con, params=(rid,)
    )
    if len(tr) < 5:
        return None
    tr["esd"] = pd.to_datetime(tr["entry_signal_date"])
    sg["d"] = pd.to_datetime(sg["date"])
    sc = tr.merge(
        sg[["symbol", "d", "score"]],
        left_on=["symbol", "esd"],
        right_on=["symbol", "d"],
        how="left",
    )
    look = {
        (r.symbol, str(pd.to_datetime(r.entry_date).date())): (
            r.score if pd.notna(r.score) else 0.0
        )
        for r in sc.itertuples()
    }
    csv = WORK / f"_vb_{nm}.csv"
    tr[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(csv, index=False)
    sim = NavSim2(str(csv), date_lo="2020-01-01")
    for t in sim.trades:
        t["score"] = look.get((t["symbol"], str(pd.to_datetime(t["entry_date"]).date())), 0.0)
    rand = shuffle_stats(sim, K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
    prio = run_priority(sim, K=K)["final"]
    return rand, prio


rids = {}
for nm, tid in TPLS.items():
    r = run_template_experiment(template_id=tid, seed=42)
    rids[nm] = r.get("run_id")
    print(f"ran {nm} t{tid} -> {rids[nm]}", flush=True)

con = psycopg2.connect(**PG)
print("\n=== FAIR-BENCH: priority-fill K16 (VALVE OPEN) — bar: frontier prio 35.93 (+4.27) ===")
print("variant   | random K16 | priority K16 | Δ(prio-rand) | vs frontier-prio")
rows = []
for nm in TPLS:
    res = valve_score(con, rids[nm], nm, K=16)
    if res is None:
        print(f"{nm:9s} | (no trades / invalid)")
        continue
    rand, prio = res
    rows.append((nm, rand, prio))
fr_prio = next((p for n, r, p in rows if n == "frontier"), 35.93)
for nm, rand, prio in rows:
    print(
        f"{nm:9s} | {rand:10.2f} | {prio:12.2f} | {prio - rand:+.2f} | {prio - fr_prio:+.2f}",
        flush=True,
    )
con.close()
print("VALVE_BENCH_DONE")
