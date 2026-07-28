"""CONFIRM under the open valve: e5_csent (+0.87) and obj_q70 (+0.17) beat the frontier under
priority-fill K16 at seed42 — but margins are small (seed noise ~1.4). Re-run frontier + the 2
candidates at seeds {42,7,99}, valve-score priority@K16 each seed, check SIGN-CONSISTENCY 3/3.
Only a variant that beats frontier's priority@K16 in ALL 3 seeds is a real valve-revealed win.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "F:/PROJECTS/hb2943_work")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd, psycopg2
from stock_ml.scripts.run_template import run_template_experiment
from stock_ml.scripts.experiments.build_valve_bench import valve_score

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
TPLS = {"frontier": 3185, "e5_csent": 3255, "obj_q70": 3238}
SEEDS = [42, 7, 99]

con = psycopg2.connect(**PG)
# results[nm][seed] = priority@K16 NAV
res = {nm: {} for nm in TPLS}
for sd in SEEDS:
    for nm, tid in TPLS.items():
        try:
            r = run_template_experiment(template_id=tid, seed=sd); rid = r.get("run_id")
            vs = valve_score(con, rid, f"{nm}_s{sd}", K=16)
            if vs:
                rand, prio = vs; res[nm][sd] = prio
                print(f"{nm} seed{sd}: priority@K16={prio:.2f} (random {rand:.2f})", flush=True)
        except Exception as e:
            print(f"ERR {nm} seed{sd}: {type(e).__name__}: {str(e)[:200]}", flush=True)

print("\n=== VALVE-OPEN (priority K16) 3-seed — vs frontier, sign-consistency ===")
frm = res["frontier"]
for nm in TPLS:
    navs = [res[nm].get(s) for s in SEEDS if res[nm].get(s) is not None]
    if len(navs) < 3:
        print(f"{nm}: incomplete"); continue
    mean = sum(navs) / 3
    if nm == "frontier":
        print(f"{nm:9s}: prio@K16 navs={[f'{n:.2f}' for n in navs]} mean={mean:.3f}")
        continue
    deltas = [res[nm][s] - frm[s] for s in SEEDS if res[nm].get(s) and frm.get(s)]
    signs = "".join("+" if d > 0 else "-" for d in deltas)
    print(f"{nm:9s}: prio@K16 navs={[f'{n:.2f}' for n in navs]} mean={mean:.3f} | Δvs-front={[f'{d:+.2f}' for d in deltas]} signs={signs}")
con.close()
print("VALVE3_DONE")
