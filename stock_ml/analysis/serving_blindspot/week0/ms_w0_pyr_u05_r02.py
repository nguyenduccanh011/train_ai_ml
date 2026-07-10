"""Week-0 multi-seed check for candidate w0_pyr_u05_r02 (template_id=2666, clone of champion 2646).

Seed 42 already run this session (composite=858.9). This script runs seeds 7, 99, 555
sequentially via run_template_experiment (config-only -> cached predictions) and reads
each leaderboard row.

Usage: python stock_ml/analysis/serving_blindspot/week0/ms_w0_pyr_u05_r02.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
NAME = "w0_pyr_u05_r02"
TMPL = 2666  # existing clone — reuse, do NOT re-clone
SEEDS = [7, 99, 555]


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    results = []
    for seed in SEEDS:
        t0 = time.time()
        r = run_template_experiment(template_id=TMPL, seed=seed)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        rec = {"name": NAME, "template_id": TMPL, "seed": seed, "runtime_s": round(dt, 1),
               "composite": float(row[0]) if row[0] is not None else None,
               "total_pnl": float(row[1]) if row[1] is not None else None,
               "pf": float(row[2]) if row[2] is not None else None,
               "mdd": float(row[3]) if row[3] is not None else None,
               "trades": int(row[4]) if row[4] is not None else None}
        results.append(rec)
        print("ROW " + json.dumps(rec), flush=True)
    print("MS_DONE")


if __name__ == "__main__":
    main()
