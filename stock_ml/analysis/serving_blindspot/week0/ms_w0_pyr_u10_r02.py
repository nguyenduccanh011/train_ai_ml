"""Week-0 multi-seed confirm for candidate w0_pyr_u10_r02 (template_id=2669).

Existing clone of champion 2646 with pyramid_add_units=1.0, pyramid_add_min_ret=0.02
(pyramid_add_bars=3 default). Seed 42 already run this session (composite=978.1).
This script runs run_template_experiment for the remaining seeds sequentially and
prints one ROW json per seed. Config-only -> reuses cached predictions.

Usage: python stock_ml/analysis/serving_blindspot/week0/ms_w0_pyr_u10_r02.py [seed ...]
       (default seeds: 7 99 555)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
NAME = "w0_pyr_u10_r02"
TEMPLATE_ID = 2669  # existing clone — do NOT re-clone
DEFAULT_SEEDS = [7, 99, 555]


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    seeds = [int(a) for a in sys.argv[1:]] or DEFAULT_SEEDS
    for seed in seeds:
        t0 = time.time()
        r = run_template_experiment(template_id=TEMPLATE_ID, seed=seed)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        rec = {"name": NAME, "template_id": TEMPLATE_ID, "seed": seed, "runtime_s": round(dt, 1),
               "composite": float(row[0]) if row[0] is not None else None,
               "total_pnl": float(row[1]) if row[1] is not None else None,
               "pf": float(row[2]) if row[2] is not None else None,
               "mdd": float(row[3]) if row[3] is not None else None,
               "trades": int(row[4]) if row[4] is not None else None}
        print("ROW " + json.dumps(rec), flush=True)
    print("MS_DONE")


if __name__ == "__main__":
    main()
