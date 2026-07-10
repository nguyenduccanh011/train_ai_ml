"""Week-0 follow-up TASK 2: SEED-123 AUDIT (out-of-batch seed, anti seed-mining).

Runs seed 123 on (a) control champion 2646, (b) candidate 2669 (w0_pyr_u10_r02),
(c) stack 2681 (w0_pyr_u10_r02_snr10). NO template is created or modified —
run_template_experiment only, rows read back from leaderboard_runs.

Usage: python stock_ml/analysis/serving_blindspot/week0/run_seed123_audit.py [template_id ...]
       (default: 2646 2669 2681)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEED = 123
NAMES = {2646: "CONTROL_2646", 2669: "w0_pyr_u10_r02", 2681: "w0_pyr_u10_r02_snr10"}
DEFAULT_TEMPLATES = [2646, 2669, 2681]


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    tmpls = [int(a) for a in sys.argv[1:]] or DEFAULT_TEMPLATES
    for tid in tmpls:
        t0 = time.time()
        r = run_template_experiment(template_id=tid, seed=SEED)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        rec = {"name": NAMES.get(tid, str(tid)), "template_id": tid, "seed": SEED,
               "runtime_s": round(dt, 1),
               "composite": float(row[0]) if row[0] is not None else None,
               "total_pnl": float(row[1]) if row[1] is not None else None,
               "pf": float(row[2]) if row[2] is not None else None,
               "mdd": float(row[3]) if row[3] is not None else None,
               "trades": int(row[4]) if row[4] is not None else None}
        print("ROW " + json.dumps(rec), flush=True)
    print("SEED123_AUDIT_DONE")


if __name__ == "__main__":
    main()
