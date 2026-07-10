"""Week-0 Task 2: CONTROL run — re-register champion 2646 seed 42."""
import sys, time, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

t0 = time.time()
r = run_template_experiment(template_id=2646, seed=42)
dt = time.time() - t0
run_id = r.get("run_id")
con = psycopg2.connect(**PG)
cur = con.cursor()
cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs WHERE run_id=%s", (run_id,))
row = cur.fetchone()
con.close()
out = dict(run_id=run_id, wall_s=round(dt, 1),
           composite=float(row[0]), total_pnl=float(row[1]), pf=float(row[2]),
           mdd=float(row[3]), trades=int(row[4]))
print("CONTROL_ROW " + json.dumps(out))
