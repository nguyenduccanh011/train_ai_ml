import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2

from stock_ml.scripts.run_template import run_template_experiment

r = run_template_experiment(template_id=2646, seed=555)
con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute(
    "SELECT run_name, run_seed, composite_score, total_pnl, pf, trades, superseded FROM leaderboard_runs "
    "WHERE run_id = %s", (r.get("run_id"),)
)
row = cur.fetchone()
print("RESTORED:", row)
expected = (555, 730.4)
ok = row[1] == expected[0] and abs(row[2] - expected[1]) < 0.05
print("RESTORE_CHECK", "PASS" if ok else "FAIL — expected seed 555 / 730.4")
con.close()
