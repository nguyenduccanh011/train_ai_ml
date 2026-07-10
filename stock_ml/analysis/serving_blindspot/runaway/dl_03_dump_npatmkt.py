# -*- coding: utf-8 -*-
"""Dump trades cua run np_atmkt tu Postgres run_trades -> CSV cho dl_03 sim."""
import psycopg2, csv

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                       user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='leaderboard_runs' ORDER BY ordinal_position")
print("leaderboard_runs cols:", [r[0] for r in cur.fetchall()])
cur.execute("SELECT run_id, run_name FROM leaderboard_runs WHERE run_name LIKE %s", ('%np_atmkt%',))
rows = cur.fetchall()
print("matches:", rows)
if len(rows) != 1:
    con.close()
    raise SystemExit("can chon run_id thu cong")
run_id = rows[0][0]

cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='run_trades' ORDER BY ordinal_position")
cols = [r[0] for r in cur.fetchall()]
print("run_trades cols:", cols)
want = [c for c in ["symbol", "entry_date", "entry_price", "exit_date", "exit_price",
                    "holding_days", "pnl_pct", "exit_reason"] if c in cols]
cur.execute(f"SELECT {','.join(want)} FROM run_trades WHERE run_id=%s ORDER BY entry_date, symbol", (run_id,))
rows = cur.fetchall()
out = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/runaway/dl_sleeveB_npatmkt.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(want)
    w.writerows(rows)
print("dumped", len(rows), "->", out)
con.close()
