# -*- coding: utf-8 -*-
"""pureml step 3: kiem tra runs + trades hien co cua t1058 (va 1053)."""
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)

runs = pd.read_sql(
    "select template_id, run_id, run_name, run_seed, composite_score, total_pnl, pf, "
    "mdd_per_symbol, trades, wr, avg_hold, superseded, generated_at "
    "from leaderboard_runs where template_id in (1058,1053,1029) order by template_id, run_seed",
    con)
print(runs.to_string(index=False))

for rid in runs["run_id"]:
    n = pd.read_sql(f"select count(*) c from run_trades where run_id='{rid}'", con).iloc[0]["c"]
    print(f"run_trades {rid}: {n}")
con.close()
print("PM03_DONE")
