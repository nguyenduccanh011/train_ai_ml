# -*- coding: utf-8 -*-
"""dump engine_config template 2783 (gb_x08) + row leaderboard canonical."""
import json, psycopg2
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG); cur = con.cursor()
cur.execute("SELECT id,name,engine_config FROM strategy_templates WHERE id=2783")
r = cur.fetchone()
cfg = r[2] if not isinstance(r[2], str) else json.loads(r[2])
print("id=", r[0], "name=", r[1])
print(json.dumps(cfg, indent=1, sort_keys=True))
cur.execute("SELECT run_name,run_seed,composite_score,total_pnl,trades FROM leaderboard_runs WHERE run_name LIKE '%%gb_x08%%' AND superseded=false ORDER BY run_seed")
for row in cur.fetchall(): print(row)
con.close()
