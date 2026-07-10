# -*- coding: utf-8 -*-
"""pr2 buoc 0: dump config bi cao pm2_hs10_zx25 + t1058 (split_config, engine_config, slots)."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

con = psycopg2.connect(**PG)
cur = con.cursor()
for name in ("pm2_hs10_zx25",):
    cur.execute("SELECT id, name, split_config, engine_config, signal_threshold, "
                "entry_threshold, exit_threshold, model_mode, seed, created_at "
                "FROM strategy_templates WHERE name=%s", (name,))
    r = cur.fetchone()
    print(f"== template {r[0]} {r[1]} created={r[9]} ==")
    print("signal_threshold:", r[4], "entry_thr:", r[5], "exit_thr:", r[6], "model_mode:", r[7], "seed:", r[8])
    sc = r[2] if not isinstance(r[2], str) else json.loads(r[2])
    ec = r[3] if not isinstance(r[3], str) else json.loads(r[3])
    print("split_config:", json.dumps(sc, indent=1))
    print("engine_config:", json.dumps(ec, indent=1, sort_keys=True))

# runs of defendant
cur.execute("""SELECT run_id, run_name, seed, composite_score, total_pnl, trades
               FROM leaderboard_runs WHERE run_name LIKE %s ORDER BY seed""", ("%pm2_hs10_zx25%",))
for row in cur.fetchall():
    print("RUN", row)
con.close()
