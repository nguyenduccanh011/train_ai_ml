import psycopg2
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG); cur = con.cursor()
cur.execute("SELECT id, name FROM strategy_templates WHERE id IN (2646,2730,2783) OR name IN ('gb_x08','a2_nopb','np_atmkt','np_pb02','np_pb03') ORDER BY id")
for r in cur.fetchall(): print("TMPL", r)
cur.execute("""SELECT run_id, run_name, run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades, superseded
               FROM leaderboard_runs WHERE run_name IN ('gb_x08','n2_2643_wavestruct_la05_lamp02','xq_snr_t08_g27','a2_nopb') AND run_seed=42""")
for r in cur.fetchall(): print("LB", r)
cur.execute("SELECT count(*) FROM run_trades WHERE run_id='template/gb_x08-32a8dfee'")
print("gb_x08 trades:", cur.fetchone())
con.close()
