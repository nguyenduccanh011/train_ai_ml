import psycopg2
con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("SELECT run_name, run_seed, composite_score FROM leaderboard_runs WHERE run_name IN ('n2_2643_wavestruct_la05_lamp02','xq_snr_t08_g27','sx_g35','sx_w40','sx_w60') ORDER BY run_name")
for r in cur.fetchall(): print(r)
con.close()
