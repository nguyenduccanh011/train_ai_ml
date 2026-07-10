import psycopg2
con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("SELECT run_name, run_seed, composite_score, total_pnl, trades, artifact_trades_csv FROM leaderboard_runs WHERE run_name LIKE 'xq_%' OR run_name LIKE 'sx_%'")
for r in cur.fetchall(): print(r)
con.close()
