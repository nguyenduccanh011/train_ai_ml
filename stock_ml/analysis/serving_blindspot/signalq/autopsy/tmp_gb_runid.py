import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("SELECT run_name, run_seed, run_id FROM leaderboard_runs WHERE run_name LIKE 'gb_%' ORDER BY run_name, run_seed")
for r in cur.fetchall():
    print(r)
con.close()
