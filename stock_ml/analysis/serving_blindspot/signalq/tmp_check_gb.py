import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute(
    "SELECT run_name, run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades "
    "FROM leaderboard_runs WHERE run_name LIKE 'gb_%' OR run_name LIKE 'gbp_%' ORDER BY run_name, run_seed"
)
rows = cur.fetchall()
for r in rows:
    print(r)
print("count:", len(rows))
con.close()
