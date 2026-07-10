import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute(
    "SELECT run_name, composite_score, total_pnl, trades FROM leaderboard_runs "
    "WHERE run_name LIKE 'r2b\\_%' ORDER BY run_name"
)
rows = cur.fetchall()
for r in rows:
    print(r)
print("count:", len(rows))
con.close()
