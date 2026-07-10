import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("UPDATE leaderboard_runs SET superseded=true WHERE (run_name LIKE 'pm\\_%' OR run_name LIKE 'pm2\\_%' OR run_name LIKE 'pr2\\_%') AND superseded=false RETURNING run_name, composite_score")
for r in cur.fetchall():
    print("superseded:", r)
con.commit()
cur.execute("SELECT run_name, composite_score FROM leaderboard_runs WHERE superseded=false ORDER BY composite_score DESC LIMIT 5")
print("\ntop-5 song:")
for r in cur.fetchall():
    print(r)
con.close()
