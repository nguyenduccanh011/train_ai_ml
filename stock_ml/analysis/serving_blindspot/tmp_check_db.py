import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("SELECT id, name FROM strategy_templates WHERE name LIKE '%wavestruct%' ORDER BY id DESC LIMIT 5")
print("templates:", cur.fetchall())
cur.execute(
    "SELECT run_name, run_seed, composite_score, total_pnl, pf, trades FROM leaderboard_runs "
    "WHERE run_name LIKE '%wavestruct%' AND superseded=false ORDER BY run_name, run_seed"
)
for r in cur.fetchall():
    print("lb:", r)
cur.execute("SELECT max(id) FROM strategy_templates")
print("max_template_id:", cur.fetchone())
con.close()
