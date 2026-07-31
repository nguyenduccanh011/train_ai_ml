import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute(
    "SELECT run_name, run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades, superseded, run_id "
    "FROM leaderboard_runs WHERE run_name='n2_2783_noT_mh16'"
)
print("row moi:", cur.fetchall())
cur.execute("SELECT id, name FROM strategy_templates WHERE id=2936")
print("template:", cur.fetchone())
cur.execute(
    "SELECT COUNT(*) FROM leaderboard_runs WHERE superseded=false AND composite_score > 652.0"
)
print("so row song diem cao hon:", cur.fetchone()[0])
con.close()
