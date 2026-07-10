import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute(
    "SELECT run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs "
    "WHERE run_name='w0_snr_10' ORDER BY run_seed"
)
for r in cur.fetchall():
    print("snr10:", r)
con.close()
