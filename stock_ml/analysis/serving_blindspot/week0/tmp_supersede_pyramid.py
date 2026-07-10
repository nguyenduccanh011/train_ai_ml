import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()

# Before: list affected rows
cur.execute(
    "SELECT run_name, run_seed, composite_score, superseded FROM leaderboard_runs "
    "WHERE run_name LIKE 'w0_pyr%' ORDER BY run_name, run_seed"
)
rows = cur.fetchall()
print("BEFORE:")
for r in rows:
    print(" ", r)

cur.execute("UPDATE leaderboard_runs SET superseded = true WHERE run_name LIKE 'w0_pyr%' AND superseded = false")
print(f"updated: {cur.rowcount} rows")
con.commit()

# Verify: pyramid rows superseded, snr_10 + champion untouched, current top of active leaderboard
cur.execute("SELECT count(*) FROM leaderboard_runs WHERE run_name LIKE 'w0_pyr%' AND superseded = false")
print("pyramid rows still active:", cur.fetchone()[0])
cur.execute(
    "SELECT run_name, run_seed, composite_score, superseded FROM leaderboard_runs "
    "WHERE run_name IN ('w0_snr_10', 'n2_2643_wavestruct_la05_lamp02') ORDER BY run_name, run_seed"
)
print("KEPT ACTIVE:")
for r in cur.fetchall():
    print(" ", r)
cur.execute(
    "SELECT run_name, run_seed, composite_score FROM leaderboard_runs WHERE superseded = false "
    "ORDER BY composite_score DESC NULLS LAST LIMIT 5"
)
print("TOP-5 active leaderboard now:")
for r in cur.fetchall():
    print(" ", r)
con.close()
