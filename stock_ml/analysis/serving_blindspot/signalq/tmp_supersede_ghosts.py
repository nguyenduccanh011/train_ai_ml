import psycopg2

GHOSTS = ["sx_w60", "sx_w40", "st_dsb60_snr08", "pr_dsb50_snr08",
          "st_dsb60_snr10", "vx_dsb60", "vx_dsb50"]

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("UPDATE leaderboard_runs SET superseded=true WHERE run_name = ANY(%s) RETURNING run_name, composite_score", (GHOSTS,))
for r in cur.fetchall():
    print("superseded:", r)
con.commit()
cur.execute(
    "SELECT run_name, composite_score, superseded FROM leaderboard_runs "
    "WHERE superseded=false ORDER BY composite_score DESC LIMIT 5"
)
print("\ntop-5 song:")
for r in cur.fetchall():
    print(r)
con.close()
