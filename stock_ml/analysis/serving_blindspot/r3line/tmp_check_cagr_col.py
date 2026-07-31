import psycopg2

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute(
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name='leaderboard_runs' ORDER BY ordinal_position"
)
for r in cur.fetchall():
    print(r)
cur.execute(
    "SELECT run_name, composite_score, cagr FROM leaderboard_runs "
    "WHERE run_name IN ('n2_2783_noT_mh16','gb_x08','r3_mh16') OR run_name LIKE 'n2_2643_wavestruct%'"
)
try:
    print(cur.fetchall())
except Exception as e:
    print("no cagr col?", e)
con.close()
