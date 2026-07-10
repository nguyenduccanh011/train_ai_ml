"""Check champion 2646 leaderboard rows incl. superseded, for the seed-42 baseline."""
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()
cur.execute("SELECT run_id, run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades, superseded "
            "FROM leaderboard_runs WHERE run_name='n2_2643_wavestruct_la05_lamp02' ORDER BY run_seed")
for r in cur.fetchall():
    print(r)
print("--- prior a1_ runs, if any ---")
cur.execute("SELECT run_name, run_seed, composite_score FROM leaderboard_runs WHERE run_name LIKE 'a1_%' ORDER BY run_name")
for r in cur.fetchall():
    print(r)
con.close()
