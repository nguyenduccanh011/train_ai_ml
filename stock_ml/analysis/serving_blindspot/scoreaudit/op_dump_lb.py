"""Leaderboard state for candidates + existing op-point runs; also list templates' strategy
and check prediction artifact availability."""
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()

print("== xl_/pv_/op_ leaderboard rows ==")
cur.execute("""SELECT run_name, run_seed, template_id, composite_score, total_pnl, pf, mdd_per_symbol, trades, superseded
FROM leaderboard_runs WHERE run_name LIKE 'xl_%' OR run_name LIKE 'pv_%' OR run_name LIKE 'op_%'
ORDER BY run_name, run_seed""")
for r in cur.fetchall():
    print(r)

print("\n== champion / gb_x08 canonical rows ==")
cur.execute("""SELECT run_name, run_seed, template_id, composite_score, pf, mdd_per_symbol, trades
FROM leaderboard_runs WHERE template_id IN (2646, 2783) AND superseded=false ORDER BY template_id, run_seed""")
for r in cur.fetchall():
    print(r)

print("\n== strategy of the 5 templates ==")
cur.execute("SELECT id, name, strategy FROM strategy_templates WHERE id IN (2646,2762,2766,2779,2783)")
for r in cur.fetchall():
    print(r)

print("\n== tables that might hold predictions ==")
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND (table_name ILIKE '%pred%' OR table_name ILIKE '%signal%' OR table_name ILIKE '%artifact%')")
print(cur.fetchall())
con.close()
