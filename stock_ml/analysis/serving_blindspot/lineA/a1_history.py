"""Line A pre-flight: entry-gate history on the leaderboard + champion 2646 config dump."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()

print("=== distinct entry_gate values across templates (count, max id) ===")
cur.execute("SELECT engine_config->>'entry_gate' AS g, count(*), max(id) FROM strategy_templates "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 40")
for r in cur.fetchall():
    print(r)

print("\n=== best leaderboard run per gate (seed 42, not superseded) ===")
cur.execute("""
SELECT t.engine_config->>'entry_gate' AS gate,
       max(lr.composite_score) AS best_comp, count(*) AS n_runs
FROM leaderboard_runs lr
JOIN strategy_templates t ON t.name = lr.run_name
WHERE lr.run_seed = 42 AND lr.superseded = false
GROUP BY 1 ORDER BY 2 DESC NULLS LAST LIMIT 40
""")
for r in cur.fetchall():
    print(r)

print("\n=== champion 2646 ===")
cur.execute("SELECT id, name, engine_config FROM strategy_templates WHERE id=2646")
tid, name, ec = cur.fetchone()
if isinstance(ec, str):
    ec = json.loads(ec)
print(tid, name)
print(json.dumps(ec, indent=1, default=str))

print("\n=== champion 2646 leaderboard rows ===")
cur.execute("SELECT run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades "
            "FROM leaderboard_runs WHERE run_name=%s ORDER BY run_seed", (name,))
for r in cur.fetchall():
    print(r)
con.close()
