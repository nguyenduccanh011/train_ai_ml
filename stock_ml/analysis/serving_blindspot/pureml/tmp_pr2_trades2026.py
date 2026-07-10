import psycopg2

conn = psycopg2.connect(
    host="localhost", port=5433, dbname="stockml",
    user="stockml", password="stockml_dev"
)
cur = conn.cursor()

# 1. Tìm run_id của template 2810
print("=== 1. run_id của template 2810 ===")
cur.execute("SELECT run_id, template_id FROM leaderboard_runs WHERE template_id = 2810")
rows = cur.fetchall()
print(rows)

run_id = None
if rows:
    run_id = rows[0][0]
    print(f"run_id = {run_id}")
else:
    print("Không tìm thấy template 2810 trong leaderboard_runs")

# 4. Schema của run_trades
print("\n=== 4. Schema run_trades ===")
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='run_trades' ORDER BY ordinal_position")
cols = cur.fetchall()
for c in cols:
    print(c[0])

if run_id:
    # 5. MIN/MAX entry_date, COUNT
    print(f"\n=== 5. Stats run_id={run_id} ===")
    cur.execute("SELECT MIN(entry_date), MAX(entry_date), COUNT(*) FROM run_trades WHERE run_id = %s", (run_id,))
    print(cur.fetchone())

    # 2. Trades 2026
    print(f"\n=== 2. Trades >= 2026-01-01 (LIMIT 20) ===")
    cur.execute("""
        SELECT entry_date, exit_date, symbol, pnl_pct
        FROM run_trades
        WHERE run_id = %s AND entry_date >= '2026-01-01'
        LIMIT 20
    """, (run_id,))
    rows2 = cur.fetchall()
    for r in rows2:
        print(r)
    if not rows2:
        print("(no rows)")

    # 3. Trade mới nhất
    print(f"\n=== 3. Trades mới nhất (DESC LIMIT 10) ===")
    cur.execute("""
        SELECT entry_date, exit_date, symbol, pnl_pct
        FROM run_trades
        WHERE run_id = %s
        ORDER BY entry_date DESC
        LIMIT 10
    """, (run_id,))
    rows3 = cur.fetchall()
    for r in rows3:
        print(r)
    if not rows3:
        print("(no rows)")

cur.close()
conn.close()
