"""Dump gb_x08 seed-42 trades + config from DB; sanity counts by exit_reason."""
import psycopg2, pandas as pd, json

con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

# columns of run_trades
cur = con.cursor()
cur.execute("""select column_name, data_type from information_schema.columns
               where table_name='run_trades' order by ordinal_position""")
print("run_trades columns:")
for r in cur.fetchall():
    print(" ", r[0], r[1])

# config JSON of the run (find config-ish columns)
cur.execute("""select column_name from information_schema.columns
               where table_name='leaderboard_runs'""")
lbcols = [r[0] for r in cur.fetchall()]
print("\nleaderboard_runs cols:", lbcols)

cur.execute("select * from leaderboard_runs where run_name='gb_x08'")
rows = cur.fetchall()
print("\nleaderboard rows gb_x08:", len(rows))
for r in rows:
    d = dict(zip(lbcols, r))
    for k, v in d.items():
        s = str(v)
        print(" ", k, "=", s[:400])

df = pd.read_sql("""select * from run_trades where run_id='template/gb_x08-32a8dfee'
                    order by entry_date, symbol""", con)
con.close()
out = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_s42_trades.csv"
df.to_csv(out, index=False)
print("\ntrades:", len(df), "pnl_sum:", round(df.pnl_pct.sum(), 3))
print("\nexit_reason counts:")
print(df.exit_reason.value_counts())
print("\ncolumns:", list(df.columns))
