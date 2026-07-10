import psycopg2, csv
con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='run_trades' ORDER BY ordinal_position")
cols = [r[0] for r in cur.fetchall()]
print("run_trades cols:", cols)
want = [c for c in ["symbol","entry_date","entry_price","exit_date","exit_price","holding_days","pnl_pct","exit_reason"] if c in cols]
cur.execute(f"SELECT {','.join(want)} FROM run_trades WHERE run_id=%s ORDER BY entry_date, symbol", ('template/gb_x08-32a8dfee',))
rows = cur.fetchall()
with open("stock_ml/analysis/serving_blindspot/signalq/nicheloss/gb_x08_s42_trades.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(want); w.writerows(rows)
print("dumped", len(rows))
con.close()
