import sys, psycopg2, pandas as pd
name = sys.argv[1]
con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
df = pd.read_sql(f"SELECT symbol, entry_date, exit_date, entry_price, exit_price, holding_days, pnl_pct, exit_reason "
                 f"FROM run_trades WHERE run_id='template/{name}-32a8dfee' ORDER BY entry_date, symbol", con)
con.close()
out = rf"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/signalq/autopsy/{name}_s42_trades.csv"
df.to_csv(out, index=False)
print(name, len(df), round(df.pnl_pct.sum(), 3))
