# -*- coding: utf-8 -*-
"""exitmap step 0: dump gb_x08 trades tu Postgres run_trades + schema check."""
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap"

con = psycopg2.connect(**PG)
cols = pd.read_sql(
    "select column_name, data_type from information_schema.columns where table_name='run_trades' order by ordinal_position",
    con)
print(cols.to_string())

runs = pd.read_sql("select distinct run_id from run_trades where run_id ilike '%%gb_x08%%'", con)
print(runs.to_string())

tr = pd.read_sql("select * from run_trades where run_id = 'template/gb_x08-32a8dfee'", con)
print("rows:", len(tr))
print(tr.head(3).to_string())
tr.to_csv(f"{OUT}\\gbx08_trades.csv", index=False)
con.close()
