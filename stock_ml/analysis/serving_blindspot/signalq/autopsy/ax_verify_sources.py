"""Verify trade CSV sources vs DB run_trades before autopsy of exit_snr_extend."""
import pandas as pd, psycopg2

D = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/signalq"
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

sv = pd.read_csv(D + "/sv_snr08_s42_trades.csv")
ch = pd.read_csv(D + "/st_champ2646_s42_trades.csv")
print("sv csv:", len(sv), round(sv.pnl_pct.sum(), 4))
print("ch csv:", len(ch), round(ch.pnl_pct.sum(), 4))

con = psycopg2.connect(**PG)
db = pd.read_sql("SELECT symbol, entry_date::text AS entry_date, exit_date::text AS exit_date, pnl_pct "
                 "FROM run_trades WHERE run_id='template/n2_2643_wavestruct_la05_lamp02-32a8dfee'", con)
print("ch db:", len(db), round(db.pnl_pct.sum(), 4))
m = ch.merge(db, on=["symbol", "entry_date"], how="outer", indicator=True, suffixes=("_csv", "_db"))
print("merge:", m._merge.value_counts().to_dict())
both = m[m._merge == "both"]
print("pnl mismatch rows:", (abs(both.pnl_pct_csv - both.pnl_pct_db) > 1e-9).sum())

# 2730 current run_trades (seed 123) vs sv csv — expect DIFFERENT (sv is seed 42)
db2 = pd.read_sql("SELECT symbol, entry_date::text AS entry_date, pnl_pct "
                  "FROM run_trades WHERE run_id='template/xq_snr_t08_g27-32a8dfee'", con)
print("2730 db (seed123):", len(db2), round(db2.pnl_pct.sum(), 4))
con.close()
