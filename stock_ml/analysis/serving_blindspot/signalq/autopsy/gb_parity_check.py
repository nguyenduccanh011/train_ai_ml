"""Parity check: trades of a gb_ run vs candidate 2730 seed-42 trades CSV.
Usage: python gb_parity_check.py <run_id>
"""
import sys
from pathlib import Path

import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
REF = Path(__file__).resolve().parents[1] / "sv_snr08_s42_trades.csv"

run_id = sys.argv[1]
con = psycopg2.connect(**PG)
cols = pd.read_sql(
    "SELECT column_name FROM information_schema.columns "
    "WHERE table_name='run_trades' ORDER BY ordinal_position", con
).column_name.tolist()
print("run_trades cols:", cols)

new = pd.read_sql("SELECT * FROM run_trades WHERE run_id=%s", con, params=(run_id,))
con.close()
ref = pd.read_csv(REF, parse_dates=["entry_date", "exit_date"])

new["entry_date"] = pd.to_datetime(new["entry_date"])
new["exit_date"] = pd.to_datetime(new["exit_date"])
key = ["symbol", "entry_date"]
print(f"ref n={len(ref)} pnl={ref.pnl_pct.sum():.6f} | new n={len(new)} pnl={new.pnl_pct.sum():.6f}")

m = ref.merge(new, on=key, how="outer", suffixes=("_r", "_n"), indicator=True)
only = m[m._merge != "both"]
print(f"matched={int((m._merge=='both').sum())} ref_only={int((m._merge=='left_only').sum())} "
      f"new_only={int((m._merge=='right_only').sum())}")
b = m[m._merge == "both"]
dpnl = (b.pnl_pct_r - b.pnl_pct_n).abs()
dexit = (b.exit_date_r != b.exit_date_n).sum()
print(f"max |dpnl|={dpnl.max():.3e}  n(|dpnl|>1e-9)={int((dpnl>1e-9).sum())}  exit_date mismatches={int(dexit)}")
if len(only):
    print(only[["symbol", "entry_date", "_merge"]].to_string())
print("PARITY:", "PASS" if len(only) == 0 and dpnl.max() < 1e-9 and dexit == 0 else "FAIL")
