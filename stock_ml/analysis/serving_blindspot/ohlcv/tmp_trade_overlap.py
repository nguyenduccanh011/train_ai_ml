# -*- coding: utf-8 -*-
"""Trade overlap: pv probe runs vs champion 2646 (seed 42). How much did the exit
head's behaviour actually change?"""
import sys
import psycopg2
import pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
CHAMP_RID = "template/n2_2643_wavestruct_la05_lamp02-32a8dfee"


def trades(run_id):
    con = psycopg2.connect(**PG)
    df = pd.read_sql(
        "SELECT symbol, entry_date, exit_date, pnl_pct, holding_days FROM run_trades "
        "WHERE run_id=%s", con, params=(run_id,))
    con.close()
    return df


champ = trades(CHAMP_RID)
for rid in sys.argv[1:]:
    p = trades(rid)
    k = ["symbol", "entry_date"]
    mg = p.merge(champ, on=k, suffixes=("_p", "_c"), how="outer", indicator=True)
    both = mg[mg["_merge"] == "both"]
    same_exit = (both["exit_date_p"] == both["exit_date_c"]).mean()
    print(f"{rid}: probe={len(p)} champ={len(champ)} shared-entries={len(both)} "
          f"probe-only={int((mg['_merge'] == 'left_only').sum())} "
          f"champ-only={int((mg['_merge'] == 'right_only').sum())} "
          f"same-exit-date-among-shared={same_exit:.3f}")
    d = (both["pnl_pct_p"] - both["pnl_pct_c"]).abs()
    print(f"  shared trades pnl |diff|>1e-9: {(d > 1e-9).sum()} "
          f"mean|diff|={d.mean():.5f}")
    # per-year entry composition of the differing trades
    for tag, sub in (("probe-only", mg[mg["_merge"] == "left_only"]),
                     ("champ-only", mg[mg["_merge"] == "right_only"])):
        if len(sub):
            yr = pd.to_datetime(sub["entry_date"]).dt.year.value_counts().sort_index()
            print(f"  {tag} by year: {yr.to_dict()}")
