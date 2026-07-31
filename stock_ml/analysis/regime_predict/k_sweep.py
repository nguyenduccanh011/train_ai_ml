"""K/concentration sweep on the double-RS + struct-trail frontier (validate the parallel-work K-lever
on MY frontier). Fewer slots (lower K) = more concentration = more compounding on winners. Offline
capital lever (single-book, no multi-lot). K=25 is the board standard.
"""
import os, sys
import pandas as pd
os.environ.setdefault("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work")
sys.path.insert(0, os.environ["NH_NAV2_DIR"])
import psycopg2
from nh_nav2 import NavSim2, shuffle_stats

WORK = "F:/PROJECTS/hb2943_work/navboard"
TMP = f"{WORK}/_tmp_ksweep.csv"
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
RUNS = {  # seed -> run_id (frontier double-RS + struct-trail)
    42: "template/x2_struct_to-69338138",
    7:  "template/struct_to_s7-69338138",
    99: "template/struct_to_s99-69338138",
}
KS = [25, 20, 16, 12]
MEAS = dict(roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)


def score(run_id, K):
    con = psycopg2.connect(**PG)
    tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price FROM run_trades "
                     "WHERE run_id=%s AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
                     "AND exit_price IS NOT NULL", con, params=(run_id,))
    con.close()
    tr.to_csv(TMP, index=False)
    sim = NavSim2(str(TMP), date_lo="2020-01-01")
    yrs = (pd.Timestamp(sim.calendar[-1]) - pd.Timestamp(sim.calendar[0])).days / 365.25
    st = shuffle_stats(sim, K=K, roundtrip=MEAS["roundtrip"], settle_lag=MEAS["settle_lag"],
                       advance_fee=MEAS["advance_fee"], n=MEAS["n"])
    nav = st["mean"]; dd = st["dd_mean"]
    cagr = nav ** (1.0 / yrs) - 1.0
    return cagr, nav, dd


print(f"{'seed':>5} " + " ".join(f"K{K:>2}:CAGR/NAV/DD" for K in KS))
rows = {}
for seed, rid in RUNS.items():
    cells = []
    for K in KS:
        cagr, nav, dd = score(rid, K)
        rows.setdefault(K, []).append(cagr)
        cells.append(f"{cagr:.3f}/{nav:.1f}/{dd:.3f}")
    print(f"{seed:>5} " + "  ".join(cells))
print("\n=== mean CAGR across 3 seeds by K ===")
for K in KS:
    print(f"  K={K}: mean CAGR {sum(rows[K])/len(rows[K]):.4f}")
