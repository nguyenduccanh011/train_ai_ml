# -*- coding: utf-8 -*-
"""r3_01: (1) verify nh_nav2 anchors 9/9; (2) dump trades s42 cua cac run canonical
CO SAN trong run_trades (2643, 2646, 2783 — KHONG re-run); (3) cham NAV baseline
CA HAI che do (advance 0.08% + no-advance) cho t2429/t2903/gb/2643/2646.
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ["STOCK_DATA_DIR"] = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
HERE = Path(__file__).parent
R2 = HERE.parent / "r2line"
sys.path.insert(0, str(R2 / "na_audit"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

# 1) verify anchors
print("== VERIFY nh_nav2 anchors ==", flush=True)
r = subprocess.run([sys.executable, str(R2 / "na_audit" / "nh_nav2.py")],
                   capture_output=True, text=True)
tail = r.stdout.strip().splitlines()[-1] if r.stdout else "(no output)"
print(tail, flush=True)
if "9/9" not in tail:
    print(r.stdout)
    sys.exit("ANCHOR FAIL")

# 2) dump trades cua run canonical co san
DUMPS = {  # run_name -> csv tag
    "n2_2515_volhold_rs8": "2643",
    "n2_2643_wavestruct_la05_lamp02": "2646",
    "gb_x08": "2783",
}
con = psycopg2.connect(**PG)
for run_name, tag in DUMPS.items():
    csv_path = HERE / f"r3_base_{tag}_s42_trades.csv"
    if csv_path.exists():
        print(f"exists: {csv_path.name}", flush=True)
        continue
    cur = con.cursor()
    cur.execute("SELECT run_id, composite_score, trades FROM leaderboard_runs "
                "WHERE run_name=%s AND run_seed=42 ORDER BY superseded ASC, created_at DESC",
                (run_name,))
    rows = cur.fetchall()
    if not rows:
        print(f"NO RUN: {run_name}", flush=True)
        continue
    run_id, comp, ntr = rows[0]
    tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                      "holding_days, pnl_pct, exit_reason from run_trades where run_id=%s",
                      con, params=(run_id,))
    tdf.to_csv(csv_path, index=False)
    print(f"dumped {run_name} s42 run={run_id} comp={comp} tr={ntr} -> {csv_path.name} "
          f"({len(tdf)} rows)", flush=True)
con.close()

# 3) NAV baseline ca 2 che do
CSVS = {
    "t2429_base": R2 / "navscan" / "nv_n2_consw20_conv04_vg_combo_hb_nbpbw_s42_trades.csv",
    "t2903_mh20": R2 / "navscan" / "nv_nv_2429_maxhold20_s42_trades.csv",
    "gb_2783": HERE / "r3_base_2783_s42_trades.csv",
    "t2643": HERE / "r3_base_2643_s42_trades.csv",
    "t2646_wavestruct": HERE / "r3_base_2646_s42_trades.csv",
}


def score(name, csv):
    if not Path(csv).exists():
        print(f"{name}: MISSING {csv}", flush=True)
        return
    for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"), ("2023-01-01", "f23")):
        sim = NavSim2(str(csv), date_lo=lo)
        a = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
        n = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, n=20)
        print(f"R3BASE {name} {tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} DDw {a['dd_worst']*100:.1f}% | "
              f"noadv x{n['mean']:.2f}±{n['sd']:.2f} DDw {n['dd_worst']*100:.1f}%", flush=True)


for name, csv in CSVS.items():
    score(name, csv)
print("R3_01_DONE", flush=True)
