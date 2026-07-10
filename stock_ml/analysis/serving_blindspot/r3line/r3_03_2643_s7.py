# -*- coding: utf-8 -*-
"""r3_03: dump trades run s7 CO SAN cua t2643 n2_2515_volhold_rs8 (khong re-run
canonical; leaderboard chi co seed 7) + cham NAV 2 che do lam baseline tham chieu.
Luu y: cac diem r3_2643_mh* chay s42 — so chenh seed nho (gem s42 vs s555 ~0.4%)."""
import os
import sys
from pathlib import Path

os.environ["STOCK_DATA_DIR"] = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r2line" / "na_audit"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
csv_path = HERE / "r3_base_2643_s7_trades.csv"
if not csv_path.exists():
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute("SELECT run_id, composite_score, trades FROM leaderboard_runs "
                "WHERE run_name='n2_2515_volhold_rs8' AND run_seed=7")
    run_id, comp, ntr = cur.fetchone()
    tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                      "holding_days, pnl_pct, exit_reason from run_trades where run_id=%s",
                      con, params=(run_id,))
    con.close()
    tdf.to_csv(csv_path, index=False)
    print(f"dumped t2643 s7 run={run_id} comp={comp} -> {csv_path.name} ({len(tdf)})", flush=True)

for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"), ("2023-01-01", "f23")):
    sim = NavSim2(str(csv_path), date_lo=lo)
    a = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
    n = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, n=20)
    print(f"R3BASE t2643_s7 {tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} DDw {a['dd_worst']*100:.1f}% | "
          f"noadv x{n['mean']:.2f}±{n['sd']:.2f} DDw {n['dd_worst']*100:.1f}%", flush=True)
print("R3_03_DONE", flush=True)
