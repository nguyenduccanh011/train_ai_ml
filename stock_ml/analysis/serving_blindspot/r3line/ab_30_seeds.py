# -*- coding: utf-8 -*-
"""ab_30: seed-luck check cho ung vien ab_modernclean_mh16 (= ab_noT, t2936:
gb_x08 + mh16 - trailing_struct 2 key). Chay seeds 7/99/555 tren CHINH t2936;
dump trades + NAV 2 che do x full/f22/f23. KHONG dung canonical."""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ["STOCK_DATA_DIR"] = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r2line" / "na_audit"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402

from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
TID = 2936  # ab_noT = ab_modernclean_mh16
SEEDS = [7, 99, 555]


def main():
    for seed in SEEDS:
        csv_path = HERE / f"ab_noT_s{seed}_trades.csv"
        if not csv_path.exists():
            r = run_template_experiment(template_id=TID, seed=seed)
            con = psycopg2.connect(**PG)
            cur = con.cursor()
            cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, "
                        "avg_hold FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
            row = cur.fetchone()
            print(f"AB30RUN ab_noT s{seed} comp={row[0]:.1f} pnl={row[1]:.1f} pf={row[2]:.2f} "
                  f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}", flush=True)
            tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                              "holding_days, pnl_pct, exit_reason from run_trades "
                              "where run_id=%s", con, params=(r.get("run_id"),))
            con.close()
            tdf.to_csv(csv_path, index=False)
            print(f"dumped {len(tdf)} -> {csv_path.name}", flush=True)
        for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"), ("2023-01-01", "f23")):
            sim = NavSim2(str(csv_path), date_lo=lo)
            a = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
            n = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, n=20)
            print(f"AB30NAV ab_noT s{seed} {tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} "
                  f"DDw {a['dd_worst']*100:.1f}% | noadv x{n['mean']:.2f}±{n['sd']:.2f} "
                  f"DDw {n['dd_worst']*100:.1f}%", flush=True)
    print("AB_30_DONE", flush=True)


if __name__ == "__main__":
    main()
