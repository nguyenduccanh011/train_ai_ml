# -*- coding: utf-8 -*-
"""hb_134: sizing CAP sweep. lo = soft-filter (de-size predicted-worst, keeps velocity vs hard-filter
K25 which failed); hi = concentration ceiling. alpha=0.6, K25, 3-seed."""
from __future__ import annotations
import os, sys, warnings, statistics
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd
from nh_nav2 import NavSim2
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M
import hb_127_conviction_sizing as S

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]


def main():
    con = psycopg2.connect(**PG); feat = None; seed = {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cv = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                         "where run_id=%s and exit_date is not null", con, params=(rid,))
        p = HERE / f"_k134_s{sd}.csv"; cv.to_csv(p, index=False)
        if feat is None: feat = M.features(cv.symbol.unique().tolist())
        seed[sd] = (str(p), M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl'))
    con.close()
    print("K25 sizing CAP sweep (lo=soft-filter, hi=concentration), alpha=0.6, 3-seed:", flush=True)
    for lo, hi in [(0.4, 2.5), (0.25, 2.5), (0.1, 2.5), (0.0, 2.5), (0.1, 3.5), (0.0, 4.0), (0.0, 6.0)]:
        cg, dd = [], []
        for sd in SEEDS:
            f, c, d = S.prun_sized(NavSim2(seed[sd][0], date_lo="2020-01-01"), seed[sd][1], K=25, alpha=0.6, pscale=0.03, cap=(lo, hi))
            cg.append(c); dd.append(d)
        print(f"  lo={lo:.2f} hi={hi:.1f} | CAGR {statistics.mean(cg)*100:5.1f}  DD {statistics.mean(dd)*100:5.1f}", flush=True)
    print("HB_134_DONE", flush=True)


if __name__ == "__main__":
    main()
