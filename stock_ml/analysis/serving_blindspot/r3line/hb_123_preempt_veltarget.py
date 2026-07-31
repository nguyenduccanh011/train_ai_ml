# -*- coding: utf-8 -*-
"""hb_123: does meta-prio TARGET sharpen preemption? t_vel (profit-velocity, rewards fast-profit ->
faster slot rotation, hop preemption) vs t_pnl (raw) vs t_win. R2 m0.01 @ K25 & K16, 3-seed."""
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
import hb_115_preempt_causal as P

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]


def main():
    con = psycopg2.connect(**PG); feat = None
    seed_tr, seed_cv = {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k123_s{sd}.csv"; cvtr.to_csv(cv, index=False); seed_cv[sd] = str(cv)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        seed_tr[sd] = M.build_tr(con, rid, feat)
    con.close()
    print("Preempt R2 m0.01 by meta-prio TARGET (3-seed mean CAGR%/DD%):", flush=True)
    for Kv in (25, 16):
        P.K = Kv
        print(f"  --- K={Kv} ---", flush=True)
        for tg in ('t_pnl', 't_vel', 't_win'):
            pm = {sd: M.meta_preds(seed_tr[sd], tgt=tg) for sd in SEEDS}
            cg, dd, nv = [], [], []
            for sd in SEEDS:
                f, c, d, e = P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), pm[sd], rule="R2", margin=0.01)
                cg.append(c); dd.append(d); nv.append(f)
            print(f"    {tg:6s} | NAV x{statistics.mean(nv):5.2f}  CAGR {statistics.mean(cg)*100:5.1f}  DD {statistics.mean(dd)*100:5.1f}", flush=True)
    print("HB_123_DONE", flush=True)


if __name__ == "__main__":
    main()
