# -*- coding: utf-8 -*-
"""hb_121: R2 slot-preemption across K-continuum (25/20/16/12). Map new return/DD frontier + check
if preemption touches FAIR board K25 (assumed neutral, book ~60% util). If K25 improves -> fair win."""
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
    seed_pm, seed_cv = {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k121_s{sd}.csv"; cvtr.to_csv(cv, index=False); seed_cv[sd] = str(cv)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        seed_pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    con.close()
    print("R2 preemption (m0.01) across K — 3-seed mean [no-preempt -> preempt]:", flush=True)
    print("  K  | base CAGR%  preempt CAGR%   DD%    evict  win", flush=True)
    for Kv in (25, 20, 16, 12):
        P.K = Kv
        b_cg, p_cg, dds, evs, w = [], [], [], [], 0
        for sd in SEEDS:
            bf, bc, bd, _ = P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], rule=None)
            pf, pc, pd_, pe = P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], rule="R2", margin=0.01)
            b_cg.append(bc); p_cg.append(pc); dds.append(pd_); evs.append(pe); w += (pf > bf)
        print(f"  {Kv:2d} | {statistics.mean(b_cg)*100:8.1f}   {statistics.mean(p_cg)*100:10.1f}   {statistics.mean(dds)*100:5.1f}  {statistics.mean(evs):5.0f}  {w}/3", flush=True)
    print("HB_121_DONE", flush=True)


if __name__ == "__main__":
    main()
