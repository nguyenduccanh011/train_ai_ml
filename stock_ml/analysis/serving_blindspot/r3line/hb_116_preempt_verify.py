# -*- coding: utf-8 -*-
"""hb_116: VERIFY R2 prio-swap preemption (hb_115: +41% NAV mean). Per-seed detail + margin sweep.
Reject if driven by 1 seed. R2 = evict held lowest entry-meta-prio if new_prio - held_prio > margin."""
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
        cv = HERE / f"_k116_s{sd}.csv"; cvtr.to_csv(cv, index=False)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        seed_tr[sd] = M.build_tr(con, rid, feat); seed_cv[sd] = str(cv)
    con.close()
    seed_pm = {sd: M.meta_preds(seed_tr[sd], tgt='t_pnl') for sd in SEEDS}
    # base per-seed
    print("R2 prio-swap preemption — PER-SEED verify (NAV / CAGR%):", flush=True)
    base = {}
    for sd in SEEDS:
        f, c, d, n = P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], rule=None)
        base[sd] = (f, c)
    hdr = "  margin | " + " | ".join(f"s{sd}" for sd in SEEDS) + " | mean NAV  CAGR%  DD%  evict"
    print(hdr, flush=True)
    print(f"  base   | " + " | ".join(f"x{base[sd][0]:.1f}" for sd in SEEDS), flush=True)
    for mg in (0.02, 0.03, 0.05, 0.07, 0.10):
        per = {}; ddl = []; evl = []
        for sd in SEEDS:
            f, c, d, n = P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], rule="R2", margin=mg)
            per[sd] = (f, c); ddl.append(d); evl.append(n)
        navs = [per[sd][0] for sd in SEEDS]; cgs = [per[sd][1] for sd in SEEDS]
        wins = sum(per[sd][0] > base[sd][0] for sd in SEEDS)
        detail = " | ".join(f"x{per[sd][0]:.1f}" for sd in SEEDS)
        print(f"  {mg:.2f}   | {detail} | x{statistics.mean(navs):.1f}  {statistics.mean(cgs)*100:.1f}  {statistics.mean(ddl)*100:.1f}  {statistics.mean(evl):.0f}  [{wins}/3 win]", flush=True)
    print("HB_116_DONE", flush=True)


if __name__ == "__main__":
    main()
