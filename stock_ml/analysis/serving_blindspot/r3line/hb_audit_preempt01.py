# -*- coding: utf-8 -*-
"""AUDIT: reproduce k16preempt (R2 margin 0.01) and k25preempt as 3-SEED MEANS (registered used seed42
only -> suspected inflated). Reuses hb_115.prun_causal + hb_112 meta-model (walk-forward OOS priority).
Prints 3-seed mean NAV/CAGR/DD for K16 m0.01 and K25 m0.01 to compare vs registered 61.76 / 33.83."""
from __future__ import annotations
import os, sys, statistics
from pathlib import Path
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd
from nh_nav2 import NavSim2
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M
import hb_115_preempt_causal as P

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SEEDS = [42, 21, 123]; MARGIN = 0.01

con = psycopg2.connect(**PG); feat = None
tr_c, cv_c = {}, {}
for sd in SEEDS:
    rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                       "where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / f"_audp_s{sd}.csv"; cvtr.to_csv(cv, index=False)
    if feat is None:
        feat = M.features(cvtr.symbol.unique().tolist())
    tr_c[sd] = M.build_tr(con, rid, feat); cv_c[sd] = str(cv)
con.close()
pm = {sd: M.meta_preds(tr_c[sd], tgt='t_pnl') for sd in SEEDS}

print(f"AUDIT R2 preempt margin {MARGIN} (3-seed mean) vs registered:", flush=True)
for K, reg in [(16, 61.76), (25, 33.83)]:
    nv, cg, dd = [], [], []
    for sd in SEEDS:
        P.K = K  # prun_causal reads module-global K
        f, c, d, ev = P.prun_causal(NavSim2(cv_c[sd], date_lo="2020-01-01"), pm[sd], rule="R2", margin=MARGIN)
        nv.append(f); cg.append(c); dd.append(d)
    mn = statistics.mean(nv)
    print(f"  K{K}: 3-seed NAV=x{mn:.2f} (seeds {[f'{x:.1f}' for x in nv]}) CAGR={statistics.mean(cg)*100:.1f}% "
          f"DD={statistics.mean(dd)*100:.1f}% | registered={reg} -> {reg/mn:.2f}x", flush=True)
print("AUDIT_PREEMPT01_DONE")
