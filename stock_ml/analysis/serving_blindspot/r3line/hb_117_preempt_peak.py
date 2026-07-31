# -*- coding: utf-8 -*-
"""hb_117: find low-margin peak of R2 preemption (hb_116 monotone up to m.02 x61.6/88%) + test
prio-source (META vs blended SCORE) robustness. If keeps climbing to m0 -> churn not costly / meta
ranking binding. If score-prio also wins -> lever robust to source."""
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


def run_grid(seed_cv, seed_prio, label):
    print(f"\n{label} (3-seed mean, [w/3 win vs base]):", flush=True)
    print("  margin | mean NAV  CAGR%  DD%   evict  win", flush=True)
    base = {}
    for sd in SEEDS:
        f, c, _, _ = P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_prio[sd], rule=None)
        base[sd] = f
    for mg in (0.0, 0.005, 0.01, 0.015, 0.02):
        nv, cg, dd, ev, w = [], [], [], [], 0
        for sd in SEEDS:
            f, c, d, n = P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_prio[sd], rule="R2", margin=mg)
            nv.append(f); cg.append(c); dd.append(d); ev.append(n); w += (f > base[sd])
        print(f"  {mg:.3f}  | x{statistics.mean(nv):5.1f}  {statistics.mean(cg)*100:5.1f}  {statistics.mean(dd)*100:5.1f}  {statistics.mean(ev):5.0f}  {w}/3", flush=True)


def main():
    con = psycopg2.connect(**PG); feat = None
    seed_tr, seed_cv = {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k117_s{sd}.csv"; cvtr.to_csv(cv, index=False)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        seed_tr[sd] = M.build_tr(con, rid, feat); seed_cv[sd] = str(cv)
    con.close()
    meta_pm = {sd: M.meta_preds(seed_tr[sd], tgt='t_pnl') for sd in SEEDS}
    # blended entry score as prio (from build_tr's 'score' col via edkey)
    score_pm = {}
    for sd in SEEDS:
        tr = seed_tr[sd]
        score_pm[sd] = {(r.symbol, r.edkey): (float(r.score) if pd.notna(r.score) else -9.9) for _, r in tr.iterrows()}
    run_grid(seed_cv, meta_pm, "PRIO=META (walk-forward pnl pred)")
    run_grid(seed_cv, score_pm, "PRIO=SCORE (blended entry score)")
    print("HB_117_DONE", flush=True)


if __name__ == "__main__":
    main()
