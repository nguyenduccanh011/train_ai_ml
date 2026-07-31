# -*- coding: utf-8 -*-
"""hb_128: PROSECUTE conviction-sizing (hb_127 +35pp @K25). Look-ahead risk = pscale (full-sample std).
Verify: (a) pscale FIXED 0.03 (no look-ahead) vs global-std -> if win holds, scale not driving it;
(b) per-seed; (c) per-YEAR NAV growth incl dead-year 2024 (sizing must not wreck dead-year)."""
from __future__ import annotations
import os, sys, warnings, statistics
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np
from nh_nav2 import NavSim2
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M
import hb_127_conviction_sizing as S

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]


def main():
    con = psycopg2.connect(**PG); feat = None
    seed_pm, seed_cv = {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k128_s{sd}.csv"; cvtr.to_csv(cv, index=False); seed_cv[sd] = str(cv)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        seed_pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    con.close()
    K = 25
    print("K25 conviction-sizing PROSECUTION (per-seed CAGR%):", flush=True)
    print("  config                    | s42  s21  s123 | mean DD%  win", flush=True)
    configs = [("alpha0 (equal-wt base)", 0.0, None), ("a0.6 pscale=global", 0.6, None),
               ("a0.6 pscale=0.03 FIXED", 0.6, 0.03), ("a1.0 pscale=global", 1.0, None),
               ("a1.0 pscale=0.03 FIXED", 1.0, 0.03)]
    base = {}
    for sd in SEEDS:
        base[sd] = S.prun_sized(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], K=K, alpha=0.0)[0]
    for name, al, ps in configs:
        cgs, dds, navs = [], [], []
        for sd in SEEDS:
            f, c, d = S.prun_sized(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], K=K, alpha=al, pscale=ps)
            cgs.append(c); dds.append(d); navs.append(f)
        w = sum(navs[i] > list(base.values())[i] for i in range(3))
        print(f"  {name:25s} | {cgs[0]*100:4.0f} {cgs[1]*100:4.0f} {cgs[2]*100:4.0f} | {statistics.mean(dds)*100:5.1f}  {w}/3", flush=True)
    # per-year NAV growth (seed42, a0.6 fixed vs base) — dead-year check
    print("\nPer-YEAR NAV growth ratio seed42 (base equal-wt vs a0.6 fixed):", flush=True)
    for al, ps, lab in [(0.0, None, "base"), (0.6, 0.03, "a0.6fix")]:
        cv = seed_cv[42]
        yr_ratio = {}
        for y in range(2020, 2027):
            f0 = S.prun_sized(NavSim2(cv, date_lo=f"{y}-01-01"), seed_pm[42], K=K, alpha=al, pscale=ps)[0]
            f1 = S.prun_sized(NavSim2(cv, date_lo=f"{y+1}-01-01"), seed_pm[42], K=K, alpha=al, pscale=ps)[0] if y < 2026 else 1.0
            yr_ratio[y] = f0 / f1 if f1 else f0
        print(f"  {lab:8s} | " + " ".join(f"{y}:{(yr_ratio[y]-1)*100:+.0f}%" for y in range(2020, 2027)), flush=True)
    print("HB_128_DONE", flush=True)


if __name__ == "__main__":
    main()
