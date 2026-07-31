# -*- coding: utf-8 -*-
"""hb_113: META-as-TRADE-FILTER @ K25 board-standard (khong phai fill-priority).
Meta IC 0.21 du bao trade-quality. O K25 priority NEUTRAL (book ~60% util) -> thay vao do
DUNG meta lam FILTER: bo lenh meta du bao xau -> nang chat trung binh -> co the cai thien
chinh con so K25 fair-comparison. Cutoff lay tu phan phoi PRED cua TRAIN (causal, khong leak
percentile trong test-year). Sweep drop-frac. So base K25 shuffle-mean. Neu duong = model manh
hon tren metric CONG BANG (khong chi operating-point K16)."""
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
from nh_nav2 import NavSim2, shuffle_stats
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M  # reuse features/build_tr (FCOLS v0)

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]; K = 25
FCOLS = M.FCOLS


def meta_keep(tr, drop_frac):
    """Walk-forward meta pnl-pred; cutoff = drop_frac-percentile cua TRAIN preds (causal).
    Return set of kept (symbol, edkey). drop_frac=0 -> keep all."""
    from lightgbm import LGBMRegressor
    if drop_frac <= 0:
        return set(zip(tr.symbol, tr.edkey))
    keep = set()
    for ty in range(2021, 2027):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=FCOLS + ['pnl'])
        test = tr[tr.yr == ty].dropna(subset=FCOLS)
        if len(train) < 100 or not len(test):
            # cannot filter (no model) -> keep all test trades of this year
            keep |= set(zip(test.symbol, test.edkey)); continue
        mdl = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30,
                            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                            verbose=-1, deterministic=True, force_col_wise=True, random_state=1)
        mdl.fit(train[FCOLS], train['pnl'])
        cut = np.quantile(mdl.predict(train[FCOLS]), drop_frac)   # cutoff from TRAIN dist
        tp = mdl.predict(test[FCOLS])
        for (_, row), p in zip(test.iterrows(), tp):
            if p >= cut:
                keep.add((row.symbol, row.edkey))
    return keep


def main():
    con = psycopg2.connect(**PG); feat = None
    seed_tr, seed_base = {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        cvtr['entry_date'] = pd.to_datetime(cvtr['entry_date'])
        cvtr['edkey'] = cvtr.entry_date.dt.strftime('%Y-%m-%d')
        seed_base[sd] = cvtr
        seed_tr[sd] = M.build_tr(con, rid, feat)
    con.close()
    print("META-as-FILTER @ K25 board-standard (shuffle-mean n=20, 3-seed):", flush=True)
    print("  drop%  | NAV(mean)  CAGR%   DD%    #trades", flush=True)
    for df in (0.0, 0.10, 0.20, 0.30):
        nav_s, dd_s, nt_s = [], [], []
        for sd in SEEDS:
            keep = meta_keep(seed_tr[sd], df)
            base = seed_base[sd]
            fcsv = HERE / f"_k113_s{sd}_d{int(df*100)}.csv"
            kept = base[[(s, e) in keep for s, e in zip(base.symbol, base.edkey)]].copy()
            kept[['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price']].to_csv(fcsv, index=False)
            st = shuffle_stats(NavSim2(str(fcsv), date_lo="2020-01-01"), K=K)
            nav_s.append(st['mean']); dd_s.append(st['dd_mean']); nt_s.append(len(kept))
        nav = statistics.mean(nav_s)
        cagr = nav ** (1 / ((pd.Timestamp('2026-07-01') - pd.Timestamp('2020-01-01')).days / 365.25)) - 1
        print(f"  {df*100:4.0f}%  | x{nav:6.2f}   {cagr*100:5.1f}  {statistics.mean(dd_s)*100:5.1f}  {statistics.mean(nt_s):6.0f}", flush=True)
    print("HB_113_DONE", flush=True)


if __name__ == "__main__":
    main()
