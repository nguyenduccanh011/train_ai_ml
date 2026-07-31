# -*- coding: utf-8 -*-
"""hb_124: enrich META (drives fill-priority + preemption) with MARKET-REGIME features (recompute
tu equal-weight index cua 61 syms: mkt vs MA50/MA200, mkt_ret20, mkt_vol, mkt_dd). Motivation: regime
tach shakeout-vs-top (AUC 0.78) + dead-year source. Do OOS-IC (pred vs realized pnl) + NAV preempt
@K25/K16. Neu tang -> meta amplify ca hai lever. Neu null -> meta near feature-ceiling that su."""
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
from scipy.stats import spearmanr
from nh_nav2 import NavSim2
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M
import hb_115_preempt_causal as P
import hb_119_hold_quality_preempt as H  # close_panel

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]
REGCOLS = ['mkt_ma50', 'mkt_ma200', 'mkt_ret20', 'mkt_vol20', 'mkt_dd']


def regime_feats(clo):
    """equal-weight market index from 61-sym close panel -> regime features by date."""
    piv = clo.pivot_table(index='date', columns='symbol', values='close').sort_index()
    ret = piv.pct_change().mean(axis=1)             # equal-weight daily market return
    idx = (1.0 + ret.fillna(0)).cumprod()
    df = pd.DataFrame({'date': idx.index})
    df['mkt_ma50'] = (idx / idx.rolling(50).mean() - 1.0).values
    df['mkt_ma200'] = (idx / idx.rolling(200).mean() - 1.0).values
    df['mkt_ret20'] = idx.pct_change(20).values
    df['mkt_vol20'] = ret.rolling(20).std().values
    df['mkt_dd'] = (idx / idx.cummax() - 1.0).values
    return df


def meta_preds_cols(tr, fcols, tgt='pnl'):
    from lightgbm import LGBMRegressor
    pm = {}; ics = []
    for ty in range(2021, 2027):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=fcols + [tgt]); test = tr[tr.yr == ty].dropna(subset=fcols + ['pnl'])
        if len(train) < 100 or not len(test): continue
        mdl = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.7,
                            bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0, verbose=-1, deterministic=True, force_col_wise=True, random_state=1)
        mdl.fit(train[fcols], train[tgt]); pred = mdl.predict(test[fcols])
        if len(test) > 5 and test['pnl'].std() > 0:
            ics.append(spearmanr(pred, test['pnl'].values)[0])
        for (_, row), p in zip(test.iterrows(), pred): pm[(row.symbol, row.edkey)] = float(p)
    return pm, (statistics.mean(ics) if ics else float('nan'))


def main():
    con = psycopg2.connect(**PG); feat = None; clo = None; reg = None
    seed_tr, seed_cv = {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k124_s{sd}.csv"; cvtr.to_csv(cv, index=False); seed_cv[sd] = str(cv)
        if feat is None:
            feat = M.features(cvtr.symbol.unique().tolist()); clo = H.close_panel(cvtr.symbol.unique().tolist()); reg = regime_feats(clo)
        tr = M.build_tr(con, rid, feat)
        tr = tr.merge(reg, left_on='entry_date', right_on='date', how='left', suffixes=('', '_r'))
        seed_tr[sd] = tr
    con.close()
    BASE = M.FCOLS; ENR = M.FCOLS + REGCOLS
    print("META regime-enrich: base(17) vs +regime(22). OOS-IC + preempt NAV/CAGR 3-seed:", flush=True)
    for label, fcols in [("base 17", BASE), ("+regime 22", ENR)]:
        pm = {}; ic = []
        for sd in SEEDS:
            p, i = meta_preds_cols(seed_tr[sd], fcols, tgt='pnl'); pm[sd] = p; ic.append(i)
        row = f"  {label:12s} | OOS-IC {statistics.mean(ic):+.3f} |"
        for Kv in (25, 16):
            P.K = Kv
            cg = []
            for sd in SEEDS:
                _, c, _, _ = P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), pm[sd], rule="R2", margin=0.01); cg.append(c)
            row += f" K{Kv} CAGR {statistics.mean(cg)*100:5.1f} |"
        print(row, flush=True)
    print("HB_124_DONE", flush=True)


if __name__ == "__main__":
    main()
