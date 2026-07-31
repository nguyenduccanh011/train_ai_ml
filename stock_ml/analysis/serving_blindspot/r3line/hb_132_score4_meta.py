# -*- coding: utf-8 -*-
"""hb_132: enrich META with score4 (amplitude head, raw signal manh nhat +0.206, co trong fold
parquets nhung chua persist/chua vao meta). Sizing khuech dai meta -> nang meta = nang ca stack.
Extract score4 (sym,date) tu fold dir moi nhat, merge vao meta features. OOS-IC + K25 sizing NAV:
base FCOLS(17) vs +score4(18). 3-seed. Neu tang -> lever moi; neu null -> meta that su tran."""
from __future__ import annotations
import os, sys, warnings, statistics, glob
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
import hb_131_risk_adj_sizing as R

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]


def latest_score4():
    """score4 from the most-recently-written tmpl_3185 fold dir (just refreshed by the seed run)."""
    dirs = glob.glob(str(REPO / "results/tmpl_3185_*/folds"))
    best = max(dirs, key=lambda d: os.path.getmtime(os.path.join(d, "test_2024.parquet")))
    parts = [pd.read_parquet(p)[['symbol', 'date', 'score4']] for p in glob.glob(best + "/test_*.parquet")]
    s4 = pd.concat(parts, ignore_index=True); s4['date'] = pd.to_datetime(s4['date']); return s4


def meta_ic(tr, fcols, tgt='pnl'):
    from lightgbm import LGBMRegressor
    pm = {}; ics = []
    for ty in range(2021, 2027):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=fcols + [tgt]); test = tr[tr.yr == ty].dropna(subset=fcols + ['pnl'])
        if len(train) < 100 or not len(test): continue
        mdl = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.7,
                            bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0, verbose=-1, deterministic=True, force_col_wise=True, random_state=1)
        mdl.fit(train[fcols], train[tgt]); pred = mdl.predict(test[fcols])
        if len(test) > 5 and test['pnl'].std() > 0: ics.append(spearmanr(pred, test['pnl'])[0])
        for (_, row), p in zip(test.iterrows(), pred): pm[(row.symbol, row.edkey)] = float(p)
    return pm, (statistics.mean(ics) if ics else float('nan'))


def main():
    con = psycopg2.connect(**PG); feat = None
    seed = {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        s4 = latest_score4()
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k132_s{sd}.csv"; cvtr.to_csv(cv, index=False)
        if feat is None: feat = M.features(cvtr.symbol.unique().tolist())
        tr = M.build_tr(con, rid, feat)
        tr = tr.merge(s4.rename(columns={'date': 'entry_date'}), on=['symbol', 'entry_date'], how='left')
        cov = tr['score4'].notna().mean()
        seed[sd] = (str(cv), tr, cov)
    con.close()
    print(f"score4 merge coverage (seed42): {seed[42][2]*100:.0f}%", flush=True)
    print("META base(17) vs +score4(18): OOS-IC + K25 sizing NAV (3-seed):", flush=True)
    for label, fcols in [("base 17", M.FCOLS), ("+score4 18", M.FCOLS + ['score4'])]:
        ics, cg, dd = [], [], []
        for sd in SEEDS:
            cv, tr, _ = seed[sd]
            pm, ic = meta_ic(tr, fcols, tgt='pnl'); ics.append(ic)
            # sizing szmap from this pm
            smap = {}
            for _, r in tr.iterrows():
                p = pm.get((r.symbol, r.edkey), np.nan)
                smap[(r.symbol, r.edkey)] = np.clip(1.0 + 0.6 * (p / 0.03), 0.4, 2.5) if pd.notna(p) else 1.0
            f, c, d = R.prun_map(NavSim2(cv, date_lo="2020-01-01"), pm, smap, K=25)
            cg.append(c); dd.append(d)
        print(f"  {label:11s} | OOS-IC {statistics.mean(ics):+.3f} | K25-size CAGR {statistics.mean(cg)*100:5.1f}  DD {statistics.mean(dd)*100:5.1f}", flush=True)
    print("HB_132_DONE", flush=True)


if __name__ == "__main__":
    main()
