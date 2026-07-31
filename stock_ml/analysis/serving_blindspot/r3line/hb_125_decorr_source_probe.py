# -*- coding: utf-8 -*-
"""hb_125: DECORRELATED-SOURCE probe. dip_in_trend_v1 (t110) per-trade pnl 2024:+11.9% vs champion
struct_to +3.0% (dead-year momentum chet). UNION 2 trade-pools vao 1 so, meta-prio + preemption
ALLOCATE across sources (OHLCV-15 features + src-flag, walk-forward causal). So struct-only. Neu
combined > struct-only (nhat la dead-year) -> decorrelated source lap ho -> build proper. Cheap: dung
trades co san trong DB, khong re-run."""
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
import hb_112_meta_target as M
import hb_115_preempt_causal as P
import hb_119_hold_quality_preempt as H

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent
STRUCT = "template/x2_struct_to-69338138"; DIP = "template/dip_in_trend_v1-56623558"
FCO = H.FEATCOLS + ['src']   # OHLCV-15 + source flag (cross-source, no strategy score)


def pull(con, rid, src):
    t = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                    "where run_id=%s and exit_date is not null", con, params=(rid,))
    t['src'] = src; return t


def build(trades, feat):
    trades = trades.copy()
    trades['entry_date'] = pd.to_datetime(trades['entry_date']); trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    trades['pnl'] = trades.exit_price / trades.entry_price - 1.0; trades['yr'] = trades.entry_date.dt.year
    tr = trades.merge(feat.rename(columns={'date': 'entry_date'}), on=['symbol', 'entry_date'], how='left')
    tr['edkey'] = tr.entry_date.dt.strftime('%Y-%m-%d'); return tr


def meta(tr):
    from lightgbm import LGBMRegressor
    pm = {}
    for ty in range(2021, 2027):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=FCO + ['pnl']); test = tr[tr.yr == ty].dropna(subset=FCO)
        if len(train) < 100 or not len(test): continue
        mdl = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.7,
                            bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0, verbose=-1, deterministic=True, force_col_wise=True, random_state=1)
        mdl.fit(train[FCO], train['pnl'])
        for (_, row), p in zip(test.iterrows(), mdl.predict(test[FCO])): pm[(row.symbol, row.edkey)] = float(p)
    return pm


def navcurve(sim, pm, Kv, margin=0.01):
    P.K = Kv
    # reuse prun_causal but need per-year -> re-derive nav curve; call prun_causal for totals
    return P.prun_causal(sim, pm, rule="R2", margin=margin)


def main():
    con = psycopg2.connect(**PG)
    st = pull(con, STRUCT, 0); dp = pull(con, DIP, 1)
    con.close()
    syms = sorted(set(st.symbol) | set(dp.symbol)); feat = M.features(syms)
    tr_s = build(st, feat); tr_c = build(pd.concat([st, dp], ignore_index=True), feat)
    cv_s = HERE / "_k125_struct.csv"; st[['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price']].to_csv(cv_s, index=False)
    comb = pd.concat([st, dp], ignore_index=True).sort_values('entry_date')
    cv_c = HERE / "_k125_comb.csv"; comb[['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price']].to_csv(cv_c, index=False)
    pm_s = meta(tr_s); pm_c = meta(tr_c)
    print(f"struct trades={len(st)} dip trades={len(dp)} combined={len(comb)}", flush=True)
    print("preempt R2 m0.01 — struct-only vs +dip decorrelated source:", flush=True)
    print("  pool          | K25 CAGR/DD        | K16 CAGR/DD", flush=True)
    for label, cv, pm in [("struct-only", cv_s, pm_s), ("struct+dip", cv_c, pm_c)]:
        out = f"  {label:13s} |"
        for Kv in (25, 16):
            f, c, d, e = navcurve(NavSim2(str(cv), date_lo="2020-01-01"), pm, Kv)
            out += f" x{f:.1f} {c*100:.1f}%/{d*100:.1f}% (ev{e}) |"
        print(out, flush=True)
    print("HB_125_DONE", flush=True)


if __name__ == "__main__":
    main()
