# -*- coding: utf-8 -*-
"""hb_141: conviction-dependent MAX-HOLD (meta for EXIT, orthogonal to entry-sizing). Cut low-meta
positions early (free capital), hold high-meta long. Truncate trade exit at entry+H trading days
(H depends on meta), realize @close. Run through EXPOSURE-MATCHED sizing (fair). Test uniform H first,
then conviction-split. Compare champion 76.6%/-10.8 @expo0.57. Known lever maxhold (gems) x new sizing."""
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
import hb_119_hold_quality_preempt as H
import hb_136_exposure_matched as E

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]


def truncate(cvtr, clo, pm, H_low, H_high, split=None):
    """cap each trade hold at H trading days (H_low if meta<split else H_high). Recompute exit@close.
    split=None -> uniform H_low."""
    datemap = {}
    for s, g in clo.sort_values('date').groupby('symbol'):
        datemap[s] = g.reset_index(drop=True)
    out = []
    for _, t in cvtr.iterrows():
        s = t.symbol; g = datemap.get(s)
        ed = pd.to_datetime(t.entry_date); xd = pd.to_datetime(t.exit_date)
        edk = ed.strftime('%Y-%m-%d'); pr = pm.get((s, edk), np.nan)
        Hn = H_low if (split is None or (pd.notna(pr) and pr < split)) else H_high
        row = dict(symbol=s, entry_date=t.entry_date, exit_date=t.exit_date, entry_price=t.entry_price, exit_price=t.exit_price)
        if g is not None and Hn is not None:
            gi = g.index[g.date == ed]
            if len(gi):
                i0 = gi[0]; icap = min(i0 + Hn, len(g) - 1)
                capd = g.loc[icap, 'date']
                if capd < xd:
                    row['exit_date'] = capd.strftime('%Y-%m-%d'); row['exit_price'] = float(g.loc[icap, 'close'])
        out.append(row)
    return pd.DataFrame(out)


def main():
    con = psycopg2.connect(**PG); feat = None; clo = None; seed = {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cv = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                         "where run_id=%s and exit_date is not null", con, params=(rid,))
        if feat is None: feat = M.features(cv.symbol.unique().tolist()); clo = H.close_panel(cv.symbol.unique().tolist())
        pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
        seed[sd] = (cv, pm)
    con.close()

    def evalcfg(H_low, H_high, split, label):
        cg, dd, ex = [], [], []
        for sd in SEEDS:
            cv, pm = seed[sd]
            tc = truncate(cv, clo, pm, H_low, H_high, split)
            p = HERE / f"_k141_{label}_s{sd}.csv"; tc.to_csv(p, index=False)
            f, c, d, e = E.prun(NavSim2(str(p), date_lo="2020-01-01"), pm, K=25, mode="conviction", alpha=0.6, deploy=0.55)
            cg.append(c); dd.append(d); ex.append(e)
        print(f"  {label:20s} | CAGR {statistics.mean(cg)*100:5.1f}  DD {statistics.mean(dd)*100:5.1f}  expo {statistics.mean(ex):.3f}", flush=True)

    print("MAX-HOLD x sizing (fair exposure-matched). champion(no cap)=76.6/-10.8/0.57:", flush=True)
    evalcfg(None, None, None, "no-cap (champion)")
    for hh in (10, 15, 20, 30):
        evalcfg(hh, hh, None, f"uniform H={hh}")
    # conviction-split: low-meta short hold, high-meta long. split at meta ~0 (median-ish)
    evalcfg(10, 40, 0.02, "conv_lo10_hi40")
    evalcfg(8, 60, 0.02, "conv_lo8_hi60")
    print("HB_141_DONE", flush=True)


if __name__ == "__main__":
    main()
