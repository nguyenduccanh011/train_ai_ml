# -*- coding: utf-8 -*-
"""hb_126: dip as IDLE-FILL sleeve (fix hb_125 DD blowout). Under single-book equal-weight, decorr
high-variance dip must NOT concentrate. Rule: momentum(src0) meta-prio normal; dip(src1) prio = LOW
(-5) -> fills only leftover slots after momentum (idle capacity, more in dead-years), NEVER preempts;
momentum CAN preempt held dip. Test if dead-year idle-fill captures dip upside w/o DD blowout."""
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
import hb_112_meta_target as M
import hb_115_preempt_causal as P
import hb_119_hold_quality_preempt as H
import hb_125_decorr_source_probe as D

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent


def main():
    con = psycopg2.connect(**PG)
    st = D.pull(con, D.STRUCT, 0); dp = D.pull(con, D.DIP, 1)
    con.close()
    syms = sorted(set(st.symbol) | set(dp.symbol)); feat = M.features(syms)
    tr_s = D.build(st, feat); tr_c = D.build(pd.concat([st, dp], ignore_index=True), feat)
    pm_s = D.meta(tr_s); pm_c = D.meta(tr_c)
    # struct-only ref pool
    cv_s = HERE / "_k126_struct.csv"; st[['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price']].to_csv(cv_s, index=False)
    comb = pd.concat([st, dp], ignore_index=True).sort_values('entry_date')
    cv_c = HERE / "_k126_comb.csv"; comb[['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price']].to_csv(cv_c, index=False)
    # dip-idle prio map: momentum keeps meta pred, dip forced to -5 (idle-fill only)
    dipkeys = set(zip(dp.symbol, pd.to_datetime(dp.entry_date).dt.strftime('%Y-%m-%d')))
    pm_idle = {}
    for (s, e), v in pm_c.items():
        pm_idle[(s, e)] = -5.0 if (s, e) in dipkeys else v
    print(f"struct={len(st)} dip={len(dp)} combined={len(comb)}", flush=True)
    print("preempt R2 m0.01:", flush=True)
    print("  variant                 | K25 CAGR/DD        | K16 CAGR/DD", flush=True)
    configs = [("struct-only", cv_s, pm_s), ("struct+dip full-meta", cv_c, pm_c), ("struct+dip IDLE-fill", cv_c, pm_idle)]
    for label, cv, pm in configs:
        out = f"  {label:23s} |"
        for Kv in (25, 16):
            P.K = Kv
            f, c, d, e = P.prun_causal(NavSim2(str(cv), date_lo="2020-01-01"), pm, rule="R2", margin=0.01)
            out += f" x{f:5.1f} {c*100:4.1f}%/{d*100:5.1f}% |"
        print(out, flush=True)
    print("HB_126_DONE", flush=True)


if __name__ == "__main__":
    main()
