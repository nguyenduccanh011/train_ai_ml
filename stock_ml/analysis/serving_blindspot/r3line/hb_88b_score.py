# -*- coding: utf-8 -*-
"""hb_88b: score cs10/cs20 entry-target runs directly from run_id (hb_88 prints lost in SQL echo)."""
from __future__ import annotations
import sys, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import psycopg2, duckdb, pandas as pd, numpy as np, scipy.stats as ss
from nh_nav2 import NavSim2, shuffle_stats

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
HERE = Path(__file__).parent
RUNS = {"en_cs10": "template/en_cs10-3d97fc76", "en_cs20": "template/en_cs20-cac9d493",
        "base_ftrs": "template/ft_rs-8bc8ec4e"}


def fwd_frames(syms):
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,close,high from ohlcv where timeframe='1D' and symbol in ({ph}) "
                   f"order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date'])
    out = []
    for s, g in px.groupby('symbol'):
        g = g.set_index('date').sort_index()
        g['fwd10'] = g['close'].shift(-10) / g['close'] - 1
        fmax = pd.concat([g['high'].shift(-k) for k in range(1, 21)], axis=1).max(axis=1)
        g['mfe20'] = fmax / g['close'] - 1
        out.append(g[['fwd10', 'mfe20']].assign(symbol=s).reset_index())
    return pd.concat(out)


def main():
    con = psycopg2.connect(**PG)
    syms = pd.read_sql("select distinct symbol from run_signals where run_id='template/ft_rs-8bc8ec4e'", con).symbol.tolist()
    fw = fwd_frames(syms)
    for name, rid in RUNS.items():
        sig = pd.read_sql("select symbol,date,signal,score from run_signals where run_id=%s and score is not null",
                          con, params=(rid,))
        sig['date'] = pd.to_datetime(sig['date']); ent = sig[sig.signal == 1]
        m = ent.merge(fw, on=['symbol', 'date']); m['year'] = m.date.dt.year
        def ic(tcol):
            mm = m.dropna(subset=['score', tcol])
            ov = ss.spearmanr(mm.score, mm[tcol]).correlation
            by = {int(y): round(ss.spearmanr(g.score, g[tcol]).correlation, 3) for y, g in mm.groupby('year') if len(g) > 30}
            return ov, by
        dov, dby = ic('fwd10'); aov, aby = ic('mfe20')
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days "
                         "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
        n18 = n25 = float('nan')
        if len(tr):
            cv = HERE / f"_k88b_{name}.csv"; tr.to_csv(cv, index=False)
            n18 = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=18, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
            n25 = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
        print(f"\n{name}: ntr={len(tr)} NAV@K18=x{n18:.1f} K25=x{n25:.1f}", flush=True)
        print(f"  DIR fwd10 IC={dov:+.3f} 2021={dby.get(2021)} 2022={dby.get(2022)} 2023={dby.get(2023)} 2024={dby.get(2024)} 2025={dby.get(2025)} 2026={dby.get(2026)}", flush=True)
        print(f"  AMP mfe20 IC={aov:+.3f} 2022={aby.get(2022)} 2023={aby.get(2023)} 2024={aby.get(2024)} 2026={aby.get(2026)}", flush=True)
    con.close(); print("\nHB_88B_DONE", flush=True)


if __name__ == "__main__":
    main()
