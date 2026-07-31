# -*- coding: utf-8 -*-
"""hb_87: ROOT entry signal diagnostic. Exit da gan tran (cs_rs20 x32.2). Nay dao ENTRY raw.
Entry head (triple_barrier pt0.15/sl0.08 h30, fs=entry_recov_rs) = amplitude-ranker? Do raw
entry `score` IC vs:
  - fwd10/fwd20 RET (DIRECTION: score cao -> len that khong?)
  - mfe20 (AMPLITUDE: score cao -> bien do lon khong, bat ke huong?)
  - cross-sectional (rank trong ngay) fwd10 (outperform PEERS?)
theo NAM -> tim regime-fragility & khe cai thien. Neu direction IC lat dau 2024 nhung amplitude
IC duong moi nam -> confirm "biet SONG khong biet HUONG"; khe = target cross-sectional direction.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd, numpy as np, scipy.stats as ss

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
RID = "template/ft_rs-8bc8ec4e"  # base 3102 ft_rs entry


def fwd_frames(syms):
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,close,high from ohlcv where timeframe='1D' and symbol in ({ph}) "
                   f"order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date'])
    out = []
    for s, g in px.groupby('symbol'):
        g = g.set_index('date').sort_index()
        g['fwd10'] = g['close'].shift(-10) / g['close'] - 1
        g['fwd20'] = g['close'].shift(-20) / g['close'] - 1
        # MFE20 = max high over next 20 bars / close - 1 (amplitude, direction-agnostic upside)
        fmax = pd.concat([g['high'].shift(-k) for k in range(1, 21)], axis=1).max(axis=1)
        g['mfe20'] = fmax / g['close'] - 1
        out.append(g[['fwd10', 'fwd20', 'mfe20']].assign(symbol=s).reset_index())
    return pd.concat(out)


def ic_by_year(m, scol, tcol):
    m = m.dropna(subset=[scol, tcol])
    ov = ss.spearmanr(m[scol], m[tcol]).correlation
    by = {int(y): round(ss.spearmanr(g[scol], g[tcol]).correlation, 3)
          for y, g in m.groupby('year') if len(g) > 30}
    return ov, by


def main():
    con = psycopg2.connect(**PG)
    sig = pd.read_sql("select symbol,date,signal,score from run_signals where run_id=%s and score is not null",
                      con, params=(RID,))
    con.close()
    sig['date'] = pd.to_datetime(sig['date'])
    # entry-candidate rows only (signal==1 = buy-eligible) — raw entry ranking quality
    ent = sig[sig.signal == 1].copy()
    print(f"entry-candidate rows (signal=1): {len(ent)}  score[{ent.score.min():.3f},{ent.score.max():.3f}]", flush=True)
    fw = fwd_frames(sig.symbol.unique().tolist())
    m = ent.merge(fw, on=['symbol', 'date']); m['year'] = m.date.dt.year
    # cross-sectional demeaned fwd10 (outperform peers on that date)
    m['fwd10_cs'] = m['fwd10'] - m.groupby('date')['fwd10'].transform('mean')

    print("\n=== ENTRY raw `score` RankIC by year (higher score = model says BUY) ===", flush=True)
    for tcol, lab in [('fwd10', 'DIR fwd10'), ('fwd20', 'DIR fwd20'),
                      ('fwd10_cs', 'DIR-CS fwd10-peers'), ('mfe20', 'AMP mfe20')]:
        ov, by = ic_by_year(m, 'score', tcol)
        yrs = " ".join(f"{y}={by.get(y)}" for y in sorted(by))
        print(f"  {lab:20s} IC={ov:+.3f} | {yrs}", flush=True)
    print("\nHB_87_DONE", flush=True)


if __name__ == "__main__":
    main()
