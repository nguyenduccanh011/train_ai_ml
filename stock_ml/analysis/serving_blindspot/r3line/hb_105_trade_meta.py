# -*- coding: utf-8 -*-
"""hb_105: TRADE-QUALITY META-MODEL (hb_104: oracle fill-priority +224% @K16, blended score chi +6.7%).
Train LightGBM du bao NET PNL moi lenh tu features luc-vao (causal), WALK-FORWARD (train tren lenh da
dong truoc test-year, predict lenh vao trong test-year). OOS prediction -> fill-priority @K16. So blended
score + shuffle-mean. Neu meta > score -> ranker tot hon -> khai thac them tran oracle."""
from __future__ import annotations
import os, sys, warnings
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import psycopg2, duckdb, pandas as pd, numpy as np
from nh_nav2 import NavSim2, shuffle_stats, FEE

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"; HERE = Path(__file__).parent
RID = "template/x2_struct_to-69338138"; K = 16


def features(syms):
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,open,high,low,close,volume from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date']); out = []
    for s, g in px.groupby('symbol'):
        g = g.sort_values('date').reset_index(drop=True); c = g['close']
        g['ret5'] = c.pct_change(5); g['ret20'] = c.pct_change(20); g['ret60'] = c.pct_change(60)
        g['vol20'] = c.pct_change().rolling(20).std()
        g['dist_h20'] = c / c.rolling(20).max() - 1; g['dist_h63'] = c / c.rolling(63).max() - 1
        g['dist_l20'] = c / c.rolling(20).min() - 1; g['dist_l63'] = c / c.rolling(63).min() - 1
        g['ma20r'] = c / c.rolling(20).mean() - 1; g['ma50r'] = c / c.rolling(50).mean() - 1
        tr = pd.concat([g['high'] - g['low'], (g['high'] - c.shift()).abs(), (g['low'] - c.shift()).abs()], axis=1).max(axis=1)
        g['atr_pct'] = tr.rolling(14).mean() / c
        g['updays10'] = (c.diff() > 0).rolling(10).sum()
        g['volr'] = g['volume'] / g['volume'].rolling(20).mean()
        out.append(g[['symbol', 'date', 'ret5', 'ret20', 'ret60', 'vol20', 'dist_h20', 'dist_h63',
                      'dist_l20', 'dist_l63', 'ma20r', 'ma50r', 'atr_pct', 'updays10', 'volr']])
    df = pd.concat(out)
    df['rs_mom20'] = df.groupby('date')['ret20'].rank(pct=True)  # cross-sectional RS
    df['rs_mom60'] = df.groupby('date')['ret60'].rank(pct=True)
    return df


FCOLS = ['score', 'exit_score', 'ret5', 'ret20', 'ret60', 'vol20', 'dist_h20', 'dist_h63',
         'dist_l20', 'dist_l63', 'ma20r', 'ma50r', 'atr_pct', 'updays10', 'volr', 'rs_mom20', 'rs_mom60']


def prun(sim, pm, desc=True, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
    entries = defaultdict(list)
    for t in sim.trades: entries[t["entry_date"]].append(t)
    for d in entries: entries[d].sort(key=lambda t: t["prio"], reverse=desc)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"]-leg["ratio0"])*(j-i0)/(i1-i0)
        v = leg["invested"]*(sc[s][j]*r)/leg["p0"]; leg["last_val"] = v; return v
    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()): cash += leg["invested"]*(1.0+leg["net"])*(1.0-advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash+pt+pos
        for t in entries.get(dt, ()):
            size = nav_now/K
            if cash+1e-12 >= size:
                s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"]*(1.0+t["net"])
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                           ratio0=t["p0"]/c0, ratio1=xe/c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash+pt+pos))
    return float(pd.DataFrame(ns, columns=["date", "nav"])["nav"].iloc[-1])


def main():
    from lightgbm import LGBMRegressor
    con = psycopg2.connect(**PG)
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(RID,))
    sig = pd.read_sql("select symbol,date,score,exit_score from run_signals where run_id=%s and signal=1 and score is not null", con, params=(RID,))
    con.close()
    cv = HERE / "_k105.csv"; tr.to_csv(cv, index=False)
    tr['entry_date'] = pd.to_datetime(tr['entry_date']); tr['exit_date'] = pd.to_datetime(tr['exit_date'])
    tr['pnl'] = tr.exit_price / tr.entry_price - 1.0; tr['yr'] = tr.entry_date.dt.year
    sig['date'] = pd.to_datetime(sig['date'])
    feat = features(tr.symbol.unique().tolist())
    # merge score/exit_score (asof backward per symbol) + ohlcv features (exact date, entry is a trading day)
    tr = tr.sort_values('entry_date')
    parts = []
    for s, tg in tr.groupby('symbol'):
        sg = sig[sig.symbol == s].sort_values('date')
        m = pd.merge_asof(tg.sort_values('entry_date'), sg[['date', 'score', 'exit_score']].rename(columns={'date': 'entry_date'}), on='entry_date', direction='backward')
        parts.append(m)
    tr = pd.concat(parts)
    tr = tr.merge(feat.rename(columns={'date': 'entry_date'}), on=['symbol', 'entry_date'], how='left')
    # WALK-FORWARD meta-model: train on trades EXITED before test-year start, predict trades entered in test-year
    tr['edkey'] = tr.entry_date.dt.strftime('%Y-%m-%d')
    pred_map = {}
    for ty in range(2021, 2027):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=FCOLS + ['pnl'])
        test = tr[tr.yr == ty].dropna(subset=FCOLS)
        if len(train) < 100 or not len(test): continue
        mdl = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30,
                            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                            verbose=-1, deterministic=True, force_col_wise=True, random_state=42)
        mdl.fit(train[FCOLS], train['pnl'])
        preds = mdl.predict(test[FCOLS])
        for (_, row), p in zip(test.iterrows(), preds): pred_map[(row.symbol, row.edkey)] = float(p)
    # score priority map
    score_map = {(r.symbol, r.edkey): (float(r.score) if pd.notna(r.score) else 0.0) for _, r in tr.iterrows()}
    base = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=40)["mean"]
    nav_score = prun(NavSim2(str(cv), date_lo="2020-01-01"), score_map)
    nav_meta = prun(NavSim2(str(cv), date_lo="2020-01-01"), pred_map)
    print(f"K={K} (2021+ trades scored by meta; 2020 unscored=shuffle-ish)", flush=True)
    print(f"  shuffle-mean   x{base:.2f}", flush=True)
    print(f"  blended-score  x{nav_score:.2f} ({(nav_score/base-1)*100:+.1f}%)", flush=True)
    print(f"  META-model     x{nav_meta:.2f} ({(nav_meta/base-1)*100:+.1f}%)  [pred {len(pred_map)} trades]", flush=True)
    print("HB_105_DONE", flush=True)


if __name__ == "__main__":
    main()
