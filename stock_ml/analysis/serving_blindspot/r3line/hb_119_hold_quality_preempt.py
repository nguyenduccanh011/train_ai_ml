# -*- coding: utf-8 -*-
"""hb_119: HOLD-QUALITY model for preemption (oracle hb_114 x1243 dung remaining-return lookahead).
Build causal predictor remaining-return cua vi the DANG MO tu features tai ngay d (khong lookahead):
train tren cac ngay-giu lich su (features@d -> exit_price/close_d - 1), walk-forward (trade da EXIT
truoc test-year), predict hold-days test-year -> hm[(sym,date)]. Rule R4: full-book + new signal ->
duoi held-leg co PREDICTED remaining-return thap nhat neu new_meta_prio - min_rem_hat > margin.
So R2 (entry-prio tinh, hb_116 89.3%). Neu R4 > R2 -> fresh hold-info bat them oracle gap."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd, numpy as np
from nh_nav2 import NavSim2, FEE
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
HERE = Path(__file__).parent; SEEDS = [42, 21, 123]
FEATCOLS = [c for c in M.FCOLS if c not in ('score', 'exit_score')]  # OHLCV-only (score is entry-time)
HCOLS = FEATCOLS + ['days_held', 'cur_ret']   # features for hold-quality model


def close_panel(syms):
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,close from ohlcv where timeframe='1D' and symbol in ({ph})", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date']); return px


def build_hold(con, rid, feat, clo):
    """Expand trades -> per-holding-day rows with features@d + target remaining-return."""
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                     "where run_id=%s and exit_date is not null", con, params=(rid,))
    tr['entry_date'] = pd.to_datetime(tr['entry_date']); tr['exit_date'] = pd.to_datetime(tr['exit_date'])
    fc = feat.merge(clo, on=['symbol', 'date'], how='left')
    parts = []
    for s, tg in tr.groupby('symbol'):
        fs = fc[fc.symbol == s].sort_values('date')
        for _, t in tg.iterrows():
            hold = fs[(fs.date >= t.entry_date) & (fs.date <= t.exit_date)].copy()
            if not len(hold): continue
            hold['days_held'] = (hold.date - t.entry_date).dt.days
            hold['cur_ret'] = hold['close'] / t.entry_price - 1.0
            hold['rem_ret'] = t.exit_price / hold['close'] - 1.0          # target
            hold['trade_exit'] = t.exit_date; hold['hyr'] = hold.date.dt.year
            parts.append(hold[['symbol', 'date', 'hyr', 'trade_exit', 'rem_ret'] + HCOLS])
    return pd.concat(parts, ignore_index=True)


def hold_preds(hold):
    from lightgbm import LGBMRegressor
    hm = {}
    for ty in range(2021, 2027):
        train = hold[hold.trade_exit < f"{ty}-01-01"].dropna(subset=HCOLS + ['rem_ret'])
        test = hold[hold.hyr == ty].dropna(subset=HCOLS)
        if len(train) < 200 or not len(test): continue
        mdl = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
                            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
                            verbose=-1, deterministic=True, force_col_wise=True, random_state=1)
        mdl.fit(train[HCOLS], train['rem_ret'])
        for (_, row), p in zip(test.iterrows(), mdl.predict(test[HCOLS])):
            hm[(row.symbol, row.date.strftime('%Y-%m-%d'))] = float(p)
    return hm


def prun_r4(sim, pm, hm, margin=0.01, advance_fee=0.0008, roundtrip=0.006):
    """R4: evict held with lowest PREDICTED remaining-return (hm), if new_prio - min_rem_hat > margin."""
    s_new = (roundtrip - FEE) / 2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"] * (1.0 - s_new)) / (t["e_raw"] * (1.0 + s_new)) - 1.0 - FEE
        t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
    entries = defaultdict(list)
    for t in sim.trades: entries[t["entry_date"]].append(t)
    for d in entries: entries[d].sort(key=lambda t: t["prio"], reverse=True)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar

    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"] + (leg["ratio1"] - leg["ratio0"]) * (j - i0) / (i1 - i0)
        v = leg["invested"] * (sc[s][j] * r) / leg["p0"]; leg["last_val"] = v; return v

    def mk(t, size):
        s = t["symbol"]; c0, c1 = sc[s][t["i0"]], sc[s][t["i1"]]; xe = t["p0"] * (1.0 + t["net"])
        return dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"],
                    ratio0=t["p0"] / c0, ratio1=xe / c1, last_val=size, exit_date=t["exit_date"], prio=t["prio"])

    cash = 1.0; pend = defaultdict(float); legs = []; exits = defaultdict(list); ns = []; n_ev = 0
    for di, dt in enumerate(cal):
        cash += pend.pop(dt, 0.0); pt = sum(pend.values())
        for leg in exits.get(dt, ()):
            if leg in legs: cash += leg["invested"] * (1.0 + leg["net"]) * (1.0 - advance_fee); legs.remove(leg)
        pos = sum(lv(l, dt) for l in legs); nav_now = cash + pt + pos
        dtk = dt if isinstance(dt, str) else pd.Timestamp(dt).strftime('%Y-%m-%d')
        for t in entries.get(dt, ()):
            size = nav_now / K
            if cash + 1e-12 >= size:
                cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
            elif legs:
                # candidate = held leg with lowest PREDICTED remaining-return (fresh, causal)
                cand = None; worst = 1e9
                for l in legs:
                    rh = hm.get((l["symbol"], dtk), l["prio"])   # fallback entry-prio if no pred
                    if rh < worst: worst = rh; cand = l
                if cand is not None and t["prio"] - worst > margin:
                    vnow = lv(cand, dt); cash += vnow * (1.0 - advance_fee); legs.remove(cand)
                    if cand in exits.get(cand["exit_date"], ()): exits[cand["exit_date"]].remove(cand)
                    n_ev += 1
                    size = (cash + pt + sum(lv(l, dt) for l in legs)) / K
                    if cash + 1e-12 >= size:
                        cash -= size; leg = mk(t, size); legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash + pt + pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    return final, final ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min()), n_ev


K = 16
import hb_115_preempt_causal as P


def main():
    con = psycopg2.connect(**PG); feat = None; clo = None
    seed_pm, seed_hm, seed_cv = {}, {}, {}
    for sd in SEEDS:
        rid = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                           "where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k119_s{sd}.csv"; cvtr.to_csv(cv, index=False); seed_cv[sd] = str(cv)
        if feat is None:
            feat = M.features(cvtr.symbol.unique().tolist()); clo = close_panel(cvtr.symbol.unique().tolist())
        seed_pm[sd] = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
        seed_hm[sd] = hold_preds(build_hold(con, rid, feat, clo))
    con.close()
    print("HOLD-QUALITY preempt (R4) vs entry-prio preempt (R2) @K16, 3-seed mean:", flush=True)
    print("  variant                 | NAV     CAGR%   DD%    evict  win", flush=True)
    # base
    base = {sd: P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], rule=None)[0] for sd in SEEDS}
    print(f"  base (no preempt)       | x{statistics.mean(base.values()):.2f}", flush=True)
    for mg in (0.0, 0.01, 0.02):
        # R2 ref
        r2 = [P.prun_causal(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], rule="R2", margin=mg) for sd in SEEDS]
        w2 = sum(r2[i][0] > list(base.values())[i] for i in range(3))
        print(f"  R2 entry-prio m{mg:.2f}     | x{statistics.mean([x[0] for x in r2]):5.2f}  {statistics.mean([x[1] for x in r2])*100:5.1f}  {statistics.mean([x[2] for x in r2])*100:5.1f}  {statistics.mean([x[3] for x in r2]):5.0f}  {w2}/3", flush=True)
        # R4 hold-quality
        r4 = [prun_r4(NavSim2(seed_cv[sd], date_lo="2020-01-01"), seed_pm[sd], seed_hm[sd], margin=mg) for sd in SEEDS]
        w4 = sum(r4[i][0] > list(base.values())[i] for i in range(3))
        print(f"  R4 hold-quality m{mg:.2f}   | x{statistics.mean([x[0] for x in r4]):5.2f}  {statistics.mean([x[1] for x in r4])*100:5.1f}  {statistics.mean([x[2] for x in r4])*100:5.1f}  {statistics.mean([x[3] for x in r4]):5.0f}  {w4}/3", flush=True)
    print("HB_119_DONE", flush=True)


if __name__ == "__main__":
    main()
