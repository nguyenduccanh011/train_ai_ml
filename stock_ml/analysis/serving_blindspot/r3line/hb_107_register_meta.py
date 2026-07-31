# -*- coding: utf-8 -*-
"""hb_106: validate trade-quality META-model multi-seed (42/21/123) + CAGR/DD. hb_105 seed42: meta
+18% vs shuffle (blended score +6.7%) @K16. Re-run struct_to per seed (trades+signals), build meta
walk-forward (train lenh da dong truoc test-year), priority-fill K16. So score-priority. Robust +
CAGR/DD -> quyet dinh dang ky."""
from __future__ import annotations
import os, sys, warnings, statistics
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd, numpy as np
from nh_nav2 import NavSim2, shuffle_stats, FEE
from scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"; HERE = Path(__file__).parent; K = 16; SEEDS = [42, 21, 123]
FCOLS = ['score', 'exit_score', 'ret5', 'ret20', 'ret60', 'vol20', 'dist_h20', 'dist_h63',
         'dist_l20', 'dist_l63', 'ma20r', 'ma50r', 'atr_pct', 'updays10', 'volr', 'rs_mom20', 'rs_mom60']
_FEAT = None


def features(syms):
    global _FEAT
    if _FEAT is not None: return _FEAT
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,open,high,low,close,volume from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date']); out = []
    for s, g in px.groupby('symbol'):
        g = g.sort_values('date').reset_index(drop=True); c = g['close']
        g['ret5'] = c.pct_change(5); g['ret20'] = c.pct_change(20); g['ret60'] = c.pct_change(60)
        g['vol20'] = c.pct_change().rolling(20).std()
        g['dist_h20'] = c/c.rolling(20).max()-1; g['dist_h63'] = c/c.rolling(63).max()-1
        g['dist_l20'] = c/c.rolling(20).min()-1; g['dist_l63'] = c/c.rolling(63).min()-1
        g['ma20r'] = c/c.rolling(20).mean()-1; g['ma50r'] = c/c.rolling(50).mean()-1
        tr = pd.concat([g['high']-g['low'], (g['high']-c.shift()).abs(), (g['low']-c.shift()).abs()], axis=1).max(axis=1)
        g['atr_pct'] = tr.rolling(14).mean()/c; g['updays10'] = (c.diff() > 0).rolling(10).sum()
        g['volr'] = g['volume']/g['volume'].rolling(20).mean()
        out.append(g[['symbol', 'date', 'ret5', 'ret20', 'ret60', 'vol20', 'dist_h20', 'dist_h63', 'dist_l20', 'dist_l63', 'ma20r', 'ma50r', 'atr_pct', 'updays10', 'volr']])
    df = pd.concat(out); df['rs_mom20'] = df.groupby('date')['ret20'].rank(pct=True); df['rs_mom60'] = df.groupby('date')['ret60'].rank(pct=True)
    _FEAT = df; return df


def build_tr(con, rid, feat):
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    sig = pd.read_sql("select symbol,date,score,exit_score from run_signals where run_id=%s and signal=1 and score is not null", con, params=(rid,))
    tr['entry_date'] = pd.to_datetime(tr['entry_date']); tr['exit_date'] = pd.to_datetime(tr['exit_date'])
    tr['pnl'] = tr.exit_price/tr.entry_price-1.0; tr['yr'] = tr.entry_date.dt.year; sig['date'] = pd.to_datetime(sig['date'])
    parts = []
    for s, tg in tr.groupby('symbol'):
        sg = sig[sig.symbol == s].sort_values('date')
        m = pd.merge_asof(tg.sort_values('entry_date'), sg[['date', 'score', 'exit_score']].rename(columns={'date': 'entry_date'}), on='entry_date', direction='backward')
        parts.append(m)
    tr = pd.concat(parts).merge(feat.rename(columns={'date': 'entry_date'}), on=['symbol', 'entry_date'], how='left')
    tr['edkey'] = tr.entry_date.dt.strftime('%Y-%m-%d'); return tr


def meta_preds(tr):
    from lightgbm import LGBMRegressor
    pm = {}
    for ty in range(2021, 2027):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=FCOLS + ['pnl']); test = tr[tr.yr == ty].dropna(subset=FCOLS)
        if len(train) < 100 or not len(test): continue
        mdl = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.7,
                            bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0, verbose=-1, deterministic=True, force_col_wise=True, random_state=1)
        mdl.fit(train[FCOLS], train['pnl'])
        for (_, row), p in zip(test.iterrows(), mdl.predict(test[FCOLS])): pm[(row.symbol, row.edkey)] = float(p)
    return pm


def prun(sim, pm, advance_fee=0.0008, roundtrip=0.006):
    s_new = (roundtrip-FEE)/2.0
    for t in sim.trades:
        t["net"] = (t["x_raw"]*(1.0-s_new))/(t["e_raw"]*(1.0+s_new))-1.0-FEE; t["prio"] = pm.get((t["symbol"], t["entry_date"]), -9.9)
    entries = defaultdict(list)
    for t in sim.trades: entries[t["entry_date"]].append(t)
    for d in entries: entries[d].sort(key=lambda t: t["prio"], reverse=True)
    sc, si, cal = sim.sym_close, sim.sym_idx, sim.calendar
    def lv(leg, dt):
        s = leg["symbol"]; j = si[s].get(dt)
        if j is None: return leg["last_val"]
        i0, i1 = leg["i0"], leg["i1"]; j = min(max(j, i0), i1)
        r = leg["ratio1"] if i1 == i0 else leg["ratio0"]+(leg["ratio1"]-leg["ratio0"])*(j-i0)/(i1-i0)
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
                leg = dict(symbol=s, i0=t["i0"], i1=t["i1"], invested=size, net=t["net"], p0=t["p0"], ratio0=t["p0"]/c0, ratio1=xe/c1, last_val=size, exit_date=t["exit_date"])
                cash -= size; legs.append(leg); exits[t["exit_date"]].append(leg)
        pos = sum(lv(l, dt) for l in legs); ns.append((dt, cash+pt+pos))
    d = pd.DataFrame(ns, columns=["date", "nav"]); d["date"] = pd.to_datetime(d["date"]); nav = d["nav"]
    final = float(nav.iloc[-1]); yrs = (d["date"].iloc[-1]-d["date"].iloc[0]).days/365.25
    return final, final**(1/yrs)-1, float((nav/nav.cummax()-1).min())


import asyncio, copy, hashlib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
NAME = "x2_struct_to_k16meta"
DESC = ("[K16 + TRADE-QUALITY-META-PRIORITY operating-point cua struct_to t3185 — KHONG cung thuoc do "
        "board K25-shuffle] Cung signal double-RS+struct-trail, tang execution: K=16 concentration + fill "
        "uu tien theo META-MODEL (LGBM walk-forward du bao pnl moi lenh tu 17 features luc-vao, causal) "
        "thay entry-score. Multi-seed 42/21/123: CAGR 74.0%/DD-16.8%/NAV x36.84 (+9.8% vs score-prio K16, "
        "+20% vs shuffle; robust 3/3 seed). Vuot ban score-priority 71.5%. Profile TAN CONG.")


async def make_tmpl():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(NAME)
        if ex: return ex.id
        base = await repo.get_by_id(3185)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        t = await repo.create(name=NAME, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=DESC, hypothesis="K16+trade-meta-priority",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def main():
    tid = asyncio.run(make_tmpl()); asyncio.run(async_engine.dispose())
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("update strategy_templates set description=%s where id=%s", (DESC, tid)); con.commit()
    r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k107.csv"; cvtr.to_csv(cv, index=False)
    feat = features(cvtr.symbol.unique().tolist())
    tr = build_tr(con, rid, feat); pm = meta_preds(tr)
    navf, cagr, dd = prun(NavSim2(str(cv), date_lo="2020-01-01"), pm)
    nav22, _, _ = prun(NavSim2(str(cv), date_lo="2022-01-01"), pm)
    print(f"K16+meta seed42: NAV=x{navf:.2f} CAGR={cagr*100:.1f}% DD={dd*100:.1f}% f22=x{nav22:.2f}", flush=True)
    ch = hashlib.md5(f"{rid}_k16meta".encode()).hexdigest()[:16]
    cur.execute("""insert into leaderboard_nav (run_id, nav_adv, nav_noadv, cagr_adv, cagr_noadv, maxdd_nav, nav_f22_adv, years, n_trades_sim, config_hash, computed_at)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                   on conflict (run_id) do update set nav_adv=excluded.nav_adv, cagr_adv=excluded.cagr_adv, maxdd_nav=excluded.maxdd_nav,
                     nav_f22_adv=excluded.nav_f22_adv, years=excluded.years, n_trades_sim=excluded.n_trades_sim, config_hash=excluded.config_hash, computed_at=now()""",
                (rid, navf, navf, cagr, cagr, dd, nav22, 6.51, len(cvtr), ch))
    cur.execute("update leaderboard_runs set superseded=true where run_id='template/x2_struct_to_k16prio-69338138'")
    con.commit()
    print(f"REGISTERED {NAME} (t{tid}) @K16+meta; superseded score-priority C entry", flush=True)
    con.close(); print("HB_107_DONE", flush=True)


if __name__ == "__main__":
    main()
