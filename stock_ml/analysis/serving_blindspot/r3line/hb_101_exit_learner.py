# -*- coding: utf-8 -*-
"""hb_101 (A): LEARNER-ARCHITECTURE cho exit head (id 78 = LGBM L2 num_leaves7 + monotone_map ma/wick).
Thu: (1) objective huber/quantile (robust voi regime-outlier returns), (2) EXTENDED monotone_map ep
regime-robust (RS/regime features monotone: high-RS=hold=-1, down-vol=sell=+1), (3) bigger leaf.
Tao model_components moi, swap vao exit slot struct_to 3185. Do raw exit IC theo nam + NAV@K25."""
from __future__ import annotations
import asyncio, copy, os, sys, json, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd, scipy.stats as ss
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"; BASE = 3185; HERE = Path(__file__).parent
BASE_P = {"n_estimators": 300, "learning_rate": 0.02, "num_leaves": 7, "min_data_in_leaf": 200,
          "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 5, "lambda_l1": 1.0, "lambda_l2": 1.0}
MM = {"ma_align": -1, "ha_trend": -1, "ma5_slope": -1, "ma10_slope": -1, "ma20_slope": -1,
      "ma5_accel": -1, "upper_wick_ratio": 1, "div_rank_macd": 1, "minus_di_14": 1}
MM_EXT = {**MM, "cs_rank_trend": -1, "price_strength_rank": -1, "rs_rank_60": -1,
          "down_vol_intensity_5": 1, "down_vol_count_10": 1, "updown_vol_20": -1, "market_trend": -1}
# name -> params (all keep monotone_map MM unless overridden)
MODELS = {
    "exit_monoext": {**BASE_P, "monotone_map": MM_EXT},
    "exit_leaf15":  {**BASE_P, "num_leaves": 15, "monotone_map": MM},
    "exit_q70n":    {**BASE_P, "objective": "quantile", "alpha": 0.7},  # quantile: NO monotone (LGBM incompat)
}


def make_component(name, params):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("select id from model_components where name=%s", (name,))
    r = cur.fetchone()
    if r: con.close(); return r[0]
    cur.execute("""insert into model_components (name, role, algorithm, params, description, is_default, is_active, component_type)
                   values (%s,'exit','lightgbm',%s,%s,false,true,'model') returning id""",
                (name, json.dumps(params), f"hb_101 learner-arch {name}"))
    nid = cur.fetchone()[0]; con.commit(); con.close(); return nid


async def make_tmpl(name, exit_mlid):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(BASE)
        slots = []
        for sl in base.component_slots:
            mlid = exit_mlid if sl.slot_type == "exit" else sl.ml_component_id
            slots.append({"slot_type": sl.slot_type, "ml_component_id": mlid, "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)})
        t = await repo.create(name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=f"exit learner {name}",
            hypothesis="learner-arch exit robustness", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def fwd_frame(syms):
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,close from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date']); out = []
    for s, g in px.groupby('symbol'):
        g = g.set_index('date').sort_index(); g['fwd10'] = g['close'].shift(-10)/g['close']-1
        out.append(g[['fwd10']].assign(symbol=s).reset_index())
    return pd.concat(out)


def exit_ic(rid, con, fw):
    sig = pd.read_sql("select symbol,date,exit_score from run_signals where run_id=%s and exit_score is not null", con, params=(rid,))
    sig['date'] = pd.to_datetime(sig['date']); m = sig.merge(fw, on=['symbol', 'date']).dropna(subset=['exit_score', 'fwd10']); m['year'] = m.date.dt.year
    ov = ss.spearmanr(m.exit_score, m.fwd10).correlation
    by = {int(y): round(ss.spearmanr(g.exit_score, g.fwd10).correlation, 3) for y, g in m.groupby('year') if len(g) > 30}
    return ov, by


def nav(csv): return shuffle_stats(NavSim2(str(csv), date_lo="2020-01-01"), K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]


def main():
    con = psycopg2.connect(**PG)
    syms = pd.read_sql("select distinct symbol from run_signals where run_id='template/ft_rs-8bc8ec4e'", con).symbol.tolist()
    fw = fwd_frame(syms)
    r = run_template_experiment(template_id=3185, seed=42); rid = r.get("run_id"); bov, bby = exit_ic(rid, con, fw)
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k101_base.csv"; tr.to_csv(cv, index=False)
    print(f"BASE (L2 mono): NAV=x{nav(str(cv)):.2f} exit-IC={bov:+.3f} 2022={bby.get(2022)} 2023={bby.get(2023)} 2024={bby.get(2024)} 2026={bby.get(2026)}", flush=True)
    for name, params in MODELS.items():
        try:
            mlid = make_component(name, params)
            tid = asyncio.run(make_tmpl(name, mlid)); asyncio.run(async_engine.dispose())
            r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
            ov, by = exit_ic(rid, con, fw)
            tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
            cv = HERE / f"_k101_{name}.csv"; tr.to_csv(cv, index=False)
            print(f"  {name:11s}(t{tid},m{mlid}) NAV=x{nav(str(cv)):.2f} exit-IC={ov:+.3f} 2022={by.get(2022)} 2023={by.get(2023)} 2024={by.get(2024)} 2026={by.get(2026)} ntr={len(tr)}", flush=True)
        except Exception as e:
            print(f"  {name:11s} ERROR {type(e).__name__}: {str(e)[:70]}", flush=True)
    con.close(); print("HB_101_DONE", flush=True)


if __name__ == "__main__":
    main()
