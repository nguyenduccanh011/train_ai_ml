# -*- coding: utf-8 -*-
"""hb_138: SIGNAL axis (user: khong ngai kho). Test entry LEARNER FAMILY = random_forest vs champion
lightgbm. Hypothesis: RF bagging (no gradient) more regime-robust across 2024 sign-flip than GBDT
(overfits recent). Create RF entry component, clone t3185 (entry->RF), run, eval trades via meta +
EXPOSURE-MATCHED sizing (deploy0.55). Compare fair champion 77%/-11. If RF-entry > -> family matters."""
from __future__ import annotations
import os, sys, warnings, asyncio, copy, json
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd
from nh_nav2 import NavSim2
from scripts.run_template import run_template_experiment
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
import hb_112_meta_target as M
import hb_136_exposure_matched as E

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent
RF_PARAMS = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 50, "max_features": 0.6}


def ensure_rf_component(con):
    cur = con.cursor()
    cur.execute("select id from model_components where name='entry_rf_regime'")
    r = cur.fetchone()
    if r: return r[0]
    cur.execute("""insert into model_components (name, role, algorithm, params, description, is_default, is_active, component_type, created_at, updated_at)
                   values (%s,%s,%s,%s,%s,false,true,'ml',now(),now()) returning id""",
                ("entry_rf_regime", "entry", "random_forest", json.dumps(RF_PARAMS), "RF entry for regime-robustness test"))
    cid = cur.fetchone()[0]; con.commit(); return cid


async def clone_rf(rf_cid, name):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(3185)
        slots = []
        for sl in base.component_slots:
            mlid = rf_cid if sl.slot_type == "entry" else sl.ml_component_id
            slots.append({"slot_type": sl.slot_type, "ml_component_id": mlid, "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)})
        t = await repo.create(name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description="RF entry family test", hypothesis="RF regime-robust",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def evalfair(con, rid, label):
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    if not len(cvtr): print(f"  {label}: NO TRADES"); return
    cv = HERE / f"_k138_{label}.csv"; cvtr.to_csv(cv, index=False)
    feat = M.features(cvtr.symbol.unique().tolist()); pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    f, c, d, e = E.prun(NavSim2(str(cv), date_lo="2020-01-01"), pm, K=25, mode="conviction", alpha=0.6, deploy=0.55)
    print(f"  {label}: trades={len(cvtr)} | fair CAGR {c*100:.1f}% DD {d*100:.1f}% expo {e:.3f}", flush=True)


def main():
    con = psycopg2.connect(**PG)
    rf_cid = ensure_rf_component(con)
    print(f"RF component id={rf_cid}", flush=True)
    tid = asyncio.run(clone_rf(rf_cid, "x2_struct_to_rfentry")); asyncio.run(async_engine.dispose())
    print(f"RF template id={tid}", flush=True)
    try:
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
        print("RF run OK", flush=True)
        evalfair(con, rid, "RF-entry")
    except Exception as ex:
        print(f"RF run FAILED: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
    # champion ref
    rc = run_template_experiment(template_id=3185, seed=42).get("run_id")
    evalfair(con, rc, "LGB-champion")
    con.close(); print("HB_138_DONE", flush=True)


if __name__ == "__main__":
    main()
