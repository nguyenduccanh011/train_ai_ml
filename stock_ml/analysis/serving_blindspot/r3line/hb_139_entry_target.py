# -*- coding: utf-8 -*-
"""hb_139: SIGNAL axis via ENTRY TARGET (family was null hb_138). Champion entry target=triple_barrier
pt0.15/sl0.08/h30. Test variants: let-winners-run (pt0.20/sl0.06), tight-fast (pt0.12/sl0.10/h20),
long-horizon (h45). Different target -> different trades. Eval via EXPOSURE-MATCHED sizing (fair).
Compare champion 77%. Config-only (target_config), fresh predict each."""
from __future__ import annotations
import os, sys, warnings, asyncio, copy
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
VARIANTS = {
    "tb_run": {"pt": 0.20, "sl": 0.06, "type": "triple_barrier", "horizon": 30, "direction": "long"},
    "tb_tight": {"pt": 0.12, "sl": 0.10, "type": "triple_barrier", "horizon": 20, "direction": "long"},
    "tb_h45": {"pt": 0.15, "sl": 0.08, "type": "triple_barrier", "horizon": 45, "direction": "long"},
}


async def clone_tgt(name, tgt):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(3185)
        slots = []
        for sl in base.component_slots:
            tc = copy.deepcopy(sl.target_config)
            if sl.slot_type == "entry": tc = tgt
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": tc})
        t = await repo.create(name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description="entry target variant", hypothesis="target->better trades",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def evalfair(con, rid, label):
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    if not len(cvtr): print(f"  {label}: NO TRADES", flush=True); return
    cv = HERE / f"_k139_{label}.csv"; cvtr.to_csv(cv, index=False)
    feat = M.features(cvtr.symbol.unique().tolist()); pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    f, c, d, e = E.prun(NavSim2(str(cv), date_lo="2020-01-01"), pm, K=25, mode="conviction", alpha=0.6, deploy=0.55)
    print(f"  {label:12s}: trades={len(cvtr)} | fair CAGR {c*100:.1f}% DD {d*100:.1f}% expo {e:.3f}", flush=True)


def main():
    con = psycopg2.connect(**PG)
    print("ENTRY TARGET variants, eval via exposure-matched fair sizing (champion=77.0%/-11.0):", flush=True)
    for name, tgt in VARIANTS.items():
        tid = asyncio.run(clone_tgt("x2_st_" + name, tgt)); asyncio.run(async_engine.dispose())
        try:
            rid = run_template_experiment(template_id=tid, seed=42).get("run_id")
            evalfair(con, rid, name)
        except Exception as ex:
            print(f"  {name}: FAILED {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    con.close(); print("HB_139_DONE", flush=True)


if __name__ == "__main__":
    main()
