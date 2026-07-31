# -*- coding: utf-8 -*-
"""hb_81: to hop CONVICTION-qua-CHON-LOC x CONCENTRATION. Entry threshold cao -> it lenh
conviction cao; K thap -> von/lenh lon. Cung nhau = tap trung von vao tin hieu tot nhat.
Cache-reuse ft_rs (threshold o recombine, predictions khong doi). NavSim moi threshold x K."""
from __future__ import annotations
import asyncio, copy, shutil, sys, os, statistics
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import psycopg2, pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE, FP = 3102, "9327ff2ec8"
RESULTS = REPO / "results"; HERE = Path(__file__).parent
THRESHS = {"th_base": -1.9, "th_10": -1.0, "th_00": 0.0, "th_05": 0.5, "th_10p": 1.0}
KS = [12, 16, 18, 20, 25]


async def make(name, thr):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        t = await repo.create(name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=thr, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=f"entry_threshold={thr} x K",
            hypothesis="conviction-selection x concentration", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    csvs = {}
    for name, thr in THRESHS.items():
        tid = asyncio.run(make(name, thr)); asyncio.run(async_engine.dispose())
        src = RESULTS / f"tmpl_{BASE}_{FP}" / "folds"; dst = RESULTS / f"tmpl_{tid}_{FP}" / "folds"
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*.parquet"):
            if not (dst / p.name).exists(): shutil.copy2(p, dst / p.name)
        r = run_template_experiment(template_id=tid, seed=42)
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                         "where run_id=%s and exit_date is not null", con, params=(r.get("run_id"),))
        cv = HERE / f"_k81_{name}.csv"; tr.to_csv(cv, index=False); csvs[name] = str(cv)
        print(f"  built {name} (thr={thr}): {len(tr)} tr", flush=True)
    print(f"\n=== NAV (adv) [threshold x K] ===", flush=True)
    print("  " + "thr".ljust(9) + "ntr".rjust(6) + "".join(f"K={k}".rjust(9) for k in KS), flush=True)
    for name, thr in THRESHS.items():
        ntr = sum(1 for _ in open(csvs[name])) - 1
        row = ""
        for k in KS:
            m = shuffle_stats(NavSim2(csvs[name], date_lo="2020-01-01"), K=k, roundtrip=0.006,
                              settle_lag=2, advance_fee=0.0008, n=20)["mean"]
            row += f"x{m:.1f}".rjust(9)
        print(f"  {str(thr):9s}{ntr:6d}{row}", flush=True)
    con.close(); print("HB_81_DONE", flush=True)


if __name__ == "__main__":
    main()
