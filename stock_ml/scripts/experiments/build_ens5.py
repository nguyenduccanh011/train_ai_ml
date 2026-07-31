"""ADDITIVE direction-robust 5th ensemble head on frontier x2_struct_to (t3185).

Diagnosis (entry-root-amplitude-not-direction): the entry score knows AMPLITUDE (mfe IC +0.206
robust) but its DIRECTION flips in dead years (market-beta). Prior fix REPLACED the target and
killed amplitude -> NAV-negative (coupling). This probe ADDS a 5th head (entry_ensemble5 -> score6,
unioned at a conservative z_threshold) trained on a direction-robust target, WITHOUT touching the
amplitude primary — keep good-year alpha, add dead-year direction picks only. Baseline = frontier
seed42 NAV 27.14, CAGR 66.0%, DD -13.7%.
"""

from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = 3185

# name -> entry_ensemble5 engine block (target + union z_threshold)
VARIANTS = {
    "e5_ampdir_z09": {
        "target": {"type": "amplitude_direction_entry", "horizon": 20},
        "z_threshold": 0.9,
    },
    "e5_ampdir_z11": {
        "target": {"type": "amplitude_direction_entry", "horizon": 20},
        "z_threshold": 1.1,
    },
    "e5_csent_z09": {
        "target": {"type": "cross_sectional_entry", "horizon": 10},
        "z_threshold": 0.9,
    },
}


async def make_all():
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE)
        bs = []
        for sl in base.component_slots:
            tc = sl.target_config
            tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            bs.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": sl.feature_set_name,
                    "target_config": tc,
                }
            )
        be0 = base.engine_config
        be0 = json.loads(be0) if isinstance(be0, str) else dict(be0)
        for nm, ens5 in VARIANTS.items():
            ex = await repo.get_by_name(nm)
            if ex:
                print(f"exists {nm} {ex.id}")
                ids[nm] = ex.id
                continue
            be = copy.deepcopy(be0)
            be["entry_ensemble5"] = ens5
            slots = copy.deepcopy(bs)
            t = await repo.create(
                name=nm,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=slots,
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=be,
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"frontier + additive direction head entry_ensemble5={ens5}",
                hypothesis="additive direction-robust head adds dead-year picks without killing amplitude alpha",
                universe_slug=base.universe_slug,
                model_mode=base.model_mode,
            )
            await s.commit()
            print(f"created {nm} {t.id}")
            ids[nm] = t.id
    return ids


def read(rid):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score,total_pnl,trades FROM leaderboard_runs WHERE run_id=%s", (rid,)
    )
    r = cur.fetchone()
    con.close()
    return r


ids = asyncio.run(make_all())
asyncio.run(async_engine.dispose())
for nm, tid in ids.items():
    try:
        r = run_template_experiment(template_id=tid, seed=42)
        rid = r.get("run_id")
        row = read(rid)
        print(
            f"  {nm}(t{tid}): comp={row[0]} pnl={row[1]:.1f} tr={row[2]} run_id={rid}", flush=True
        )
    except Exception as e:
        print(f"  {nm}: ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
print("ENS5_DONE")
