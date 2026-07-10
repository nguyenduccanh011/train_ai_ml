"""Sweep virgin axis #5: z-normalization lookback (z_norm_window / z_norm_min_periods).

Clones champion template 2646 (never modified) + ONLY the 2 new engine keys per cell.
Pattern: stock_ml/scripts/ops/deploy_wavestruct.py. Seed 42 (leaderboard batch seed).
Usage: python stock_ml/scripts/experiments/sweep_znorm.py NAME WINDOW MINP [seed ...]
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2646
CHAMP = dict(comp=729.6, pnl=127.25301810134556, pf=5.963310002880663,
             mdd=0.17454927579356125, tr=1384)


async def make_clone(name: str, window: int, minp: int) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config)
                                    if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng.update({"z_norm_window": window, "z_norm_min_periods": minp})
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"champion 2646 + z-norm lookback sweep (virgin axis #5): "
                        f"z_norm_window={window}, z_norm_min_periods={minp} "
                        f"(hardcoded 252/60 since inception, never swept).",
            hypothesis="the per-symbol causal z lookback (252/60) that normalizes zE/zX and "
                       "all ensemble heads was never swept in 2600+ templates; a shorter/longer "
                       "adaptation window may materially shift entry/exit timing.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} z_norm_window={window} minp={minp}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone()
    con.close()
    return r


def main():
    name, window, minp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seeds = [int(x) for x in sys.argv[4:]] or [42]
    new_id = asyncio.run(make_clone(name, window, minp))
    asyncio.run(async_engine.dispose())
    for sd in seeds:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        if row is None:
            print(f"RESULT {name} seed={sd}: NO ROW")
            continue
        comp, pnl, pf, mdd, tr = (float(row[0]), float(row[1]), float(row[2]),
                                  float(row[3]), int(row[4]))
        print(f"RESULT {name} tmpl={new_id} seed={sd} comp={comp} d={comp - CHAMP['comp']:+.1f} "
              f"pnl={pnl:.3f} pf={pf:.3f} mdd={mdd:.4f} tr={tr}", flush=True)
    print("SWEEP_ZNORM_DONE")


if __name__ == "__main__":
    main()
