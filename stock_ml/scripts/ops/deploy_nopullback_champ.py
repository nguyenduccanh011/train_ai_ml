"""Bake the nopullback champion = 2457 (n2_2355_exit_vol_dist) + nonbull regime-exit + min_age6
into a deployable template and multi-seed it. Round-5 found nonbull_mage6 = +23.1 (429.3, mdd 0.338)
config-only on 2457; this persists it as a template and confirms across seeds.
Usage: python stock_ml/scripts/deploy_nopullback_champ.py [seed ...]   default 42 7 99 555
"""

from __future__ import annotations
import asyncio, copy, json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

import os

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = int(os.environ.get("BASE_TMPL", "2457"))  # 2485 = the struct-RIDE crutch-free champ
NEW_NAME = f"n2_{BASE_TMPL}_nbmage6"
NB = {"nonbull_persist": 2, "nonbull_ma_win": 35, "signal_exit_min_age": 6}
SEEDS = [int(x) for x in sys.argv[1:]] or [42, 7, 99, 555]


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [
            {
                "slot_type": sl.slot_type,
                "ml_component_id": sl.ml_component_id,
                "rule_component_id": sl.rule_component_id,
                "feature_set_name": sl.feature_set_name,
                "target_config": (
                    json.loads(sl.target_config)
                    if isinstance(sl.target_config, str)
                    else copy.deepcopy(sl.target_config)
                ),
            }
            for sl in base.component_slots
        ]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng.update(NB)  # bake the nonbull regime-exit + min_age6
        t = await repo.create(
            name=NEW_NAME,
            market=base.market,
            strategy=base.strategy,
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=copy.deepcopy(slots),
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold,
            split_config=base.split_config,
            engine_config=eng,
            validation_config=base.validation_config,
            seed=base.seed,
            description="Nopullback (crutch-free) champion: 2355 red_dcf entry + exit_vol_dist (volume-"
            "distribution) exit + nonbull regime-exit + min_age6. No pullback crutch.",
            hypothesis="distribution-day exit unmasks on nopullback (+8.4); nonbull regime-exit cuts "
            "non-bull-holder MDD (+23.1). Crutch-free MDD 0.364->0.338.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    seeds = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        comp = float(row[0]) if row and row[0] is not None else None
        seeds[sd] = comp
        print(
            f"  {NEW_NAME} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    comps = [v for v in seeds.values() if v is not None]
    if comps:
        print(
            f"\n== {NEW_NAME} (tmpl {new_id}): MEAN={statistics.mean(comps):.1f} "
            f"std={statistics.pstdev(comps):.1f} seeds={seeds}"
        )
        print(
            f"== nopullback base 2355 seed42=397.8; 2457(+dist)=406.2; this(+nonbull+mage6) MEAN above"
        )
    print("DEPLOY_NOPULLBACK_DONE")


if __name__ == "__main__":
    main()
