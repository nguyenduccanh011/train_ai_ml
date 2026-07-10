"""Line A entry-gate wave-start sweep: clone champion 2646, replace ONLY entry_gate, run seed 42.

Usage: python a1_gate_sweep.py <clone_name> <gate|NONE>
Pattern from stock_ml/scripts/ops/deploy_wavestruct.py. Never touches template 2646 itself.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2646
BASE_COMP_S42 = 729.6

NEW_NAME = sys.argv[1]
GATE = None if sys.argv[2] == "NONE" else sys.argv[2]


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng["entry_gate"] = GATE  # REPLACE only this key
        t = await repo.create(
            name=NEW_NAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"lineA blind-spot #1 entry-gate sweep: champion 2646 with entry_gate={GATE!r} "
                        f"(champ gate 'upleg_abovema20' blocks 99% of MISSED >=30% wave starts below MA20).",
            hypothesis="relaxing/re-pocketing the abovema20 token opens below-MA20 wave-start entries; "
                       "quantify what that buys on composite under the champion's pullback execution.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME} gate={GATE!r}")
        return t.id


def main():
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    r = run_template_experiment(template_id=new_id, seed=42)
    run_id = r.get("run_id")
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    row = cur.fetchone()
    con.close()
    if row:
        comp = float(row[0])
        print(f"RESULT {NEW_NAME} gate={GATE!r} seed=42: comp={comp:.1f} (Δ{comp - BASE_COMP_S42:+.1f} vs champ {BASE_COMP_S42}) "
              f"pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} trades={row[4]}")
    else:
        print(f"RESULT {NEW_NAME}: no leaderboard row for run_id={run_id}")
    print("A1_GATE_SWEEP_DONE")


if __name__ == "__main__":
    main()
