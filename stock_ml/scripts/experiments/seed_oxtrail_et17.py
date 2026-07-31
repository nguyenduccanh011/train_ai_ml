"""Clone champion 1776 with TWO stacked, harness+multi-seed-validated levers:
  (1) overext->tight-trail handoff (overext_trail_pct=0.04)         -> +3.1 comp, mdd 2.11->1.89
  (2) light entry-ML application (entry_threshold -3.0 -> -1.7)     -> +1.8..+2.2 across 4 seeds
Net ~435.5 -> ~440.5. (2) culls the ~1% lowest temporal-z entries (the capital-hog q0 cohort),
turning the entry ML back ON a little instead of the pure -3 (ML fully off).
"""

from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

BASE_ID = 1776
NEW_NAME = "n2_v17_oxtrail04_et17"


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"= {NEW_NAME} ({ex.id})")
            await async_engine.dispose()
            return

        es = next(s for s in base.component_slots if s.slot_type == "entry")
        xs = next(s for s in base.component_slots if s.slot_type == "exit")

        def tc(sl):
            t = sl.target_config
            return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)

        slots = [
            {
                "slot_type": "entry",
                "ml_component_id": es.ml_component_id,
                "rule_component_id": None,
                "feature_set_name": es.feature_set_name,
                "target_config": tc(es),
            },
            {
                "slot_type": "exit",
                "ml_component_id": xs.ml_component_id,
                "rule_component_id": None,
                "feature_set_name": xs.feature_set_name,
                "target_config": tc(xs),
            },
        ]
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else be
        eng = copy.deepcopy(be)
        eng["overext_trail_pct"] = 0.04

        tmpl = await repo.create(
            name=NEW_NAME,
            market=base.market,
            strategy=base.strategy,
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=copy.deepcopy(slots),
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=-1.7,
            exit_threshold=base.exit_threshold,
            split_config=base.split_config,
            engine_config=eng,
            validation_config=base.validation_config,
            seed=base.seed,
            description=f"{NEW_NAME}: champion 1776 + overext->trail handoff (0.04) + light entry-ML (entry_threshold -1.7).",
            hypothesis="Overext->trail captures the extension; a small entry_threshold (-1.7) re-applies the entry ML "
            "to cull the lowest-z capital-hog entries. Both validated multi-seed in the offline harness.",
            universe_slug=base.universe_slug,
        )
        print(f"* {NEW_NAME} ({tmpl.id})")
        await session.commit()
        print(f"ID={tmpl.id}")
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
