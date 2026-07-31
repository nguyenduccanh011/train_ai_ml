"""Ablation: clone champion 1327 (n2_af04_st55) with overext DISABLED, to test
whether the overext top-sell currently helps or hurts. Disable cleanly:
drop "overext" from exit_priority AND set overext_ma_window=0.
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

BASE_ID = 1327
NEW_NAME = "n2_af04_st55_noox"


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
        eng["exit_priority"] = [r for r in eng.get("exit_priority", []) if r != "overext"]
        eng["overext_ma_window"] = 0

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
            entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold,
            split_config=base.split_config,
            engine_config=eng,
            validation_config=base.validation_config,
            seed=base.seed,
            description=f"{NEW_NAME}: champion 1327 with overext DISABLED (ablation).",
            hypothesis="Does overext top-sell help or hurt the current champion?",
            universe_slug=base.universe_slug,
        )
        print(f"* {NEW_NAME} ({tmpl.id})")
        await session.commit()
        print(f"ID={tmpl.id}")
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
