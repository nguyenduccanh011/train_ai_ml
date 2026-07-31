"""New champion: clone 1799 + trailing_activate_pct 0.15->0.20 (arm the trailing stop only after
+20% MFE so winners run further before any give-back protection). Under Sortino this is a clean
Pareto win — bigger winners (same downside) are no longer penalised. 4-seed validated: comp
~446->~451, pnl 106->108.9, mdd 1.92->1.74."""

from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

BASE_ID, NEW_NAME = 1799, "n2_v20_act20"


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_ID)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"= {NEW_NAME} ({ex.id})")
            await async_engine.dispose()
            return
        e_ = next(x for x in base.component_slots if x.slot_type == "entry")
        x_ = next(x for x in base.component_slots if x.slot_type == "exit")
        tc = lambda sl: (
            json.loads(sl.target_config)
            if isinstance(sl.target_config, str)
            else copy.deepcopy(sl.target_config)
        )
        slots = [
            {
                "slot_type": "entry",
                "ml_component_id": e_.ml_component_id,
                "rule_component_id": None,
                "feature_set_name": e_.feature_set_name,
                "target_config": tc(e_),
            },
            {
                "slot_type": "exit",
                "ml_component_id": x_.ml_component_id,
                "rule_component_id": None,
                "feature_set_name": x_.feature_set_name,
                "target_config": tc(x_),
            },
        ]
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else be
        eng = copy.deepcopy(be)
        eng["trailing_activate_pct"] = 0.20
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
            description=f"{NEW_NAME}: champion 1799 + trailing_activate_pct 0.20 (arm trail later, ride winners further).",
            hypothesis="Arming the trailing stop at +20% MFE (not +15%) lets winners run further; under Sortino the bigger winners (same downside) are a win.",
            universe_slug=base.universe_slug,
        )
        print(f"* {NEW_NAME} ({t.id})\nID={t.id}")
        await s.commit()
    await async_engine.dispose()


asyncio.run(main())
