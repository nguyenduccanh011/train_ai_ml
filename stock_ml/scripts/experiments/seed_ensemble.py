"""Ensemble 2nd entry head: clone champion 1801 + a REVERSAL-bottom head (entry_ensemble)
unioned with the momentum head. The momentum head (triple_barrier, upleg gate) buys confirmed
uplegs and MISSES the V-bottoms; the reversal head (reversal_entry_regression: high at clean
dip-rebounds, low at tops) catches them. Union buy = catch BOTH wave types — the orthogonal
signal the parameter space can't provide. h10 target < entry h30 so gap stays 65 (leak-free)."""

from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

BASE_ID, NEW_NAME = 1801, "n2_ens_reversal"


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
        tc = lambda sl: (
            json.loads(sl.target_config)
            if isinstance(sl.target_config, str)
            else copy.deepcopy(sl.target_config)
        )
        slots = [
            {
                "slot_type": sl.slot_type,
                "ml_component_id": sl.ml_component_id,
                "rule_component_id": sl.rule_component_id,
                "feature_set_name": sl.feature_set_name,
                "target_config": tc(sl),
            }
            for sl in base.component_slots
        ]
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else be
        eng = copy.deepcopy(be)
        eng["entry_ensemble"] = {
            "target": {
                "type": "reversal_entry_regression",
                "horizon": 10,
                "penalty": 1.0,
                "dip_window": 50,
                "min_fwd_rally": 0.10,
            },
            "z_threshold": 1.5,
        }
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
            description=f"{NEW_NAME}: champion 1801 + reversal-bottom 2nd entry head (union). Orthogonal V-bottom signal.",
            hypothesis="Union a reversal_entry head (high at clean dip-rebounds) with the momentum head to catch the "
            "V-bottom winners the upleg-gated momentum head misses. Test net add under Sortino.",
            universe_slug=base.universe_slug,
        )
        print(f"* {NEW_NAME} ({t.id})\nID={t.id}")
        await s.commit()
    await async_engine.dispose()


asyncio.run(main())
