"""Register the optimized RSI-slope shield model (2026-06-18, research-driven: rsi_slope_5 topIC -0.155 =
strongest top-signal). Clone champion 1930, engine_config only: add rsi_shield (slope_thr -14 = fires only
on a SHARP RSI rollover = fewest false fires, deep-gated min_gain 0.20 = protects the dead-zone winners
like VND +23.7%). Grid-optimized: composite 485.8 (-3.2 vs champ, the cheapest VND-catcher), VND +17%.
"""
from __future__ import annotations
import asyncio
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

BASE_ID = 1930
NAME = "n2_rsi_shield"
OV = {"rsi_shield_enabled": True, "rsi_shield_slope_thr": -14.0, "rsi_shield_min_gain": 0.20}


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_ID)
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
        tc = lambda sl: (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                         else copy.deepcopy(sl.target_config))
        ex = await repo.get_by_name(NAME)
        if ex:
            print(f"= {NAME} ({ex.id})  IDS={ex.id}"); await async_engine.dispose(); return
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": tc(sl)} for sl in base.component_slots]
        t = await repo.create(
            name=NAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold, split_config=base.split_config,
            engine_config={**copy.deepcopy(be), **OV}, validation_config=base.validation_config,
            seed=base.seed,
            description="Champion 1930 + rsi_shield (slope_thr -14, min_gain 0.20): research-optimized "
                        "risk-first shield using the strongest top-signal (rsi_slope_5 IC -0.155).",
            hypothesis="A SHARP RSI rollover (slope<=-14 over 5 bars) below MA, gated to +20% gain, exits "
                       "the dead-zone moderate-gain winners (e.g. VND +23.7%) the mechanical trailing leaves "
                       "naked, cutting giveback with minimal continuation-clip.",
            universe_slug=base.universe_slug)
        print(f"* {NAME} ({t.id})")
        await s.commit(); print(f"IDS={t.id}")
    await async_engine.dispose()


asyncio.run(main())
