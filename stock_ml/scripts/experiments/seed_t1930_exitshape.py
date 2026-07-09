"""Clone champion 1930, swap ONLY the EXIT slot feature_set to exit_vol_market_shape (= exit_vol_market +
the MACD-hist SHAPE family: raw + percentile + slope/accel/lags). Tests the user's temporal-pattern insight
PROPERLY (lags/curvature/percentile, not the shape-destroying 5-bar mean the B-retrain used) on the EXIT
head, where rollover detection / the dead-zone giveback lives. (2026-06-18.)
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
NAME = "n2_exit_macdshape"
NEW_FS = "exit_vol_market_shape"


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
        slots = []
        for sl in base.component_slots:
            fsn = NEW_FS if sl.slot_type == "exit" else sl.feature_set_name
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id, "feature_set_name": fsn,
                          "target_config": tc(sl)})
        t = await repo.create(
            name=NAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold, split_config=base.split_config,
            engine_config=copy.deepcopy(be), validation_config=base.validation_config, seed=base.seed,
            description="Champion 1930 + exit_vol_market_shape (MACD-hist shape: raw+pctile+slope/accel/lags).",
            hypothesis="Temporal SHAPE (lags/curvature/percentile) of the MACD histogram lets the EXIT head "
                       "read the rollover the single-point head can't, timing the protective exit to cut the "
                       "dead-zone giveback (the user's eye reads the curve, not a point).",
            universe_slug=base.universe_slug)
        print(f"* {NAME} ({t.id})")
        await s.commit(); print(f"IDS={t.id}")
    await async_engine.dispose()


asyncio.run(main())
