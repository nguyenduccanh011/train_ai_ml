"""Register risk-first dead-zone models from the VND clue (2026-06-18): clone champion 1930, change ONLY
the engine_config to add the tier2 dead-zone trailing band (+ the user's MACD deep-gated shield in the
combo). Heads/features/targets unchanged (deterministic, reproduce 1930's fold predictions; only the exit
engine differs). Two templates:
  n2_tier2_deadzone : trailing_tier2 0.12/0.11 (covers the [12,27%] peak-gain protection hole)
  n2_tier2_shield   : + macd_shield gated to +20% gain (the user's rollover rule, deep-gated so it does
                      not clip early pullbacks; catches the deeply-profitable round-tops e.g. VND +16.7%)
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
T2 = {"trailing_tier2_activate_pct": 0.12, "trailing_tier2_stop_pct": 0.11}
SHIELD = {"macd_shield_enabled": True, "macd_shield_min_gain": 0.20}
VARIANTS = [
    ("n2_tier2_deadzone", T2, "tier2 0.12/0.11 dead-zone trailing band (VND-clue risk-first fix)"),
    ("n2_tier2_shield", {**T2, **SHIELD},
     "tier2 0.12/0.11 + MACD deep-shield (+20% gain): user's rollover rule, deep-gated"),
]


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_ID)
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
        tc = lambda sl: (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                         else copy.deepcopy(sl.target_config))
        ids = []
        for name, ov, desc in VARIANTS:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); ids.append(ex.id); continue
            eng = {**copy.deepcopy(be), **ov}
            slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                      "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                      "target_config": tc(sl)} for sl in base.component_slots]
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
                direction=base.direction, signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold, split_config=base.split_config,
                engine_config=eng, validation_config=base.validation_config, seed=base.seed,
                description=desc,
                hypothesis="The 27%/12%-extension protection dead-zone (20% of trades, capture 0.44) leaves "
                           "moderate-gain slow-grind winners (e.g. VND +23.7%) naked; tier2 (+ deep shield) "
                           "protects them, a risk-first model the composite undervalues (giveback is in-trade).",
                universe_slug=base.universe_slug)
            print(f"* {name} ({t.id})"); ids.append(t.id)
        await s.commit()
        print("IDS=" + ",".join(str(i) for i in ids))
    await async_engine.dispose()


asyncio.run(main())
