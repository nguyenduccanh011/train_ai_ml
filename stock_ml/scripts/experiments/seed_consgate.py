"""Clone champion 1930, add the CONSOLIDATION exit-gate (exit_gate='cons2') — allow the ML signal-exit
only when consolidation_score>=2 (distribution/sideways zone = a real top), HOLD through low-consolidation
clean-uptrend pullbacks (the SOLD_THEN_RAN premature exits). Offline sweep (exit_consgate_sweep.py) found
cons2 = +2.7 composite (unimodal peak); this registers the canonical full train+backtest. 2026-06-18.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

BASE_ID = 1930
NAME = "n2_consgate2"
GATE = "cons2"


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NAME)
        if ex:
            print(f"= {NAME} ({ex.id})"); print(f"IDS={ex.id}"); await async_engine.dispose(); return
        base = await repo.get_by_id(BASE_ID)
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
        be["exit_gate"] = GATE
        tc = lambda sl: (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                         else copy.deepcopy(sl.target_config))
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": tc(sl)} for sl in base.component_slots]
        t = await repo.create(
            name=NAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold, split_config=base.split_config,
            engine_config=be, validation_config=base.validation_config, seed=base.seed,
            description="Champion 1930 + consolidation exit-gate (exit_gate=cons2): ML sell only in a "
                        "distribution/sideways zone; hold clean-uptrend pullbacks (SOLD_THEN_RAN fix).",
            hypothesis="The distribution/sideways regime (consolidation_score) is the strongest top-vs-"
                       "premature-exit separator (sep +0.82sd); gating the ML sell to it holds healthy "
                       "pullbacks without losing the mechanical downside backstop.",
            universe_slug=base.universe_slug)
        print(f"* {NAME} ({t.id})")
        await s.commit()
        print(f"IDS={t.id}")
    await async_engine.dispose()


asyncio.run(main())
