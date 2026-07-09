"""THREE-head entry on champion 1804: momentum (score, triple_barrier) + reversal (score2,
reversal_entry, the t1804 breakthrough) + NEW breakout (score3, continuation_entry, orthogonal
to BOTH dip-leaning heads). Adds engine_config.entry_ensemble2 (3rd head). Needs the score3
pipeline support just added to experiment.py. Clone 1804, add entry_ensemble2 only.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

BASE_ID = 1804
# (tag, 3rd-head target, z_threshold)
GRID = [
    ("3h_brk05_z09", {"type": "continuation_entry_regression", "horizon": 10, "penalty": 0.5, "trend_window": 50}, 0.9),
    ("3h_brk10_z09", {"type": "continuation_entry_regression", "horizon": 10, "penalty": 1.0, "trend_window": 50}, 0.9),
]


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); base = await repo.get_by_id(BASE_ID)
        be = base.engine_config; be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
        tc = lambda sl: (json.loads(sl.target_config) if isinstance(sl.target_config, str) else copy.deepcopy(sl.target_config))
        created = []
        for tag, etgt, z in GRID:
            name = f"n2_{tag}"
            ex = await repo.get_by_name(name)
            if ex: print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be)
            eng["entry_ensemble2"] = {"target": etgt, "z_threshold": z}   # 3rd head (keep entry_ensemble = reversal)
            slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                      "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                      "target_config": tc(sl)} for sl in base.component_slots]
            t = await repo.create(name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
                direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"3-head {tag}: momentum + reversal(score2) + breakout(score3={etgt['type']} p{etgt['penalty']}) on champ 1804.",
                hypothesis="A near-high breakout 3rd head is orthogonal to BOTH dip-leaning heads; unioning it should add "
                           "winner entries the other two miss (same mdd as the 2-head breakout test) and push past 456.9.",
                universe_slug=base.universe_slug)
            print(f"* {name} ({t.id})"); created.append((t.id, name))
        await s.commit(); print("IDS=" + ",".join(str(i) for i, _ in created))
    await async_engine.dispose()

asyncio.run(main())
