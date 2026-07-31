"""Breakout 2nd-entry-head diagnostic on champion 1804. The momentum head (triple_barrier)
AND the reversal head (score2) are BOTH dip-leaning (corr score x dist20hi -0.41); a continuation/
breakout head (scores near-high high, corr +0.41) is orthogonal to BOTH. Cheap test FIRST: swap
the single ensemble slot reversal->continuation (config-only retrain) to see if a breakout 2nd head
is competitive with the reversal one / the champion. If yes -> worth coding a true 3rd head (keep
both reversal + breakout). Clone 1804, change ONLY engine_config.entry_ensemble.target.
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
# (tag, ensemble target) — the 2nd entry head's target
GRID = [
    (
        "brk_p10",
        {
            "type": "continuation_entry_regression",
            "horizon": 10,
            "penalty": 1.0,
            "trend_window": 50,
        },
    ),
    (
        "brk_p05",
        {
            "type": "continuation_entry_regression",
            "horizon": 10,
            "penalty": 0.5,
            "trend_window": 50,
        },
    ),
]


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_ID)
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
        tc = lambda sl: (
            json.loads(sl.target_config)
            if isinstance(sl.target_config, str)
            else copy.deepcopy(sl.target_config)
        )
        created = []
        for tag, etgt in GRID:
            name = f"n2_ens_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            ens = dict(eng.get("entry_ensemble") or {})
            ens["target"] = etgt  # swap reversal -> continuation (breakout)
            ens.setdefault("z_threshold", 0.9)
            eng["entry_ensemble"] = ens
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
            t = await repo.create(
                name=name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=slots,
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=eng,
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"Breakout 2nd-head {tag}: entry_ensemble target -> {etgt['type']} (was reversal) on champ 1804.",
                hypothesis="A near-high breakout head is orthogonal to BOTH the dip-leaning momentum and reversal heads; "
                "union should add winner entries they miss without starving the engine.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({t.id})")
            created.append((t.id, name))
        await s.commit()
        print("IDS=" + ",".join(str(i) for i, _ in created))
    await async_engine.dispose()


asyncio.run(main())
