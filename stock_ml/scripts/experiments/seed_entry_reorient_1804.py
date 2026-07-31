"""Entry-head REORIENT on champion 1804. Root cause (confirm_rangepos_target.py): the entry
head is a DIP-BUYER (corr score x dist20hi = -0.41) while the real edge is near-high continuation
(fwd10 +1.52% at range_pos>0.8 vs +0.06% at <0.2). range_pos_20 is already an INPUT but the
triple_barrier target makes the head score the OPPOSITE way. Fix = swap the entry TARGET to one
that rewards continuation/breakout, keeping all features + engine. Then sweep entry_threshold on
the new folds (selectivity should finally help once the head ranks correctly).
Clone 1804, change ONLY the entry slot target_config.
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
# (tag, entry target_config)
GRID = [
    (
        "cont_p10",
        {
            "type": "continuation_entry_regression",
            "horizon": 10,
            "penalty": 1.0,
            "trend_window": 50,
        },
    ),
    (
        "cont_p05",
        {
            "type": "continuation_entry_regression",
            "horizon": 10,
            "penalty": 0.5,
            "trend_window": 50,
        },
    ),
    ("fwdret", {"type": "forward_return_regression", "horizon": 10}),
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
            name = f"n2_re_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            slots = []
            for sl in base.component_slots:
                cfg = etgt if sl.slot_type == "entry" else tc(sl)
                slots.append(
                    {
                        "slot_type": sl.slot_type,
                        "ml_component_id": sl.ml_component_id,
                        "rule_component_id": sl.rule_component_id,
                        "feature_set_name": sl.feature_set_name,
                        "target_config": copy.deepcopy(cfg),
                    }
                )
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
                engine_config=copy.deepcopy(be),
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"Entry REORIENT {tag}: entry target -> {etgt['type']} on champ 1804 (was triple_barrier dip-buyer).",
                hypothesis="Reorienting the entry head toward continuation/breakout (it currently scores dips high, "
                "corr score x dist20hi -0.41, anti the +1.52% near-high edge) lets a tighter entry_threshold "
                "finally cull the -18.7u signal-exit drain without starving the winner engine.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({t.id})")
            created.append((t.id, name))
        await s.commit()
        print("IDS=" + ",".join(str(i) for i, _ in created))
    await async_engine.dispose()


asyncio.run(main())
