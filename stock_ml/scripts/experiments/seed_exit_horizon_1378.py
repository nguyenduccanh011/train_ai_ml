"""Faster exit head: lower the signal-exit TARGET horizon on champion t1378.

The signal exit (reward_risk h10) reacts late (lag 7.4 bars from peak). Top-sell
add-ons (pop-lock trail, reversal-confirmed overext) both tested negative. Distinct
lever: make the head ITSELF faster by shortening the forward window it regresses —
reward_risk over h5/h7 instead of h10 should let the exit score roll over sooner.

Clone t1378, change ONLY the exit slot target horizon. A/B vs FRESH t1378 (405.0).
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

BASE_ID = 1378
GRID = [("exh5", 5), ("exh7", 7), ("exh8", 8)]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        if base is None:
            raise ValueError(f"base template id={BASE_ID} not found")

        def _tc(slot):
            tc = slot.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

        base_ec = base.engine_config
        base_ec = json.loads(base_ec) if isinstance(base_ec, str) else copy.deepcopy(base_ec)

        created = []
        for tag, h in GRID:
            name = f"n2_1378_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id}), skip")
                created.append((ex.id, name))
                continue
            new_slots = []
            for s in base.component_slots:
                tc = _tc(s)
                if s.slot_type == "exit":
                    tc = dict(tc); tc["horizon"] = h
                new_slots.append({
                    "slot_type": s.slot_type, "ml_component_id": s.ml_component_id,
                    "rule_component_id": s.rule_component_id, "feature_set_name": s.feature_set_name,
                    "target_config": tc})
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=new_slots, direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(base_ec),
                validation_config=base.validation_config, seed=base.seed,
                description=f"Faster exit head: reward_risk horizon {h} (was 10) on t1378; all else = t1378.",
                hypothesis="Signal exit reacts late (lag 7.4 bars). A shorter reward_risk horizon "
                           "should make the exit score roll over sooner on faded winners. Test vs t1378 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(tid) for tid, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
