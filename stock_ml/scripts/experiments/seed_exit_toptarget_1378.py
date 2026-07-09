"""Re-target the EXIT head to be TOP-SHAPED — champion t1378 model-quality fix.

Score-shape diagnostic (_tmp_analysis/score_shape_t1378.py): the exit head (reward_risk
h10) does NOT peak at price tops — exit_z is LOWEST at tops (-0.07) and HIGHEST while
rising (+0.15). reward_risk = MFE/MAE is structurally inverted for an exit (small forward
MFE at a top -> low score). So the head carries no top-timing signal; all top-selling is
done by the mechanical overext/trail rules. That's why shortening its horizon barely moved
anything (exh5 +0.5).

Fix: give the exit head a label that PEAKS at tops — the mirror of the entry survival head
(triple_barrier long, which lifted entry). Two families:
  * triple_barrier direction="short" — label 1 if price DROPS pt before rising sl within
    horizon = "a real top / imminent drop". High score at tops.
  * forward_drawdown_regression — directly regress the forward drawdown (high = drop coming).

Clone t1378, change ONLY the exit slot target. A/B vs FRESH t1378 (405.0). Keep feature
set, engine_config (overext/trail/gate/incubation) unchanged — isolate the exit TARGET.
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

# (tag, exit target_config) — a top-shaped label so the exit score peaks at tops.
GRID = [
    ("xtop_tb08", {"type": "triple_barrier", "horizon": 15, "pt": 0.08, "sl": 0.05, "direction": "short"}),
    ("xtop_tb10", {"type": "triple_barrier", "horizon": 20, "pt": 0.10, "sl": 0.05, "direction": "short"}),
    ("xtop_tb06", {"type": "triple_barrier", "horizon": 12, "pt": 0.06, "sl": 0.04, "direction": "short"}),
    ("xtop_fdd10", {"type": "forward_drawdown_regression", "horizon": 10}),
]


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
        for tag, xt in GRID:
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
                    tc = copy.deepcopy(xt)
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
                description=f"Top-shaped exit head: exit target -> {xt} on t1378; all else = t1378.",
                hypothesis="Exit head (reward_risk) doesn't peak at tops (exit_z -0.07 at tops, +0.15 "
                           "while rising) -> no top-timing signal. A top-shaped target (triple_barrier "
                           "short / forward_drawdown) should make the exit score spike at tops, mirroring "
                           "the entry survival-head win. Test vs t1378 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(tid) for tid, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
