"""Batch 3 — VOL-NORMALIZED (z-score) market-regime exit gate on champ base 1166.

Fixed -5% 5d drop means very different things in calm 2024 vs the 2020 crash. A
z-scored washout (window-return standardized vs its trailing distribution) adapts
across regimes — the same 'vol-normalized helps' insight that won on the entry side.
Bar to beat: t1181 (1166 cumret 5d -0.05) comp 386.8.
"""

from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

BASE_ID = 1166
# (name, window, z_threshold, lookback)
GRID = [
    ("n2_1166_mgz5_10_l60", 5, -1.00, 60),
    ("n2_1166_mgz5_125_l60", 5, -1.25, 60),
    ("n2_1166_mgz5_15_l60", 5, -1.50, 60),
    ("n2_1166_mgz5_175_l60", 5, -1.75, 60),
    ("n2_1166_mgz5_15_l120", 5, -1.50, 120),
    ("n2_1166_mgz10_15_l60", 10, -1.50, 60),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        entry_slot = next(s for s in base.component_slots if s.slot_type == "entry")
        exit_slot = next(s for s in base.component_slots if s.slot_type == "exit")

        def _tc(slot):
            tc = slot.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

        slots_def = [
            {
                "slot_type": "entry",
                "ml_component_id": entry_slot.ml_component_id,
                "rule_component_id": None,
                "feature_set_name": entry_slot.feature_set_name,
                "target_config": _tc(entry_slot),
            },
            {
                "slot_type": "exit",
                "ml_component_id": exit_slot.ml_component_id,
                "rule_component_id": None,
                "feature_set_name": exit_slot.feature_set_name,
                "target_config": _tc(exit_slot),
            },
        ]
        base_engine = base.engine_config
        if isinstance(base_engine, str):
            base_engine = json.loads(base_engine)

        created = []
        for name, w, zt, lb in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(base_engine)
            eng["exit_market_gate_enabled"] = True
            eng["exit_market_drop_mode"] = "zscore"
            eng["exit_market_drop_window"] = w
            eng["exit_market_drop_threshold"] = zt
            eng["exit_market_z_lookback"] = lb
            tmpl = await repo.create(
                name=name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=copy.deepcopy(slots_def),
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=eng,
                validation_config=base.validation_config,
                seed=base.seed,
                description=(
                    f"Vol-normalized (zscore) market gate window {w}, z<={zt}, "
                    f"lookback {lb} on champ base 1166. Batch3."
                ),
                hypothesis="Vol-normalized washout adapts across calm vs crisis regimes; "
                "test vs t1181 cumret -0.05 comp 386.8.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
