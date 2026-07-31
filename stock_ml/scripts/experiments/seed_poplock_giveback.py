"""Batch 1 — protect the +5..15% MFE giveback zone on the champion (t1327, comp 404.3).

Forensics (champion_blindspot_map): 1677 trades reached MFE>+5% but gave back 121u
(49% of peak). The +5..10% band (548 trades) round-trips to -0.7% over 6.3 bars because
the trailing stop only ARMS at +15% and overext needs +14% over MA20 — the +5..15% zone
is unprotected. pop_lock arms a tight trail EARLY but ONLY for weak pops (close barely
above its short MA), letting strong pops keep the loose +15% arm so big runners aren't
clipped. Engine already supports pop_lock_*; champion does not use it.

Each variant = deepcopy(champion engine_config) + pop_lock keys (or earlier trail arm).
Predictions are identical to the champion (same features/target/model/split) so this is
an engine-only post-prediction sweep.
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

BASE = 1327
# (name, engine overrides on top of champion)
GRID = [
    (
        "n2_af04_pl_a05_e04",
        {
            "pop_lock_arm_pct": 0.05,
            "pop_lock_ext_threshold": 0.04,
            "pop_lock_ext_window": 10,
            "pop_lock_trail_pct": 0.05,
        },
    ),
    (
        "n2_af04_pl_a05_e08",
        {
            "pop_lock_arm_pct": 0.05,
            "pop_lock_ext_threshold": 0.08,
            "pop_lock_ext_window": 10,
            "pop_lock_trail_pct": 0.05,
        },
    ),
    (
        "n2_af04_pl_a06_e05",
        {
            "pop_lock_arm_pct": 0.06,
            "pop_lock_ext_threshold": 0.05,
            "pop_lock_ext_window": 10,
            "pop_lock_trail_pct": 0.05,
        },
    ),
    (
        "n2_af04_pl_a07_e06",
        {
            "pop_lock_arm_pct": 0.07,
            "pop_lock_ext_threshold": 0.06,
            "pop_lock_ext_window": 10,
            "pop_lock_trail_pct": 0.05,
        },
    ),
    (
        "n2_af04_pl_a05_e06w20",
        {
            "pop_lock_arm_pct": 0.05,
            "pop_lock_ext_threshold": 0.06,
            "pop_lock_ext_window": 20,
            "pop_lock_trail_pct": 0.05,
        },
    ),
    # comparison: unconditional earlier trail arm (no strength gate)
    ("n2_af04_ta10", {"trailing_activate_pct": 0.10}),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE)
        es = next(s for s in base.component_slots if s.slot_type == "entry")
        xs = next(s for s in base.component_slots if s.slot_type == "exit")

        def tc(sl):
            t = sl.target_config
            return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)

        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else be
        created = []
        for name, ov in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            eng.update(ov)
            slots = [
                {
                    "slot_type": "entry",
                    "ml_component_id": es.ml_component_id,
                    "rule_component_id": None,
                    "feature_set_name": es.feature_set_name,
                    "target_config": tc(es),
                },
                {
                    "slot_type": "exit",
                    "ml_component_id": xs.ml_component_id,
                    "rule_component_id": None,
                    "feature_set_name": xs.feature_set_name,
                    "target_config": tc(xs),
                },
            ]
            tmpl = await repo.create(
                name=name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=copy.deepcopy(slots),
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=eng,
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"giveback-zone protect: {ov}; base {BASE}.",
                hypothesis="Lock weak +5-15% pops early -> recover 121u giveback, beat 404.3.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
