"""Broad exit-research batch — efficiency + downside-magnitude axes (NOT direction).
Each entry: (name, base_template_id, engine_overrides dict)."""

from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

OX = {
    "overext_ma_window": 20,
    "overext_pct": 0.14,
    "overext_reversal": False,
    "exit_priority": ["trailing_stop", "overext", "signal"],
}


def merge(*ds):
    o = {}
    for d in ds:
        o.update(d)
    return o


GRID = [
    # mgz low-mdd base (1187, no age-incubation, mdd 2.91) + overext  -> low mdd + efficiency
    ("n2_mgz_ox14", 1187, OX),
    ("n2_mgz_ox12", 1187, merge(OX, {"overext_pct": 0.12})),
    ("n2_mgz_ox16", 1187, merge(OX, {"overext_pct": 0.16})),
    # threshold gap-fill around the 0.14 peak (no-rev) on age champ
    ("n2_1204_ox20_13", 1204, merge(OX, {"overext_pct": 0.13})),
    ("n2_1204_ox20_15n", 1204, merge(OX, {"overext_pct": 0.15})),
    # trailing tweaks on the overext champ (the profitable mechanism)
    ("n2_1204_ox14_tr06", 1204, merge(OX, {"trailing_stop_pct": 0.06})),
    ("n2_1204_ox14_act10", 1204, merge(OX, {"trailing_activate_pct": 0.10})),
    # hard stop to cut the entry-driven loser tail (mdd)
    (
        "n2_1204_ox14_hs10",
        1204,
        merge(
            OX,
            {
                "hard_stop_pct": -0.10,
                "exit_priority": ["hard_stop", "trailing_stop", "overext", "signal"],
            },
        ),
    ),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        created = []
        for name, base_id, ov in GRID:
            base = await repo.get_by_id(base_id)
            es = next(s for s in base.component_slots if s.slot_type == "entry")
            xs = next(s for s in base.component_slots if s.slot_type == "exit")

            def tc(sl):
                t = sl.target_config
                return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)

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
            be = base.engine_config
            be = json.loads(be) if isinstance(be, str) else be
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            eng.update(ov)
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
                description=f"Broad exit batch {name} (base {base_id}): {ov}.",
                hypothesis="Efficiency/magnitude-axis exit levers; beat overext champ 393.8.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
