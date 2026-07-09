"""v14 on the final champion 1760 (pb045_w50, 433.4). The entry SCORE threshold (-1.2 = buys
~88%, barely filters) is the cleanest UNTESTED lever -> sweep selectivity. Also re-test a couple
entry FEATURE sets on the heavily-changed new base (patient-pullback profile differs a lot from
the old base where feature swaps were tested). Multi-seeded.
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

BASE = 1760
# (name, entry_threshold or None=keep, entry_feature_set or None=keep)
GRID = [
    ("n2_v14_et_m20", -2.0, None),
    ("n2_v14_et_m08", -0.8, None),
    ("n2_v14_et_00", 0.0, None),
    ("n2_v14_et_05", 0.5, None),
    ("n2_v14_fs_xsec", None, "entry_lvup126_xsec"),
    ("n2_v14_fs_recov", None, "entry_lvup126_recov"),
    ("n2_v14_fs_cheap", None, "entry_lvup126_cheap"),
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
        for name, ethr, efs in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": efs or es.feature_set_name, "target_config": tc(es)},
                {"slot_type": "exit", "ml_component_id": xs.ml_component_id, "rule_component_id": None,
                 "feature_set_name": xs.feature_set_name, "target_config": tc(xs)},
            ]
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=ethr if ethr is not None else base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(be),
                validation_config=base.validation_config, seed=base.seed,
                description=f"v14 entry-thr={ethr} feat={efs} on champ {BASE}.",
                hypothesis="Entry selectivity / feature on the new patient-pullback base -> push past 433.4.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
