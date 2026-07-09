"""v12: patient-pullback breakthrough. Base = 1751 (n2_v11_pbw35: pullback window 25->35,
multiseed 423.4). Deeper (pct 3%->4%) and longer (window 25->35) pullback both gained big;
shallower/shorter hurt. Sweep the joint optimum (pct 4-6% x window 35-50) and stack. Multi-seeded.
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

BASE = 1751  # pullback window 35, pct 0.03
GRID = [
    ("n2_v12_pb04_w35", {"entry_pullback_pct": 0.04}),
    ("n2_v12_pb05_w35", {"entry_pullback_pct": 0.05}),
    ("n2_v12_pb06_w35", {"entry_pullback_pct": 0.06}),
    ("n2_v12_pb03_w45", {"entry_pullback_window": 45}),
    ("n2_v12_pb04_w45", {"entry_pullback_pct": 0.04, "entry_pullback_window": 45}),
    ("n2_v12_pb05_w45", {"entry_pullback_pct": 0.05, "entry_pullback_window": 45}),
    ("n2_v12_pb04_w50", {"entry_pullback_pct": 0.04, "entry_pullback_window": 50}),
    ("n2_v12_pb05_w40", {"entry_pullback_pct": 0.05, "entry_pullback_window": 40}),
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
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be); eng.update(ov)
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": es.feature_set_name, "target_config": tc(es)},
                {"slot_type": "exit", "ml_component_id": xs.ml_component_id, "rule_component_id": None,
                 "feature_set_name": xs.feature_set_name, "target_config": tc(xs)},
            ]
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"v12 pullback sweep {ov} on base {BASE}.",
                hypothesis="Find joint pullback pct x window optimum -> push past 423.4 robust.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
