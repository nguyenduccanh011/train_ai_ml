"""v11: the EXIT/defensive axis is flattening (+10.5 robust banked). Switch to the ENTRY axis,
untested on the new base 1743 (no_incubate + nonbull_ma40 + belowma20p2 + min_hold2, 417.3).
Tune the entry-side levers (entry_market_gate threshold/window, pullback pct/window) which may
compose differently now that the exit defense is faster. Multi-seeded directly.
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

BASE = 1743  # n2_v10_minhold2 (417.3)
GRID = [
    ("n2_v11_emt08", {"entry_market_threshold": -0.8}),
    ("n2_v11_emt14", {"entry_market_threshold": -1.4}),
    ("n2_v11_emw3", {"entry_market_window": 3}),
    ("n2_v11_emw10", {"entry_market_window": 10}),
    ("n2_v11_pb02", {"entry_pullback_pct": 0.02}),
    ("n2_v11_pb04", {"entry_pullback_pct": 0.04}),
    ("n2_v11_pbw15", {"entry_pullback_window": 15}),
    ("n2_v11_pbw35", {"entry_pullback_window": 35}),
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
                description=f"v11 entry-axis tune {ov} on base {BASE}.",
                hypothesis="Tune entry levers on the faster-defense base -> push past 417.3 robust.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
