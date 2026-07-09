"""v10: the pattern is 'champion was too SLOW to cut weak trades' -> every gain came from
faster/cleaner defense (no_incubate, nonbull_ma 50->40, persist 3->2). Base = 1731
(ma40_p2, multiseed 416.4). Now: re-ablate the BIG levers on the new base (lever interactions
shifted) + tune the other DEFENSIVE-DELAY lever = exit_market_gate (it SUPPRESSES signal-exits
in market washouts = a 'hold through the dip' delay; maybe too protective now that nonbull is
faster). Also re-probe overext depth + entry-gate looseness. Multi-seeded directly.
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

BASE = 1731  # n2_v9_ma40_p2 (416.4)
GRID = [
    ("n2_v10_noexitmkt", {"exit_market_gate_enabled": False}),
    ("n2_v10_exitmkt_z25", {"exit_market_drop_threshold": -2.5}),
    ("n2_v10_exitmkt_z125", {"exit_market_drop_threshold": -1.25}),
    ("n2_v10_exitmkt_w10", {"exit_market_drop_window": 10}),
    ("n2_v10_ox10", {"overext_pct": 0.10}),
    ("n2_v10_ox14", {"overext_pct": 0.14}),
    ("n2_v10_nocool", {"reentry_cooldown_bars": 0}),
    ("n2_v10_minhold2", {"min_hold_bars": 2}),
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
                description=f"v10 defense re-ablate/tune {ov} on base {BASE}.",
                hypothesis="Find more 'too-slow-defense' levers on the new base -> push past 416.4 robust.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
