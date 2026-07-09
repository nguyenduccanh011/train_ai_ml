"""Close the +5..10% MFE protection gap. Champion arms the trailing stop only at +15% gain
(trailing_activate_pct 0.15), so the 548 trades that pop +5..10% but never reach +15% get ZERO
trailing protection and are cut late by the downleg (giveback ~100%, lag 6.3 bars). Sweep an
earlier+tighter trail to protect modest gains. No prediction needed. A/B vs champion 404.3.
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

BASE_ID = 1327
# (name, trailing_activate_pct, trailing_atr_mult, trailing_stop_pct)
GRID = [
    ("n2_prot_a07_m20", 0.07, 2.0, 0.08),
    ("n2_prot_a05_m15", 0.05, 1.5, 0.06),
    ("n2_prot_a07_m15", 0.07, 1.5, 0.06),
    ("n2_prot_a10_m15", 0.10, 1.5, 0.06),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        es = next(s for s in base.component_slots if s.slot_type == "entry")
        xs = next(s for s in base.component_slots if s.slot_type == "exit")

        def tc(sl):
            t = sl.target_config
            return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)

        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else be
        created = []
        for name, act, atr, stp in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be)
            eng["trailing_activate_pct"] = act
            eng["trailing_atr_mult"] = atr
            eng["trailing_stop_pct"] = stp
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
                description=f"{name}: champion 1327, trailing activate {act}/atr {atr}/stop {stp}.",
                hypothesis="Protect the +5-10% MFE faders with an earlier+tighter trail. Beat 404.3?",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
