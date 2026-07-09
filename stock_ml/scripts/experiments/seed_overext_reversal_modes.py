"""Test reversal-confirmed overext on champion 1327. Baseline (no confirm) = champion
404.3. Each variant only flips overext_reversal_mode (+ a param) so the overext top-sell
waits for a reversal pattern instead of firing on extension alone.
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
# (name, overrides)
GRID = [
    ("n2_ox_down1",      {"overext_reversal_mode": "down1"}),
    ("n2_ox_strongdn2",  {"overext_reversal_mode": "strong_down", "overext_strong_down_pct": 0.02}),
    ("n2_ox_strongdn3",  {"overext_reversal_mode": "strong_down", "overext_strong_down_pct": 0.03}),
    ("n2_ox_engulf2",    {"overext_reversal_mode": "engulf2"}),
    ("n2_ox_3down",      {"overext_reversal_mode": "three_down"}),
    ("n2_ox_emax10",     {"overext_reversal_mode": "ema_cross", "overext_ema_span": 10}),
    ("n2_ox_emax20",     {"overext_reversal_mode": "ema_cross", "overext_ema_span": 20}),
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
                description=f"{name}: champion 1327 + overext reversal {ov}.",
                hypothesis="Reversal-confirmed overext sells AT the turn — beat fire-on-extension 404.3?",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
