"""Batch 2 — the signal-exit head is directionless (corr fwd_ret +0.02) and its -33u bleed
is irreducible by any context gate (premature/correct are structurally indistinguishable).
So test the only remaining cheap lever: SUPPRESS it harder and let trailing+overext+downleg
carry the exits. If composite rises, the ML exit head is net dead-weight; if it falls, its
exits provide occupancy/turnover value and the lever is elsewhere (target rebuild).

Variants on champion 1327 (exit_threshold 0.07, signal_exit_min_age 8):
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
# (name, exit_threshold_override, engine_overrides)
GRID = [
    ("n2_af04_xthr03", 0.3, {}),
    ("n2_af04_xthr07", 0.7, {}),
    ("n2_af04_xthr15", 1.5, {}),
    ("n2_af04_sigoff", None, {"signal_exit_enabled": False}),
    ("n2_af04_minage14", None, {"signal_exit_min_age": 14}),
    ("n2_af04_minage20_fl06", None, {"signal_exit_min_age": 20, "signal_exit_incubate_floor": -0.06}),
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
        for name, xthr, ov in GRID:
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
                entry_threshold=base.entry_threshold,
                exit_threshold=(xthr if xthr is not None else base.exit_threshold),
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"suppress signal-exit: xthr={xthr} {ov}; base {BASE}.",
                hypothesis="Directionless exit head is dead-weight (-33u). Suppress -> defer to trailing/overext/downleg.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
