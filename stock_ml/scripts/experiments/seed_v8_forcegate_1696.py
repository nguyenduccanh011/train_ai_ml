"""v8: re-tune the FORCE-GATES on the new champion base 1696 (no_incubate, 412.3).
The hard_stop no-op proved the force-gates (downleg deep-reversal backstop + nonbull
belowma) are the real loss-cutters. They're also the highest-impact REAL levers (nonbull
-21, downleg essential). Sweep their sensitivity: downleg reversal depth, nonbull SMA/persist/
regime-window. Multi-seeded directly.
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

BASE = 1696
GRID = [
    ("n2_v8_dl10", {"exit_force_gate": "downleg10"}),
    ("n2_v8_dl14", {"exit_force_gate": "downleg14"}),
    ("n2_v8_dlvol12", {"exit_force_gate": "dlvol12"}),
    ("n2_v8_nb_p2", {"exit_force_gate_nonbull": "belowma20p2"}),
    ("n2_v8_nb_p4", {"exit_force_gate_nonbull": "belowma20p4"}),
    ("n2_v8_nb_ma40", {"nonbull_ma_win": 40}),
    ("n2_v8_nb_ma60", {"nonbull_ma_win": 60}),
    ("n2_v8_nb30p3", {"exit_force_gate_nonbull": "belowma30p3"}),
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
                description=f"v8 force-gate re-tune {ov} on no_incubate base {BASE}.",
                hypothesis="Tune the real loss-cutting force-gates on the new base -> push past 412.3 robust.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
