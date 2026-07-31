"""Entry-TARGET sweep on champion 1586. Feature swaps barely changed picks (entry head
saturated) and exit feature swaps were exact no-ops (exit head dormant under the rules).
The entry head DOES fire on every trade, so changing its LABEL (target) changes WHICH trades
get picked more than features do. Champion entry target = triple_barrier(h30,pt.15,sl.08).
Sweep pt/sl/horizon (forensic: edge = trades reaching >=16% MFE; tighter sl may select
cleaner names; bigger pt may chase the rotor) + one continuation-regression alt. vs 407.7.
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

BASE = 1586
TB = lambda pt, sl, h: {
    "type": "triple_barrier",
    "horizon": h,
    "pt": pt,
    "sl": sl,
    "direction": "long",
}
GRID = [
    ("n2_et_pt20", TB(0.20, 0.08, 30)),
    ("n2_et_pt12", TB(0.12, 0.08, 30)),
    ("n2_et_sl06", TB(0.15, 0.06, 30)),
    ("n2_et_sl10", TB(0.15, 0.10, 30)),
    ("n2_et_h20", TB(0.15, 0.08, 20)),
    ("n2_et_h40", TB(0.15, 0.08, 40)),
    ("n2_et_pt20_sl06", TB(0.20, 0.06, 30)),
    ("n2_et_pt25_sl08_h40", TB(0.25, 0.08, 40)),
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
        for name, etgt in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            slots = [
                {
                    "slot_type": "entry",
                    "ml_component_id": es.ml_component_id,
                    "rule_component_id": None,
                    "feature_set_name": es.feature_set_name,
                    "target_config": copy.deepcopy(etgt),
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
                engine_config=copy.deepcopy(be),
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"entry-target swap {etgt} on champ {BASE}.",
                hypothesis="Reshape entry label (pt/sl/horizon) -> pick more escape-velocity names -> total_pnl+ at constant trades+Sharpe -> beat 407.7.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
