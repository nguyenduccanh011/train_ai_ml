"""Multi-seed confirm rv_z07 vs base xh_skip300w. Clone seed-named templates (distinct run_ids so
NAV can score each), run at their seed. Compare mean CAGR across seeds 7/99 (42 already on board).
"""

from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

# (new_name, base_template_id, seed)
JOBS = [
    ("base_xh300_s7", 3064, 7),
    ("base_xh300_s99", 3064, 99),
    ("rvz07_s7", 3085, 7),
    ("rvz07_s99", 3085, 99),
]


async def make_all():
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        for new_name, base_id, seed in JOBS:
            ex = await repo.get_by_name(new_name)
            if ex:
                print(f"exists {new_name} id={ex.id}")
                ids[new_name] = (ex.id, seed)
                continue
            base = await repo.get_by_id(base_id)
            slots = []
            for sl in base.component_slots:
                tc = sl.target_config
                tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
                slots.append(
                    {
                        "slot_type": sl.slot_type,
                        "ml_component_id": sl.ml_component_id,
                        "rule_component_id": sl.rule_component_id,
                        "feature_set_name": sl.feature_set_name,
                        "target_config": tc,
                    }
                )
            eng = base.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else dict(eng)
            t = await repo.create(
                name=new_name,
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
                engine_config=copy.deepcopy(eng),
                validation_config=base.validation_config,
                seed=seed,
                description=f"multiseed confirm clone of {base_id} at seed {seed}",
                hypothesis="seed-robustness of rv_z07 vs base",
                universe_slug=base.universe_slug,
                model_mode=base.model_mode,
            )
            await s.commit()
            print(f"created {new_name} id={t.id} seed={seed}")
            ids[new_name] = (t.id, seed)
    return ids


def main():
    ids = asyncio.run(make_all())
    asyncio.run(async_engine.dispose())
    for name, (tid, seed) in ids.items():
        r = run_template_experiment(template_id=tid, seed=seed)
        print(f"RAN {name} t{tid} seed={seed} -> {r.get('run_id')}", flush=True)
    print("MULTISEED_REVTILT_DONE")


if __name__ == "__main__":
    main()
