"""Refinement: 2nd exit head on MAJOR-top zigzag (higher pct = real tops only,
not minor pullbacks) + min_fwd_leg (the peak must be followed by a real drop).
The pct=0.06 head sold at every minor swing-high (clips uptrends). Base 1217
(overext-0.14 champ) so the peak head only needs to add the tops overext misses.
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

def peak(pct, mfl):
    return {"type": "zigzag_pivot", "direction": "peak", "pct": pct,
            "min_leg_bars": 3, "tau": 5.0, "one_sided": "pre", "min_fwd_leg": mfl}
# (name, base, target, z)
GRID = [
    ("n2_1204_x2pk_p10_z25", 1204, peak(0.10, 0.0),  2.5),
    ("n2_1204_x2pk_p12_z25", 1204, peak(0.12, 0.0),  2.5),
    ("n2_1204_x2pk_p10m08_z20", 1204, peak(0.10, 0.08), 2.0),
    ("n2_1217_x2pk_p12_z25", 1217, peak(0.12, 0.0),  2.5),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        created = []
        for name, base_id, tgt, zt in GRID:
            base = await repo.get_by_id(base_id)
            es = next(s for s in base.component_slots if s.slot_type == "entry")
            xs = next(s for s in base.component_slots if s.slot_type == "exit")
            def tc(sl):
                t = sl.target_config
                return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": es.feature_set_name, "target_config": tc(es)},
                {"slot_type": "exit", "ml_component_id": xs.ml_component_id, "rule_component_id": None,
                 "feature_set_name": xs.feature_set_name, "target_config": tc(xs)},
            ]
            be = base.engine_config
            be = json.loads(be) if isinstance(be, str) else be
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be)
            eng["exit_ensemble"] = {"target": copy.deepcopy(tgt), "z_threshold": zt}
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"2nd exit head MAJOR-top {tgt['pct']} mfl{tgt['min_fwd_leg']} z<={zt} base {base_id}.",
                hypothesis="Major-tops-only peak head -> selective sell-at-top that beats blunt overext 393.8.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
