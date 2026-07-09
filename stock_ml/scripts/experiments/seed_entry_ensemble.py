"""Structural test: REGIME-DIVERSIFIED entry via a 2nd entry head on a reversal
(V-bottom) target, unioned into the buy (entry_ensemble infra). The champion is a
momentum/trend model (buys up-legs); a reversal head adds the deep-dip V-bottom
winners it skips. Union buy = capture BOTH styles. Base = champions 1264/1249."""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

REV = {"type": "reversal_entry_regression", "horizon": 10, "penalty": 1.0,
       "dip_window": 50, "min_fwd_rally": 0.12, "trend_window": 0}
GRID = [
    ("n2_champ_rev_z15", 1264, 1.5),
    ("n2_champ_rev_z20", 1264, 2.0),
    ("n2_champ_rev_z25", 1264, 2.5),
    ("n2_mgzu_rev_z20",  1249, 2.0),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        created = []
        for name, base_id, zt in GRID:
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
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be)
            eng["entry_ensemble"] = {"target": copy.deepcopy(REV), "z_threshold": zt}
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"{name}: reversal entry-ensemble z<={zt} base {base_id}.",
                hypothesis="Regime-diversify: union a V-bottom reversal head with the momentum entry.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
