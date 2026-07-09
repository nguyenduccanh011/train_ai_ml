"""v16: the entry ML score IS predictive (IC +0.13 vs target, deciles +2.4%->+5.6%) but the
recombine applies it as a PER-SYMBOL z (wastes the cross-sectional signal) and statically (it
inverts in 2024). Test applying it BETTER on the pure-rule base 1776 (435.5):
 - entry_raw_threshold: buy on the ABSOLUTE triple-barrier P(profit) (keeps cross-sectional quality)
 - entry_xs_mom_pct: cross-sectional relative-strength screen (engine note: IC +0.13, WR .52->.74)
If any beats 435.5 robustly, the ML/cross-section adds value when applied right. Multi-seeded.
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

BASE = 1776
GRID = [
    ("n2_v16_raw10", {"entry_raw_threshold": 0.10}),
    ("n2_v16_raw15", {"entry_raw_threshold": 0.15}),
    ("n2_v16_raw20", {"entry_raw_threshold": 0.20}),
    ("n2_v16_raw25", {"entry_raw_threshold": 0.25}),
    ("n2_v16_xsm30", {"entry_xs_mom_pct": 0.30}),
    ("n2_v16_xsm50", {"entry_xs_mom_pct": 0.50}),
    ("n2_v16_raw15_xsm30", {"entry_raw_threshold": 0.15, "entry_xs_mom_pct": 0.30}),
    ("n2_v16_raw20_xsm50", {"entry_raw_threshold": 0.20, "entry_xs_mom_pct": 0.50}),
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
                description=f"v16 apply-ML-right {ov} on pure-rule base {BASE}.",
                hypothesis="Raw-score / cross-sectional ML application captures the +0.13 IC the per-symbol z wastes -> beat 435.5.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
