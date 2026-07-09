"""A/B: regime-conditional entry/exit on the pure-rule champion n3_rule_champ (t1787, 428.5).
User hypothesis (downtrend_timing_review): in a downtrend the uptrend-tuned slow mechanics
bleed — either enter fast or cut fast. Forensic supported a TIGHTER STOP (asymmetric: helps
downtrend, hurts uptrend) and flagged "enter fast" (skip pullback) as the untested piece.

A  = downtrend tighter hard stop (-0.08 / -0.06), uptrend untouched.
B  = downtrend skip the patient pullback (fill immediately), uptrend keeps pullback.
AB = both.  Deterministic (rule_only_no_ml) -> single seed is exact.
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

BASE = 1787  # n3_rule_champ (428.5)
GRID = [
    ("n3_dtstop08", {"downtrend_hard_stop_pct": -0.08}),
    ("n3_dtstop06", {"downtrend_hard_stop_pct": -0.06}),
    ("n3_dtfast",   {"downtrend_skip_pullback": True}),
    ("n3_dtstop08_fast", {"downtrend_hard_stop_pct": -0.08, "downtrend_skip_pullback": True}),
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
                description=f"regime-conditional timing {ov} on n3_rule_champ {BASE}.",
                hypothesis="Downtrend: cut losers fast / enter fast; uptrend untouched -> push past 428.5.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
