"""Batch A2 on champion t1378 — resolve the X1/X3 tension. overext HARD-sells at +14% ext but
76% of those names CONTINUE +14.8% (sold too early); yet 33% of winners give back 8.3% from a
bear-div top (selling later risks giveback). Resolution: overext_trail_pct converts the hard
overext sell into a TIGHT trailing ride (overext_armed) — let the runner run but exit on a tight
give-back band, capturing continuation while capping the drawdown. Champion has it UNSET (hard sell).
Sweep the tight band; also one combo with a light skip. Backtest vs 405.0.
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

BASE = 1378
GRID = [
    ("n2_am20_oxt03", {"overext_trail_pct": 0.03}),
    ("n2_am20_oxt04", {"overext_trail_pct": 0.04}),
    ("n2_am20_oxt05", {"overext_trail_pct": 0.05}),
    ("n2_am20_oxt06", {"overext_trail_pct": 0.06}),
    (
        "n2_am20_oxt04_sk03",
        {"overext_trail_pct": 0.04, "overext_skip_ma_slope_pct": 0.03, "overext_skip_lookback": 5},
    ),
    (
        "n2_am20_oxt05_sk04",
        {"overext_trail_pct": 0.05, "overext_skip_ma_slope_pct": 0.04, "overext_skip_lookback": 5},
    ),
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
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            eng.update(ov)
            slots = [
                {
                    "slot_type": "entry",
                    "ml_component_id": es.ml_component_id,
                    "rule_component_id": None,
                    "feature_set_name": es.feature_set_name,
                    "target_config": tc(es),
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
                engine_config=eng,
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"overext->tight-trail ride {ov} on champ {BASE}.",
                hypothesis="Ride overext runners with a tight band (capture +14.8% continuation, cap giveback) -> beat 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
