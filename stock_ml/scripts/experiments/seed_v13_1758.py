"""v13: base = 1758 (pb04_w50, multiseed 432.2). Confirm the pullback peak (pct 0.045, window
55/60) and RE-TEST the levers that gained earlier on this much-higher base (the pullback changed
which trades fill, so defensive/entry levers may re-compose). Multi-seeded.
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

BASE = 1758  # pb04_w50
TB = lambda pt, sl, h: {
    "type": "triple_barrier",
    "horizon": h,
    "pt": pt,
    "sl": sl,
    "direction": "long",
}
GRID = [
    ("n2_v13_pb045_w50", None, {"entry_pullback_pct": 0.045}),
    ("n2_v13_pb04_w55", None, {"entry_pullback_window": 55}),
    ("n2_v13_pb04_w60", None, {"entry_pullback_window": 60}),
    ("n2_v13_pb05_w50", None, {"entry_pullback_pct": 0.05}),
    ("n2_v13_nbma35", None, {"nonbull_ma_win": 35}),
    ("n2_v13_emt08", None, {"entry_market_threshold": -0.8}),
    ("n2_v13_sl10", TB(0.15, 0.10, 30), None),
    (
        "n2_v13_div08",
        None,
        {"overext_pct": 0.08, "overext_reversal_mode": "bear_div", "overext_div_threshold": 0.30},
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
        for name, etgt, eng_ov in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            if eng_ov:
                eng.update(eng_ov)
            slots = [
                {
                    "slot_type": "entry",
                    "ml_component_id": es.ml_component_id,
                    "rule_component_id": None,
                    "feature_set_name": es.feature_set_name,
                    "target_config": copy.deepcopy(etgt) if etgt else tc(es),
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
                description=f"v13 peak+re-test entry={etgt} eng={eng_ov} on base {BASE}.",
                hypothesis="Confirm pullback peak + re-test levers on the 432 base.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
