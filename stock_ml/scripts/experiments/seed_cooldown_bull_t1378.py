"""Batch C on champion t1378 — two WIRED, off-by-default, untested-on-champion levers that
target documented leaks WITHOUT holding longer (avoiding the mdd trap that sank skip/trail):

 reentry_cooldown_bars: after a LOSING exit on a symbol, block a new buy there for N bars.
   Fixes the whipsaw the user asked about — sold (often a premature signal exit) then rebought
   the SAME name ~6% higher within 15 bars (60.9% of the time, 25.5u give-up). Cutting those
   low-quality rebuys should hold/raise pnl while lowering churn — a quality/risk lever.

 overext_skip_bull_enabled: skip the overext top-sell in a strong MARKET bull (let runners run
   with the tape). Market-level (distinct from the per-symbol overext_skip that tested negative).
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
    ("n2_am20_cd03", {"reentry_cooldown_bars": 3}),
    ("n2_am20_cd05", {"reentry_cooldown_bars": 5}),
    ("n2_am20_cd08", {"reentry_cooldown_bars": 8}),
    ("n2_am20_cd12", {"reentry_cooldown_bars": 12}),
    ("n2_am20_cd20", {"reentry_cooldown_bars": 20}),
    (
        "n2_am20_bull10",
        {
            "overext_skip_bull_enabled": True,
            "overext_bull_window": 5,
            "overext_bull_threshold": 1.0,
            "overext_bull_mode": "zscore",
            "overext_bull_z_lookback": 60,
        },
    ),
    (
        "n2_am20_bull05",
        {
            "overext_skip_bull_enabled": True,
            "overext_bull_window": 5,
            "overext_bull_threshold": 0.5,
            "overext_bull_mode": "zscore",
            "overext_bull_z_lookback": 60,
        },
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
                description=f"{ov} on champ {BASE}.",
                hypothesis="Cut whipsaw rebuys / skip overext in market bull -> beat 405.0 without holding longer.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
