"""Regime-conditional overext: skip the top-sell while the SMA is rising fast
(strong uptrend -> let runners run), fire only in flat/choppy extension. Aims to
recover the 2021 bull-year pnl that plain overext clips. Base 1204 (ox14 = 393.8).
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

OX = {
    "overext_ma_window": 20,
    "overext_pct": 0.14,
    "overext_reversal": False,
    "exit_priority": ["trailing_stop", "overext", "signal"],
}
# (name, base_id, skip_slope_pct, lookback)
GRID = [
    ("n2_1204_ox14_sk02", 1204, 0.02, 5),
    ("n2_1204_ox14_sk04", 1204, 0.04, 5),
    ("n2_1204_ox14_sk06", 1204, 0.06, 5),
    ("n2_1204_ox14_sk03l10", 1204, 0.03, 10),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        created = []
        for name, base_id, sk, lb in GRID:
            base = await repo.get_by_id(base_id)
            es = next(s for s in base.component_slots if s.slot_type == "entry")
            xs = next(s for s in base.component_slots if s.slot_type == "exit")

            def tc(sl):
                t = sl.target_config
                return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)

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
            be = base.engine_config
            be = json.loads(be) if isinstance(be, str) else be
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            eng.update(OX)
            eng["overext_skip_ma_slope_pct"] = sk
            eng["overext_skip_lookback"] = lb
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
                description=f"Regime-conditional overext skip slope {sk} lb{lb}, base {base_id}.",
                hypothesis="Skip overext in strong uptrend -> recover bull pnl, keep chop gains, beat 393.8.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
