"""The champion keeps the ML exit head OFF (signal_threshold 5.5), so exit-feature changes
are no-ops. To actually test whether distribution-day / candle features make the exit head
USEFUL, re-enable the signal exit (signal_threshold 2.0) and A/B the exit feature set:
  base   = exit_vol_market   (control, head ON)
  dist   = exit_vol_dist     (+ distribution-day pressure)
  candle = exit_vol_candle   (+ bearish engulfing + doji)
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

BASE_ID = 1327
SIG_THR = 2.0
# (name, exit_feature_set)
GRID = [
    ("n2_exiton_base",   "exit_vol_market"),
    ("n2_exiton_dist",   "exit_vol_dist"),
    ("n2_exiton_candle", "exit_vol_candle"),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        es = next(s for s in base.component_slots if s.slot_type == "entry")
        xs = next(s for s in base.component_slots if s.slot_type == "exit")

        def tc(sl):
            t = sl.target_config
            return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)

        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else be
        created = []
        for name, exit_fs in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": es.feature_set_name, "target_config": tc(es)},
                {"slot_type": "exit", "ml_component_id": xs.ml_component_id, "rule_component_id": None,
                 "feature_set_name": exit_fs, "target_config": tc(xs)},
            ]
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=SIG_THR,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(be),
                validation_config=base.validation_config, seed=base.seed,
                description=f"{name}: champion 1327, exit head ON (sig_thr {SIG_THR}), exit={exit_fs}.",
                hypothesis="With the exit head active, do distribution/candle features improve downside detection?",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
