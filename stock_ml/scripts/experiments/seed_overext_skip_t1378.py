"""Batch A on CURRENT champion t1378 (n2_ent_uplg_am20, comp 405.0) — capture the runner
continuation overext throws away. Forensics (entry_exit_deep_forensics): 76% of overext sells
CONTINUE +14.8% over the next 20d — overext fires at a FIXED +14% extension regardless of
trend strength, so it sells strong uptrends that keep extending. overext_skip_ma_slope_pct
SKIPS the overext sell while the SMA(20) is rising >= slope% over the lookback (strong trend ->
let the runner run); fires overext only in flat/choppy extension. Champion has it UNSET.
Sweep slope x lookback; backtest vs 405.0.
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
    ("n2_am20_oxsk02_l5", 0.02, 5),
    ("n2_am20_oxsk03_l5", 0.03, 5),
    ("n2_am20_oxsk04_l5", 0.04, 5),
    ("n2_am20_oxsk05_l5", 0.05, 5),
    ("n2_am20_oxsk03_l10", 0.03, 10),
    ("n2_am20_oxsk04_l10", 0.04, 10),
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
        for name, sk, lb in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be)
            eng["overext_skip_ma_slope_pct"] = sk
            eng["overext_skip_lookback"] = lb
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
                description=f"overext_skip slope {sk} lb{lb} on champ {BASE}.",
                hypothesis="Let strong-uptrend runners run (76% of overext sells continue +14.8%) -> beat 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
