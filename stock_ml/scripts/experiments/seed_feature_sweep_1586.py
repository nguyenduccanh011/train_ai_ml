"""Feature-axis sweep on champion n2_v5_ox12 (id 1586). Exit-mechanics axis is exhausted
(30 backtests all <= 407.7, frontier-bound). Per the composite formula, the ONLY way to push
the frontier is higher total_pnl + maintained trades + maintained Sharpe = better PREDICTION.
Swap the entry/exit feature SET (retrains the head) to candidate sets that encode this session's
forensic signals (cheapness rank-IC 0.238, xsec +24% IC, recovery dist_low/range_pos, accum/dist
volume = the entry vol-climax IC -0.085, market context) + exit phase/distribution/candle.
Each is tested on the CURRENT champion engine (v5: cooldown/incubation/market-gates). vs 407.7.
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

BASE = 1586
# (name, entry_feature_set or None=keep, exit_feature_set or None=keep)
GRID = [
    ("n2_fs_e_cheap", "entry_lvup126_cheap", None),
    ("n2_fs_e_recov", "entry_lvup126_recov", None),
    ("n2_fs_e_xsec", "entry_lvup126_xsec", None),
    ("n2_fs_e_accdist", "entry_lvup126_accdist", None),
    ("n2_fs_e_mkt", "entry_lvup126_mkt", None),
    ("n2_fs_e_clean", "entry_lvup126_clean", None),
    ("n2_fs_x_phase", None, "exit_vol_phase"),
    ("n2_fs_x_dist", None, "exit_vol_dist"),
    ("n2_fs_x_candle", None, "exit_vol_candle"),
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
        for name, efs, xfs in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": efs or es.feature_set_name, "target_config": tc(es)},
                {"slot_type": "exit", "ml_component_id": xs.ml_component_id, "rule_component_id": None,
                 "feature_set_name": xfs or xs.feature_set_name, "target_config": tc(xs)},
            ]
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(be),
                validation_config=base.validation_config, seed=base.seed,
                description=f"feature-axis swap entry={efs} exit={xfs} on champ {BASE}.",
                hypothesis="Better feature set -> better picks -> total_pnl+ at constant trades+Sharpe -> beat 407.7.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
