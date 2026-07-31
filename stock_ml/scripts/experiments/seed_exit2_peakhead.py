"""2nd EXIT HEAD ensemble — learned sell-the-top, the root-target fix.

Diagnosis (project_graded_risk_exit_gate): the reward_risk exit head is phase-
inverted (flat at tops, peaks at bottoms) because its target rewards downside
MAGNITUDE. Add a 2nd exit head on a zigzag PEAK target, one_sided='pre' (labels
the approach to a confirmed top -> fires AT the top, the proven sell-at-peak
target, project_exit_sell_at_peak). Unioned into the SELL via exit_ensemble +
exit2_z_threshold (infra at experiment.py:703/1792). gap=85 already -> leak-safe.
Goal: a SELECTIVE learned top-sell (real tops only, lets runners run) that beats
the BLUNT overext profit-take champion (1217 = 393.8) and recovers bull-year pnl.
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

PEAK = {
    "type": "zigzag_pivot",
    "direction": "peak",
    "pct": 0.06,
    "min_leg_bars": 3,
    "tau": 5.0,
    "one_sided": "pre",
}
# (name, base_template_id, z_threshold)
GRID = [
    ("n2_1204_x2peak_z15", 1204, 1.5),
    ("n2_1204_x2peak_z20", 1204, 2.0),
    ("n2_1204_x2peak_z25", 1204, 2.5),
    ("n2_1217_x2peak_z20", 1217, 2.0),  # stack on overext-0.14 champ
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
                print(f"= {name} exists ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            eng["exit_ensemble"] = {"target": copy.deepcopy(PEAK), "z_threshold": zt}
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
                description=f"2nd exit head zigzag-peak pre pct0.06, z<={zt}, base {base_id}. project_graded_risk_exit_gate.",
                hypothesis="Learned sell-at-top head fires AT the top (vs reward_risk lagging/blind); "
                "selective -> beats blunt overext 393.8 + recovers bull pnl.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
