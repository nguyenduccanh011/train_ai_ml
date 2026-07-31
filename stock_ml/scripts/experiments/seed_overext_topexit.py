"""Sell-the-top experiment on age-champ base 1204 (n2_1187_age8_f04).

Behavioral audit (project_graded_risk_exit_gate): the exit head is PHASE-INVERTED
— flat +0.12z at wave-tops (blind), peaks +0.36z at bottoms (panic). ext_ma20
peaks AT tops (+1.15z) and leads the drop (IC -0.16). Add an over-extension top
exit (new engine rule 'overext', OUTSIDE the age/market gates) and sweep; plus one
cross-check using the existing reversal-confirm top_reversal_exit machinery.
Bar to beat: 1204 (comp 389.7, pnl 99.8, mdd 3.94).
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

BASE_ID = 1204


# (name, engine-mutator)
def _ox(pct, reversal, win=20):
    def f(eng):
        eng["overext_ma_window"] = win
        eng["overext_pct"] = pct
        eng["overext_reversal"] = reversal
        eng["exit_priority"] = ["trailing_stop", "overext", "signal"]

    return f


def _topx(near, run, confirms):
    def f(eng):
        eng["top_reversal_exit"] = {"near": near, "run": run, "min_confirms": confirms}

    return f


GRID = [
    ("n2_1204_ox20_10r", _ox(0.10, True), "overext ma20 pct0.10 +reversal"),
    ("n2_1204_ox20_12r", _ox(0.12, True), "overext ma20 pct0.12 +reversal"),
    ("n2_1204_ox20_15r", _ox(0.15, True), "overext ma20 pct0.15 +reversal"),
    (
        "n2_1204_ox20_12",
        _ox(0.12, False),
        "overext ma20 pct0.12 no-reversal (sell every extended bar)",
    ),
    (
        "n2_1204_topx_n3r15",
        _topx(0.03, 0.15, 2),
        "reuse top_reversal_exit near0.03 run0.15 confirms2",
    ),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        entry_slot = next(s for s in base.component_slots if s.slot_type == "entry")
        exit_slot = next(s for s in base.component_slots if s.slot_type == "exit")

        def _tc(slot):
            tc = slot.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

        slots_def = [
            {
                "slot_type": "entry",
                "ml_component_id": entry_slot.ml_component_id,
                "rule_component_id": None,
                "feature_set_name": entry_slot.feature_set_name,
                "target_config": _tc(entry_slot),
            },
            {
                "slot_type": "exit",
                "ml_component_id": exit_slot.ml_component_id,
                "rule_component_id": None,
                "feature_set_name": exit_slot.feature_set_name,
                "target_config": _tc(exit_slot),
            },
        ]
        base_engine = base.engine_config
        if isinstance(base_engine, str):
            base_engine = json.loads(base_engine)

        created = []
        for name, mutate, desc in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(base_engine)
            mutate(eng)
            tmpl = await repo.create(
                name=name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=copy.deepcopy(slots_def),
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=eng,
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"Sell-the-top: {desc}. Base 1204. project_graded_risk_exit_gate.",
                hypothesis="Exit head blind at tops; extension-above-MA leads the drop. "
                "Add top-exit OUTSIDE age/market gates -> mdd down + pnl up vs 1204 (389.7).",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
