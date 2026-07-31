"""Faded-mid-winner protection: add an EARLY pop-lock trail tier to champion t1378.

Exit-timing forensic on t1378 (the live champion): the signal exit reacts LATE
(lag 7.4 bars from peak, sells 7.6% below peak). Splitting its -32.7u pool by peak
excursion: 723 trades (mfe>=5%) PEAKED +9.3% but realized -0.2% => gave back ~9.6%.
These faded winners peak BELOW the +15% trailing arm and BELOW the +14%-over-SMA20
overext bar, so neither fast exit catches them — they fall through to the slow
signal exit. (The other -31u is mfe<5% DUDS = an entry/incubation problem, already
handled by signal_exit_min_age=8 / incubate_floor=-0.04; exit-speed won't fix those.)

pop_lock arms a protective trail at a LOW pop (+5-6% MFE), classifying a pop "weak"
iff close is below its SMA(ext_window) at the arm bar (already rolling over) and
trailing those tightly — while pops still above their MA keep the loose default trail
and run to the +15% tier. This catches the faded band at its peak without clipping the
strong runners. NOTE: pop_lock was +1.3 (mdd-only) on the OLD t1137; the champion has
since changed a lot (survival entry, incubation, market-gate, overext) and the faded
pool is now precisely quantified — net effect is a genuine open question -> A/B.

Clone t1378, keep EVERYTHING (slots, targets, market-gate, incubation, overext,
trailing) and inject ONLY pop_lock_*. Compare comp vs FRESH t1378 (re-run, not stale).
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

BASE_ID = 1378

# (tag, pop_lock overrides) — arm an early protective trail at a low pop, ext-gated.
GRID = [
    (
        "pl_a05_t05",
        {
            "pop_lock_arm_pct": 0.05,
            "pop_lock_trail_pct": 0.05,
            "pop_lock_ext_window": 10,
            "pop_lock_ext_threshold": 0.0,
        },
    ),
    (
        "pl_a06_t05",
        {
            "pop_lock_arm_pct": 0.06,
            "pop_lock_trail_pct": 0.05,
            "pop_lock_ext_window": 10,
            "pop_lock_ext_threshold": 0.0,
        },
    ),
    (
        "pl_a05_t06",
        {
            "pop_lock_arm_pct": 0.05,
            "pop_lock_trail_pct": 0.06,
            "pop_lock_ext_window": 10,
            "pop_lock_ext_threshold": 0.0,
        },
    ),
    # only lock the WEAKEST pops (already >2% below MA10) -> let more run, less clip
    (
        "pl_a06_t06_xn2",
        {
            "pop_lock_arm_pct": 0.06,
            "pop_lock_trail_pct": 0.06,
            "pop_lock_ext_window": 10,
            "pop_lock_ext_threshold": -0.02,
        },
    ),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        if base is None:
            raise ValueError(f"base template id={BASE_ID} not found")
        slots = [s for s in base.component_slots]

        def _tc(slot):
            tc = slot.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

        base_ec = base.engine_config
        base_ec = json.loads(base_ec) if isinstance(base_ec, str) else copy.deepcopy(base_ec)

        created = []
        for tag, ov in GRID:
            name = f"n2_1378_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id}), skip")
                created.append((ex.id, name))
                continue
            ec = copy.deepcopy(base_ec)
            ec.update(ov)
            new_slots = [
                {
                    "slot_type": s.slot_type,
                    "ml_component_id": s.ml_component_id,
                    "rule_component_id": s.rule_component_id,
                    "feature_set_name": s.feature_set_name,
                    "target_config": _tc(s),
                }
                for s in slots
            ]
            tmpl = await repo.create(
                name=name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=new_slots,
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=ec,
                validation_config=base.validation_config,
                seed=base.seed,
                description=(
                    f"Faded-mid-winner pop-lock {tag} on t1378: early protective trail "
                    f"arm={ov['pop_lock_arm_pct']} trail={ov['pop_lock_trail_pct']} "
                    f"ext_thr={ov['pop_lock_ext_threshold']}; everything else = t1378."
                ),
                hypothesis="723 signal-exit trades peak +9.3% then give back ~9.6% (lag 7.4 bars), "
                "falling below both the +15% trail and +14% overext. An ext-gated early "
                "trail at +5-6% should lock the faded band near its peak without clipping "
                "strong runners. Test vs fresh t1378.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} created (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(tid) for tid, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
