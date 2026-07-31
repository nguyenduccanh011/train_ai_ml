"""Exit-head retrain loop, batch 1 — clone champion 1789, vary ONLY the exit slot along two
axes: features {directional-only, vol+phase} x target {forward_drawdown, risk_exit} (both
correctly SELL-signed: HIGH=time to exit). Goal: an exit head that is directionally correct and
beats the hard rule under the ML-only (un-masked) eval. Clean: bounded h10 targets, entry h30
already forces gap>=65 so no leakage.
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

BASE_ID = 1789
FDD = {"type": "forward_drawdown_regression", "horizon": 10}
REXIT = {"type": "risk_exit_regression", "horizon": 10}
# (tag, exit_feature_set, exit_target)
GRID = [
    ("xdir_fdd", "exit_directional", FDD),
    ("xdir_rex", "exit_directional", REXIT),
    ("xphase_fdd", "exit_vol_phase", FDD),
    ("xphase_rex", "exit_vol_phase", REXIT),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE_ID)
        base_ec = base.engine_config
        base_ec = json.loads(base_ec) if isinstance(base_ec, str) else copy.deepcopy(base_ec)

        def _tc(s):
            tc = s.target_config
            return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

        created = []
        for tag, fs, xt in GRID:
            name = f"n2_xr_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} (id={ex.id})")
                created.append((ex.id, name))
                continue
            slots = []
            for s in base.component_slots:
                tc = _tc(s)
                fsn = s.feature_set_name
                if s.slot_type == "exit":
                    fsn = fs
                    tc = copy.deepcopy(xt)
                slots.append(
                    {
                        "slot_type": s.slot_type,
                        "ml_component_id": s.ml_component_id,
                        "rule_component_id": s.rule_component_id,
                        "feature_set_name": fsn,
                        "target_config": tc,
                    }
                )
            tmpl = await repo.create(
                name=name,
                market=base.market,
                strategy=base.strategy,
                feature_set_id=base.feature_set_id,
                target_id=base.target_id,
                component_slots=slots,
                direction=base.direction,
                signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold,
                split_config=base.split_config,
                engine_config=copy.deepcopy(base_ec),
                validation_config=base.validation_config,
                seed=base.seed,
                description=f"Exit retrain {tag}: exit={fs} target={xt['type']} on champion 1789.",
                hypothesis="Directionally-signed exit target + trend/structure features -> a head that "
                "sells near TOPS and provides tail control under ML-only (un-masked) eval.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} (id={tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
