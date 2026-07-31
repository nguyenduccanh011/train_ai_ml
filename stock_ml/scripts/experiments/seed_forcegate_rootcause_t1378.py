"""ROOT-CAUSE fix (real one). Discovery: the ML exit head is DECORATIVE — exit_threshold sweeps
(0.07..0.65) AND exit-target swaps (reward_risk -> triple_barrier-short) ALL give byte-identical
results (405.0, 2385 trades). Exits are 100% driven by the MECHANICAL force-gate
exit_force_gate="downleg12_belowma20p3" (downleg12 OR belowma20p3) + overext + trailing.

The premature-sell-then-bounce (root cause of the rebuy) = belowma20p3 force-selling a SHALLOW
3-bar-below-MA20 dip that bounces. Fix at the SOURCE: make the force confirmation more ROBUST /
SELECTIVE so it only fires on a real breakdown, not a bounceable dip:
  - belowma20p5  : require 5 consecutive bars below MA20 (veto the 3-4 bar shakeout)
  - belowma50p3  : below the 50-SMA = a real trend break (hold through above-MA50 shakeouts)
  - bear3 / bear3p2 : confirmed bearish (MACD<0 AND close<MA20 AND red bar) — far more selective
  - downleg15    : deeper reversal before forcing
A/B vs champion 405.0. (Keeps downleg12 unless noted; only the shallow belowma token is hardened.)
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
    ("n2_am20_g_bma20p5", "downleg12_belowma20p5"),
    ("n2_am20_g_bma50p3", "downleg12_belowma50p3"),
    ("n2_am20_g_bma50", "downleg12_belowma50"),
    ("n2_am20_g_bear3p2", "downleg12_bear3p2"),
    ("n2_am20_g_bear3", "downleg12_bear3"),
    ("n2_am20_g_dl15bma20p3", "downleg15_belowma20p3"),
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
        for name, gate in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            eng["exit_force_gate"] = gate
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
                description=f"harden force-gate -> {gate} on champ {BASE} (root-cause: belowma20p3 sells bounceable dips).",
                hypothesis="More selective/robust force confirmation stops force-selling bounceable shallow dips -> beat 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})  gate={gate}")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
