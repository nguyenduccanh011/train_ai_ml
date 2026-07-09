"""Conditional force-gate suppression on champion t1378 (n2_ent_uplg_am20, comp 405.0).

ROOT CAUSE (forcegate_forensic.py): belowma20p3 = 84% of signal-exits; 41% fire while
ABOVE MA50 (healthy wave-pullback, bounce +5.9%/10d 62%, hold-instead +5.8u) — the
churn AND the wave-cap. Blanket belowma50 failed (pnl 96->84) because it ALSO stops
cutting the below-MA50 breakdowns. Fix = suppress the SOFT force-sell only in a healthy
context (above MA50 / RSI>K); downleg12 backstop still fires. New token exit_force_suppress.
Plus a config-only control via the pre-existing mabreak20p3m50 token (sell only if below
MA20 x3 AND below MA50 = breakdown-only).
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

BASE_ID = 1378


def s_ma50(e): e["exit_force_suppress"] = "abovema50"
def s_ma50r40(e): e["exit_force_suppress"] = "abovema50_rsi40"
def s_ma30(e): e["exit_force_suppress"] = "abovema30"
def s_rsi45(e): e["exit_force_suppress"] = "rsi45"
def s_mabrk(e): e["exit_force_gate"] = "downleg12_mabreak20p3m50"  # config-only control

GRID = [
    ("n2_xfs_ma50",    s_ma50,    "suppress belowma20p3 sell when close>MA50 (healthy pullback)"),
    ("n2_xfs_ma50r40", s_ma50r40, "suppress when close>MA50 AND RSI14>40"),
    ("n2_xfs_ma30",    s_ma30,    "suppress when close>MA30 (looser context)"),
    ("n2_xfs_rsi45",   s_rsi45,   "suppress when RSI14>45 (not breaking down)"),
    ("n2_mabrk_m50",   s_mabrk,   "config-only: replace belowma20p3 with mabreak20p3m50"),
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
            {"slot_type": "entry", "ml_component_id": entry_slot.ml_component_id,
             "rule_component_id": entry_slot.rule_component_id,
             "feature_set_name": entry_slot.feature_set_name, "target_config": _tc(entry_slot)},
            {"slot_type": "exit", "ml_component_id": exit_slot.ml_component_id,
             "rule_component_id": exit_slot.rule_component_id,
             "feature_set_name": exit_slot.feature_set_name, "target_config": _tc(exit_slot)},
        ]
        base_engine = base.engine_config
        if isinstance(base_engine, str):
            base_engine = json.loads(base_engine)

        created = []
        for name, mutate, desc in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} exists (id={ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(base_engine); mutate(eng)
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots_def), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"Force-gate conditional suppress: {desc}.",
                hypothesis="belowma20p3 dumps bounceable wave-pullbacks (above MA50); suppress "
                           "those, keep below-MA50 breakdown cuts -> +pnl, fewer churn, lower mdd.",
                universe_slug=base.universe_slug, model_mode=base.model_mode,
            )
            print(f"* {name} created (id={tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("\nIDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
