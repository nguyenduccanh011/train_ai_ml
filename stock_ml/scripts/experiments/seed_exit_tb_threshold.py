"""ROOT-CAUSE fix for the directionless exit head. The exit head was re-targeted to a path-aware
triple_barrier-SHORT label (P(drop pt before rise sl) — peaks at real tops, LOW at bounceable
dips) in t1450(tb08)/t1451(tb10), BUT exit_threshold stayed 0.07 — far below the [0,1] P scale,
so the recombine (SELL when pred_exit>thr) fires SELL on almost every bar → the head's
prediction is IGNORED (identical to champion).

Fix: RAISE exit_threshold so SELL fires ONLY when P(drop) is genuinely high. Then a bounceable
dip (low P) is NOT sold (keep position through the bounce, no premature sell, no rebuy) and a
real top (high P) IS sold. Clone the trained tb-short exit heads (no retrain) and sweep the
threshold. A/B vs champion 405.0.
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

# (base_template_with_trained_tb_exit, tag, exit_threshold)
GRID = [
    (1450, "tb08_xt25", 0.25), (1450, "tb08_xt35", 0.35), (1450, "tb08_xt45", 0.45),
    (1450, "tb08_xt55", 0.55), (1450, "tb08_xt65", 0.65),
    (1451, "tb10_xt40", 0.40), (1451, "tb10_xt55", 0.55),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        created = []
        for base_id, tag, xt in GRID:
            base = await repo.get_by_id(base_id)
            name = f"n2_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue

            def _tc(s):
                tc = s.target_config
                return json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)

            be = base.engine_config
            be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
            slots = [{"slot_type": s.slot_type, "ml_component_id": s.ml_component_id,
                      "rule_component_id": s.rule_component_id, "feature_set_name": s.feature_set_name,
                      "target_config": _tc(s)} for s in base.component_slots]
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=slots, direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=xt,   # <-- the recalibration
                split_config=base.split_config, engine_config=copy.deepcopy(be),
                validation_config=base.validation_config, seed=base.seed,
                description=f"tb-short exit head (from {base_id}) + RECALIBRATED exit_threshold={xt}.",
                hypothesis="High exit_threshold makes the path-aware tb-short head actually gate exits: "
                           "sell only high-P(drop) tops, hold bounceable dips. Beat 405.0.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})  xt={xt}"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
