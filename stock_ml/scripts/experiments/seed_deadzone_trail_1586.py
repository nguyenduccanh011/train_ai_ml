"""Tier-1 dead-zone experiment on champion n2_v5_ox12 (id 1586).
Forensic (full_lifecycle/mfe_deadzone): a trade needs >=16% MFE to finish green 99.7%
of the time, but protection only arms at +15% gain (trailing) / +12% over MA20 (overext)
-> an unprotected MFE +2-12% 'dead zone' where 386 giveback losers (peaked +8.8%, lost
-23.3u) round-trip to a signal-exit loss. Counterfactual: a peak-trail w/ 6-8% band
recovers +35-49u gross. Hypothesis: arming the vol-scaled trail EARLIER + tighter catches
faders WITHOUT clipping runners (trail follows the peak), and EXITS faders sooner so it is
MDD-neutral-to-positive (opposite of the overext 'ride longer' tension). Base champ:
activate 0.15, atr_mult 2.0, atr_floor 0.04, atr_cap 0.16. Backtest vs 407.7.
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
GRID = [
    ("n2_dz_act08", {"trailing_activate_pct": 0.08}),
    ("n2_dz_act05", {"trailing_activate_pct": 0.05}),
    ("n2_dz_act05_m15", {"trailing_activate_pct": 0.05, "trailing_atr_mult": 1.5}),
    ("n2_dz_act05_m15_fl03", {"trailing_activate_pct": 0.05, "trailing_atr_mult": 1.5, "trailing_atr_floor": 0.03}),
    ("n2_dz_act05_m10_fl03_cap08", {"trailing_activate_pct": 0.05, "trailing_atr_mult": 1.0, "trailing_atr_floor": 0.03, "trailing_atr_cap": 0.08}),
    ("n2_dz_act08_m15_fl03", {"trailing_activate_pct": 0.08, "trailing_atr_mult": 1.5, "trailing_atr_floor": 0.03}),
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
        for name, ov in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be); eng.update(ov)
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
                description=f"dead-zone early-arm vol-trail {ov} on champ {BASE}.",
                hypothesis="Arm vol-trail earlier+tighter to plug MFE +2-12% dead zone (recover giveback) -> beat 407.7 w/o MDD cost.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
