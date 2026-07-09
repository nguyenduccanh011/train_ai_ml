"""Exploit the ablation breakthrough: removing incubation (signal_exit_min_age=8 +
incubate_floor) gained +5.5 ROBUST multi-seed (no_incubate=412.3 vs champ 406.8, all 3 seeds).
Base = 1696 (champ - incubation). Now: (a) strip the other ~neutral band-aids (cooldown,
absfloor) for a simpler robust model; (b) RE-TEST levers that failed on the OLD base
(sl10 entry-target, bear-div giveback, early-trail, overext-raise) -- with young trades no
longer protected by incubation, the exit dynamics changed, so they may compose differently now.
Single-seed42 filter; multi-seed the winners vs 412.3.
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

BASE = 1696  # n2_abl_noincub (champ - incubation)
TB = lambda pt, sl, h: {"type": "triple_barrier", "horizon": h, "pt": pt, "sl": sl, "direction": "long"}
# (name, entry_target or None, engine_override or None)
GRID = [
    ("n2_v6_nocool", None, {"reentry_cooldown_bars": 0}),
    ("n2_v6_noabs", None, {"entry_market_abs_floor": None}),
    ("n2_v6_lean", None, {"reentry_cooldown_bars": 0, "entry_market_abs_floor": None}),
    ("n2_v6_sl10", TB(0.15, 0.10, 30), None),
    ("n2_v6_div08", None, {"overext_pct": 0.08, "overext_reversal_mode": "bear_div", "overext_div_threshold": 0.30}),
    ("n2_v6_act08", None, {"trailing_activate_pct": 0.08}),
    ("n2_v6_ox16", None, {"overext_pct": 0.16}),
    ("n2_v6_lean_sl10", TB(0.15, 0.10, 30), {"reentry_cooldown_bars": 0, "entry_market_abs_floor": None}),
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
        for name, etgt, eng_ov in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            eng = copy.deepcopy(be)
            if eng_ov:
                eng.update(eng_ov)
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": es.feature_set_name, "target_config": copy.deepcopy(etgt) if etgt else tc(es)},
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
                description=f"v6 (no-incubation base) entry={etgt} eng={eng_ov} on {BASE}.",
                hypothesis="Build simpler robust model on the no-incubation base + re-test levers that failed on old base -> push past 412.3.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
