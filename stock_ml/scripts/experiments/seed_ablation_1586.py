"""Multi-seed ABLATION of champion 1586's lever stack. Hypothesis: the champion is ~10 levers
each justified by a noisy single-seed42 +2-3 composite -> some are SEED-42 OVERFIT, not real.
Remove ONE lever per template; each is then multi-seed evaluated (separate script). A lever
whose removal does NOT lower the multi-seed MEAN (or raises it) is overfit -> drop it for a
simpler, more robust model. Suspect #1: reentry_cooldown (user previously rejected as band-aid).
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
# (name, engine_override removing one lever)
GRID = [
    ("n2_abl_nocool", {"reentry_cooldown_bars": 0}),
    ("n2_abl_noincub", {"signal_exit_min_age": 0, "signal_exit_incubate_floor": None}),
    ("n2_abl_nononbull", {"exit_force_gate_nonbull": None}),
    ("n2_abl_noexitmkt", {"exit_market_gate_enabled": False}),
    ("n2_abl_nopullback", {"entry_pullback_pct": None, "entry_pullback_window": 0}),
    ("n2_abl_noentrymkt", {"entry_market_gate_enabled": False}),
    ("n2_abl_noupleggate", {"entry_gate": None}),
    ("n2_abl_notrailactiv", {"trailing_activate_pct": 0.0}),
    ("n2_abl_noatrtrail", {"trailing_atr_mult": None}),
    ("n2_abl_noabsfloor", {"entry_market_abs_floor": None}),
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
        for name, eng_ov in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                created.append((ex.id, name))
                continue
            eng = copy.deepcopy(be)
            eng.update(eng_ov)
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
                description=f"ABLATION remove {eng_ov} from champ {BASE}.",
                hypothesis="If multi-seed mean doesn't drop, this lever is seed-42 overfit -> simpler robust model.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id})")
            created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
