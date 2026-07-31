"""Distribution-aware EXIT retrain (user 2026-06-18, "retrain head exit phân phối"). Clone champion 1930,
swap the EXIT slot feature_set to exit_vol_dist2 (= exit_vol_market + consolidation_score + macd_hist_chg_5
+ dist_day_25, the 3 signals that POSITIVELY separate real tops from premature SOLD_THEN_RAN exits). Two
variants: feature-only, and feature + cons2 gate (synergy). Compare vs consgate2 (490.5, gate-only) and
champ (489.0) to see if feeding the distribution signal to the exit HEAD (not just gating) adds.
"""

from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

BASE_ID = 1930
EXIT_FS = "exit_vol_dist2"
VARIANTS = [
    (
        "n2_distexit",
        None,
        "Champion 1930 + distribution-aware exit feature set (exit_vol_dist2), no gate.",
    ),
    (
        "n2_distexit_g",
        "cons2",
        "Champion 1930 + exit_vol_dist2 + consolidation exit-gate (feature+gate).",
    ),
]


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    ids = {}
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_ID)
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
        tc = lambda sl: (
            json.loads(sl.target_config)
            if isinstance(sl.target_config, str)
            else copy.deepcopy(sl.target_config)
        )
        for name, gate, desc in VARIANTS:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})")
                ids[name] = ex.id
                continue
            eng = copy.deepcopy(be)
            if gate:
                eng["exit_gate"] = gate
            slots = []
            for sl in base.component_slots:
                fsn = EXIT_FS if sl.slot_type == "exit" else sl.feature_set_name
                slots.append(
                    {
                        "slot_type": sl.slot_type,
                        "ml_component_id": sl.ml_component_id,
                        "rule_component_id": sl.rule_component_id,
                        "feature_set_name": fsn,
                        "target_config": tc(sl),
                    }
                )
            t = await repo.create(
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
                engine_config=eng,
                validation_config=base.validation_config,
                seed=base.seed,
                description=desc,
                hypothesis="Feeding the distribution-top signal (consolidation/dist-day/hist-decline) to "
                "the exit HEAD lets it score real tops natively, beyond the cons gate.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({t.id})")
            ids[name] = t.id
        await s.commit()
    await async_engine.dispose()
    print("IDS=" + ",".join(f"{k}:{v}" for k, v in ids.items()))


asyncio.run(main())
