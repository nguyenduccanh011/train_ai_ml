"""Entry-head retrain — clone champion 1800, vary ONLY the entry slot feature_set to add the
features that DISTINGUISH never-worked knife-catches from winners (diagnosis entry_separation_hunt:
range_pos_20 sep +0.90, higher_lows, reversal-confirm, recovery). Goal: an entry head whose score
culls the -22u never-worked drain (which the current entry_lvup126_lean can't — precision 29%).
Eval: does a higher entry_threshold now win (cull never-worked without cutting winners)?
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

BASE_ID = 1800
# (tag, entry_feature_set)
GRID = [
    ("recov",  "entry_lvup126_recov"),   # + range_pos_20 (#1 separator), dist_10d_low, recov_setup
    ("engulf", "entry_lvup126_engulf"),  # + bull_engulf (bottom-reversal confirm ~ higher_low)
    ("xsec",   "entry_lvup126_xsec"),    # + cross-sectional ranks (raises forward-IC per catalog note)
]


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); base = await repo.get_by_id(BASE_ID)
        be = base.engine_config; be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
        tc = lambda sl: (json.loads(sl.target_config) if isinstance(sl.target_config, str) else copy.deepcopy(sl.target_config))
        created = []
        for tag, fs in GRID:
            name = f"n2_er_{tag}"
            ex = await repo.get_by_name(name)
            if ex: print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            slots = []
            for sl in base.component_slots:
                fsn = fs if sl.slot_type == "entry" else sl.feature_set_name
                slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                              "rule_component_id": sl.rule_component_id, "feature_set_name": fsn,
                              "target_config": tc(sl)})
            t = await repo.create(name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
                direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(be),
                validation_config=base.validation_config, seed=base.seed,
                description=f"Entry retrain {tag}: entry={fs} on champion 1800 (add never-worked separators).",
                hypothesis="Adding range_pos/reversal/recovery features lets the entry score separate "
                           "never-worked knife-catches from winners → a higher entry_threshold can cull the -22u drain.",
                universe_slug=base.universe_slug)
            print(f"* {name} ({t.id})"); created.append((t.id, name))
        await s.commit(); print("IDS=" + ",".join(str(i) for i, _ in created))
    await async_engine.dispose()

asyncio.run(main())
