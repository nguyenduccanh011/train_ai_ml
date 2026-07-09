"""PULLBACK-OFF crutch-replacement sandbox (user 2026-06-18, Option A of the decouple_oracle follow-up).

Clone champion 1930 TWICE with the 4.5% pullback crutch DISABLED (entry_pullback_pct=None), so entry
fills at-market (close_next) and the price-buffer that masks entry-ML is removed:
  n2_pboff_recov  = sandbox BASELINE  (entry slot = entry_lvup126_recov, the champion's set)
  n2_pboff_knife  = sandbox CHALLENGER (entry slot = entry_recov_knife = recov + dist_63d_high + sma_200_ratio)

Judge by SANDBOX DELTA (knife vs recov, both pullback-off) per feedback_crutch_replacement_sandbox — NOT
the crutch-loaded composite, where every entry-feature add already tested-negative (masked). Tests whether
knife/structural-decliner awareness lets the entry head self-manage risk and start closing the crutch gap.
"""
from __future__ import annotations
import asyncio
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

BASE_ID = 1930
VARIANTS = [
    ("n2_pboff_recov", "entry_lvup126_recov",
     "Champion 1930, pullback crutch OFF (sandbox baseline)."),
    ("n2_pboff_knife", "entry_recov_knife",
     "Champion 1930, pullback crutch OFF + knife-axis entry set (dist_63d_high + sma_200_ratio)."),
]


async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    ids = {}
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_ID)
        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else copy.deepcopy(be)
        tc = lambda sl: (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                         else copy.deepcopy(sl.target_config))
        for name, entry_fs, desc in VARIANTS:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id}) exists")
                ids[name] = ex.id
                continue
            eng = copy.deepcopy(be)
            eng["entry_pullback_pct"] = None          # disable the 4.5% pullback crutch
            slots = []
            for sl in base.component_slots:
                fsn = entry_fs if sl.slot_type == "entry" else sl.feature_set_name
                slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                              "rule_component_id": sl.rule_component_id, "feature_set_name": fsn,
                              "target_config": tc(sl)})
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
                direction=base.direction, signal_mode=base.signal_mode,
                signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold,
                exit_threshold=base.exit_threshold, split_config=base.split_config,
                engine_config=eng, validation_config=base.validation_config, seed=base.seed,
                description=desc,
                hypothesis="Pullback-OFF sandbox: does knife/structural awareness in the entry head "
                           "replace the price-buffer crutch (judge by sandbox delta vs recov)?",
                universe_slug=base.universe_slug)
            print(f"* {name} ({t.id})")
            ids[name] = t.id
        await s.commit()
    await async_engine.dispose()
    print("IDS=" + ",".join(f"{k}:{v}" for k, v in ids.items()))


asyncio.run(main())
