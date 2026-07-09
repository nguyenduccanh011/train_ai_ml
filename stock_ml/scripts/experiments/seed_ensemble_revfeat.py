"""Clone the ensemble champion 1804 but give the REVERSAL head its OWN feature set
(entry_reversal_confirm: range_pos/recov + rsi_div/macd_div + zigzag-structure zz_* the
momentum set lacks) instead of sharing the momentum features. Tests if reversal-specific
features make the 2nd head's V-bottom score sharper."""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository
BASE_ID, NEW_NAME = 1804, "n2_ens_revfeat"
async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); base = await repo.get_by_id(BASE_ID)
        ex = await repo.get_by_name(NEW_NAME)
        if ex: print(f"= {NEW_NAME} ({ex.id})"); await async_engine.dispose(); return
        tc=lambda sl:(json.loads(sl.target_config) if isinstance(sl.target_config,str) else copy.deepcopy(sl.target_config))
        slots=[{"slot_type":sl.slot_type,"ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,
                "feature_set_name":sl.feature_set_name,"target_config":tc(sl)} for sl in base.component_slots]
        be=base.engine_config; be=json.loads(be) if isinstance(be,str) else be; eng=copy.deepcopy(be)
        eng["entry_ensemble"]["features"]="entry_reversal_confirm"   # own feature set for the reversal head
        eng["entry_ensemble"]["z_threshold"]=0.9
        t=await repo.create(name=NEW_NAME,market=base.market,strategy=base.strategy,feature_set_id=base.feature_set_id,
            target_id=base.target_id,component_slots=copy.deepcopy(slots),direction=base.direction,signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,entry_threshold=base.entry_threshold,exit_threshold=base.exit_threshold,
            split_config=base.split_config,engine_config=eng,validation_config=base.validation_config,seed=base.seed,
            description=f"{NEW_NAME}: ensemble champ 1804 + reversal head on its OWN feature set entry_reversal_confirm.",
            hypothesis="Reversal-confirm + zigzag-structure features (range_pos/divergence/zz_*) make the 2nd head's "
                       "V-bottom score sharper than sharing the momentum features → more/better orthogonal trades.",
            universe_slug=base.universe_slug)
        print(f"* {NEW_NAME} ({t.id})\nID={t.id}"); await s.commit()
    await async_engine.dispose()
asyncio.run(main())
