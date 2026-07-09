"""Tune the ENSEMBLE reversal TARGET (the axis where ensemble value lives) — clone champion 1804,
vary reversal_entry_regression params. penalty=punish deep-drawdown 'gains' (sharper V-bottom),
min_fwd_rally=require a real rebound (cleaner), dip_window=dip context, horizon=rebound window
(<=20 keeps gap<=65). Eval which sharpens the 2nd head's orthogonal V-bottom adds."""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository
BASE_ID = 1804
def T(**kw):  # reversal target with overrides on the champion default
    d = {"type":"reversal_entry_regression","horizon":10,"penalty":1.0,"dip_window":50,"min_fwd_rally":0.10}
    d.update(kw); return d
GRID = [
    ("pen15", T(penalty=1.5)), ("pen20", T(penalty=2.0)),
    ("mfr15", T(min_fwd_rally=0.15)), ("mfr05", T(min_fwd_rally=0.05)),
    ("dip30", T(dip_window=30)), ("h15", T(horizon=15)),
]
async def main():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); base = await repo.get_by_id(BASE_ID)
        be = base.engine_config; be = json.loads(be) if isinstance(be,str) else be
        tc=lambda sl:(json.loads(sl.target_config) if isinstance(sl.target_config,str) else copy.deepcopy(sl.target_config))
        created=[]
        for tag, tgt in GRID:
            name=f"n2_rt_{tag}"; ex=await repo.get_by_name(name)
            if ex: print(f"= {name} ({ex.id})"); created.append((ex.id,name)); continue
            slots=[{"slot_type":sl.slot_type,"ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,"feature_set_name":sl.feature_set_name,"target_config":tc(sl)} for sl in base.component_slots]
            eng=copy.deepcopy(be); eng["entry_ensemble"]={"target":tgt,"z_threshold":0.9}
            t=await repo.create(name=name,market=base.market,strategy=base.strategy,feature_set_id=base.feature_set_id,target_id=base.target_id,component_slots=copy.deepcopy(slots),direction=base.direction,signal_mode=base.signal_mode,signal_threshold=base.signal_threshold,entry_threshold=base.entry_threshold,exit_threshold=base.exit_threshold,split_config=base.split_config,engine_config=eng,validation_config=base.validation_config,seed=base.seed,description=f"reversal-target tune {tag}: {tgt}",hypothesis="sharper reversal target -> better orthogonal V-bottom adds",universe_slug=base.universe_slug)
            print(f"* {name} ({t.id})"); created.append((t.id,name))
        await s.commit(); print("IDS="+",".join(str(i) for i,_ in created))
    await async_engine.dispose()
asyncio.run(main())
