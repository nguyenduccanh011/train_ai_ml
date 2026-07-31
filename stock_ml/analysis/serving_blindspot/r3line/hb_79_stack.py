# -*- coding: utf-8 -*-
"""hb_79: stack marginal wins — exit-RS (t3112) + overext_trail 0.025. Cache-reuse t3112."""
from __future__ import annotations
import asyncio, copy, shutil, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"stock_ml"))
import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
PG=dict(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev")
BASE,FP=3112,"565c005aa3"; NEWNAME="ft_rs_exitrs_oxt025"
async def make():
    S=sessionmaker(async_engine,class_=AsyncSession,expire_on_commit=False)
    async with S() as s:
        repo=StrategyTemplateRepository(s); ex=await repo.get_by_name(NEWNAME)
        if ex: return ex.id
        base=await repo.get_by_id(BASE)
        slots=[{"slot_type":sl.slot_type,"ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,"feature_set_name":sl.feature_set_name,"target_config":copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        ec=copy.deepcopy(base.engine_config); ec["overext_trail_pct"]=0.025
        t=await repo.create(name=NEWNAME,market=base.market,strategy=base.strategy,feature_set_id=base.feature_set_id,target_id=base.target_id,component_slots=slots,direction=base.direction,signal_mode=base.signal_mode,signal_threshold=base.signal_threshold,entry_threshold=base.entry_threshold,exit_threshold=base.exit_threshold,split_config=copy.deepcopy(base.split_config),engine_config=ec,validation_config=base.validation_config,seed=42,description="stack exit-RS+oxt025",hypothesis="stack",universe_slug=base.universe_slug,model_mode="ml_only")
        await s.commit(); return t.id
def main():
    tid=asyncio.run(make()); asyncio.run(async_engine.dispose())
    src=REPO/f"results/tmpl_{BASE}_{FP}/folds"; dst=REPO/f"results/tmpl_{tid}_{FP}/folds"; dst.mkdir(parents=True,exist_ok=True)
    for p in src.glob("*.parquet"):
        if not (dst/p.name).exists(): shutil.copy2(p,dst/p.name)
    r=run_template_experiment(template_id=tid,seed=42); rid=r.get("run_id")
    con=psycopg2.connect(**PG);cur=con.cursor(); cur.execute("select composite_score,trades from leaderboard_runs where run_id=%s",(rid,)); print("stack:",rid,cur.fetchone()); con.close()
    print("HB_79_DONE",flush=True)
if __name__=="__main__": main()
