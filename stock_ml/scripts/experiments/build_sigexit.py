from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from stock_ml.db.engine import async_engine
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository
from stock_ml.scripts.run_template import run_template_experiment
PG=dict(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev")
BASE=3185
VARIANTS=[
 ("se_off", {"signal_exit_enabled": False}),
 ("se_ma150", {"signal_exit_skip_if_mkt_above_ma": 150}),
 ("se_ma100", {"signal_exit_skip_if_mkt_above_ma": 100}),
 ("se_ma50", {"signal_exit_skip_if_mkt_above_ma": 50}),
 ("se_gb15", {"exit_snr_defer_min_giveback": 0.15}),
]
async def make_all():
    ids={}
    Session=sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo=StrategyTemplateRepository(s); base=await repo.get_by_id(BASE)
        bs=[]
        for sl in base.component_slots:
            tc=sl.target_config; tc=json.loads(tc) if isinstance(tc,str) else copy.deepcopy(tc)
            bs.append({"slot_type":sl.slot_type,"ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,"feature_set_name":sl.feature_set_name,"target_config":tc})
        be=base.engine_config; be=json.loads(be) if isinstance(be,str) else dict(be)
        for nm,ov in VARIANTS:
            ex=await repo.get_by_name(nm)
            if ex: print(f"exists {nm} {ex.id}"); ids[nm]=ex.id; continue
            eng=copy.deepcopy(be); eng.update(ov)
            t=await repo.create(name=nm,market=base.market,strategy=base.strategy,feature_set_id=base.feature_set_id,target_id=base.target_id,component_slots=copy.deepcopy(bs),direction=base.direction,signal_mode=base.signal_mode,signal_threshold=base.signal_threshold,entry_threshold=base.entry_threshold,exit_threshold=base.exit_threshold,split_config=base.split_config,engine_config=eng,validation_config=base.validation_config,seed=base.seed,description=f"frontier + sigexit fix {ov}",hypothesis="signal-exit is net -28u leak; suppress/disable helps",universe_slug=base.universe_slug,model_mode=base.model_mode)
            await s.commit(); print(f"created {nm} {t.id} ov={ov}"); ids[nm]=t.id
    return ids
def read(rid):
    con=psycopg2.connect(**PG);cur=con.cursor();cur.execute("SELECT composite_score,total_pnl,trades FROM leaderboard_runs WHERE run_id=%s",(rid,));r=cur.fetchone();con.close();return r
ids=asyncio.run(make_all()); asyncio.run(async_engine.dispose())
for nm,tid in ids.items():
    try:
        r=run_template_experiment(template_id=tid,seed=42); rid=r.get("run_id"); row=read(rid)
        print(f"  {nm}(t{tid}): comp={row[0]} pnl={row[1]:.1f} tr={row[2]} run_id={rid}",flush=True)
    except Exception as e: print(f"  {nm}: ERROR {type(e).__name__}: {str(e)[:150]}",flush=True)
print("SIGEXIT_DONE")
