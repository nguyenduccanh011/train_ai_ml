# -*- coding: utf-8 -*-
"""hb_78: TO HOP LON — mean-rev nhu 1 HEAD trong entry ensemble (entry_ensemble5). Clone ft_rs,
them head reversal ngan-han (dip_window=15, min_fwd_rally=0.05 = oversold-bounce) vao ensemble.
Recombine union-rank -> dead-year momentum yeu, mean-rev candidates rank cao -> fill (priority-
aware, khac sleeve equal-priority da fail). Train + NAV + by-year 2024."""
from __future__ import annotations
import asyncio, copy, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"stock_ml"))
import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
PG=dict(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev")
BASE=3102; NEWNAME="ft_rs_mrhead"
ENS5={"target":{"type":"reversal_entry_regression","horizon":10,"penalty":1.0,
                "dip_window":15,"min_fwd_rally":0.05},"z_threshold":0.7}
async def make():
    S=sessionmaker(async_engine,class_=AsyncSession,expire_on_commit=False)
    async with S() as s:
        repo=StrategyTemplateRepository(s); ex=await repo.get_by_name(NEWNAME)
        if ex: return ex.id
        base=await repo.get_by_id(BASE)
        slots=[{"slot_type":sl.slot_type,"ml_component_id":sl.ml_component_id,
                "rule_component_id":sl.rule_component_id,"feature_set_name":sl.feature_set_name,
                "target_config":copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        ec=copy.deepcopy(base.engine_config); ec["entry_ensemble5"]=ENS5
        t=await repo.create(name=NEWNAME,market=base.market,strategy=base.strategy,
            feature_set_id=base.feature_set_id,target_id=base.target_id,component_slots=slots,
            direction=base.direction,signal_mode=base.signal_mode,signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold,exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config),engine_config=ec,
            validation_config=base.validation_config,seed=42,
            description="ft_rs + entry_ensemble5 mean-rev head (short-dip reversal) — union priority-aware 1 book",
            hypothesis="mean-rev head fills dead-year khi momentum yeu, khong displacement vi recombine rank",
            universe_slug=base.universe_slug,model_mode="ml_only")
        await s.commit(); return t.id
def main():
    tid=asyncio.run(make()); asyncio.run(async_engine.dispose())
    print(f"[hb_78] t{tid} = {NEWNAME}",flush=True)
    r=run_template_experiment(template_id=tid,seed=42); rid=r.get("run_id")
    con=psycopg2.connect(**PG);cur=con.cursor()
    cur.execute("select composite_score,total_pnl,pf,mdd_per_symbol,trades from leaderboard_runs where run_id=%s",(rid,))
    row=cur.fetchone();con.close()
    if row: print(f"[hb_78] {rid}: comp={row[0]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",flush=True)
    print(f"[hb_78] RUN_ID={rid}",flush=True); print("HB_78_DONE",flush=True)
if __name__=="__main__": main()
