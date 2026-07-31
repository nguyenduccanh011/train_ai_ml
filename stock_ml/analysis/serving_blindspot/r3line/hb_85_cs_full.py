# -*- coding: utf-8 -*-
"""hb_85: cs_rs20 (regime-robust exit) trong model FULL (rule on) @K18 — raw robust co giup
model deploy + bot phu thuoc mask khong? So velocity-full ft_rs (x31.8). Va cs_rs20 + BOT mask
(bo regime-skip, max_hold 30) — tháo mask duoc bao nhieu."""
from __future__ import annotations
import asyncio, copy, sys, os
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"stock_ml"))
sys.path.insert(0,os.environ.get("NH_NAV2_DIR","F:/PROJECTS/hb2943_work"))
import psycopg2,pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats
PG=dict(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev"); HERE=Path(__file__).parent
CS20={"type":"cross_sectional_exit","horizon":20}
# variants: (exit_target, exit_fs, engine_overrides)
V={
 "csf_full":(CS20,"exit_vol_rs",{}),                                   # cs_rs20 + full rules
 "csf_noskip":(CS20,"exit_vol_rs",{"signal_exit_skip_if_mkt_above_ma":None}),  # bo regime-skip mask
 "csf_lessmask":(CS20,"exit_vol_rs",{"signal_exit_skip_if_mkt_above_ma":None,"max_hold_bars":30}), # bot mask
}
async def make(name,tgt,fs,ov):
    S=sessionmaker(async_engine,class_=AsyncSession,expire_on_commit=False)
    async with S() as s:
        repo=StrategyTemplateRepository(s); ex=await repo.get_by_name(name)
        if ex: return ex.id
        base=await repo.get_by_id(3102); slots=[]
        for sl in base.component_slots:
            if sl.slot_type=="exit": slots.append({"slot_type":"exit","ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,"feature_set_name":fs,"target_config":copy.deepcopy(tgt)})
            else: slots.append({"slot_type":sl.slot_type,"ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,"feature_set_name":sl.feature_set_name,"target_config":copy.deepcopy(sl.target_config)})
        ec=copy.deepcopy(base.engine_config); ec.update(ov)
        t=await repo.create(name=name,market=base.market,strategy=base.strategy,feature_set_id=base.feature_set_id,target_id=base.target_id,component_slots=slots,direction=base.direction,signal_mode=base.signal_mode,signal_threshold=base.signal_threshold,entry_threshold=base.entry_threshold,exit_threshold=base.exit_threshold,split_config=copy.deepcopy(base.split_config),engine_config=ec,validation_config=base.validation_config,seed=42,description="cs_rs20 full/less-mask",hypothesis="robust raw -> less mask",universe_slug=base.universe_slug,model_mode="ml_only")
        await s.commit(); return t.id
def yr_nav(csv):
    import statistics
    out={}
    for lo,tag in (("2020-01-01","full"),("2022-01-01","f22"),("2024-01-01","f24")):
        out[tag]=shuffle_stats(NavSim2(csv,date_lo=lo),K=18,roundtrip=0.006,settle_lag=2,advance_fee=0.0008,n=20)["mean"]
    return out
def main():
    con=psycopg2.connect(**PG)
    print("=== cs_rs20 FULL/less-mask @K18 (so velocity-full ft_rs x31.8) ===",flush=True)
    for name,(tgt,fs,ov) in V.items():
        tid=asyncio.run(make(name,tgt,fs,ov)); asyncio.run(async_engine.dispose())
        r=run_template_experiment(template_id=tid,seed=42); rid=r.get("run_id")
        tr=pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days from run_trades where run_id=%s and exit_date is not null",con,params=(rid,))
        cv=HERE/f"_k85_{name}.csv"; tr.to_csv(cv,index=False); o=yr_nav(str(cv))
        print(f"  {name:12s} ntr={len(tr):4d} hold={tr.holding_days.mean():4.1f} NAV=x{o['full']:.1f} f22=x{o['f22']:.2f} f24=x{o['f24']:.2f}",flush=True)
    con.close(); print("HB_85_DONE",flush=True)
if __name__=="__main__": main()
