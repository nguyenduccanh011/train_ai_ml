# -*- coding: utf-8 -*-
"""hb_86: day DO MANH raw exit — cs_rs20 target + exit_vol_rs + exit MODEL LON (id66, 31 la,
no monotone-velocity) vs model nho (id78, 7 la). Do raw IC + NAV_sig (stripped) va full @K18.
Model lon manh hon -> raw signal manh+robust -> dan thao mask?"""
from __future__ import annotations
import asyncio, copy, sys, os
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"stock_ml"))
sys.path.insert(0,os.environ.get("NH_NAV2_DIR","F:/PROJECTS/hb2943_work"))
import psycopg2,duckdb,pandas as pd,scipy.stats as ss
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats
PG=dict(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev"); HERE=Path(__file__).parent
CS20={"type":"cross_sectional_exit","horizon":20}
STRIP={"exit_priority":["signal"],"max_hold_bars":10000,"signal_exit_hold_ext_atr":None,"signal_exit_hold_rs_scale":0.0,"signal_exit_hold_mkt_scale":0.0,"signal_exit_hold_legage_scale":0.0,"signal_exit_hold_legamp_scale":0.0,"signal_exit_hold_min_score3_z":None,"signal_exit_protect_lo":None,"signal_exit_protect_hi":None,"signal_exit_protect_release_drop_k":None,"signal_exit_skip_if_score3_z":None,"exit_snr_extend_threshold":None,"signal_exit_skip_if_mkt_above_ma":None}
V={"big_strip":(66,STRIP),"big_full":(66,{})}   # exit model 66, stripped vs full
async def make(name,mid,ov):
    S=sessionmaker(async_engine,class_=AsyncSession,expire_on_commit=False)
    async with S() as s:
        repo=StrategyTemplateRepository(s); ex=await repo.get_by_name(name)
        if ex: return ex.id
        base=await repo.get_by_id(3102); slots=[]
        for sl in base.component_slots:
            if sl.slot_type=="exit": slots.append({"slot_type":"exit","ml_component_id":mid,"rule_component_id":sl.rule_component_id,"feature_set_name":"exit_vol_rs","target_config":copy.deepcopy(CS20)})
            else: slots.append({"slot_type":sl.slot_type,"ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,"feature_set_name":sl.feature_set_name,"target_config":copy.deepcopy(sl.target_config)})
        ec=copy.deepcopy(base.engine_config); ec.update(ov)
        t=await repo.create(name=name,market=base.market,strategy=base.strategy,feature_set_id=base.feature_set_id,target_id=base.target_id,component_slots=slots,direction=base.direction,signal_mode=base.signal_mode,signal_threshold=base.signal_threshold,entry_threshold=base.entry_threshold,exit_threshold=base.exit_threshold,split_config=copy.deepcopy(base.split_config),engine_config=ec,validation_config=base.validation_config,seed=42,description="cs_rs20 big exit model",hypothesis="bigger model stronger raw",universe_slug=base.universe_slug,model_mode="ml_only")
        await s.commit(); return t.id
def eic(rid,con):
    sig=pd.read_sql("select symbol,date,exit_score from run_signals where run_id=%s and exit_score is not null",con,params=(rid,))
    sig['date']=pd.to_datetime(sig['date']); syms=sig.symbol.unique().tolist()
    d=duckdb.connect(r"market_data/market.duckdb",read_only=True); ph=",".join("?"*len(syms))
    px=d.execute(f"select symbol,date,close from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date",syms).fetchdf(); d.close()
    px['date']=pd.to_datetime(px['date']); fr=[]
    for s,g in px.groupby('symbol'):
        g=g.set_index('date').sort_index(); g['fwd10']=g['close'].shift(-10)/g['close']-1; fr.append(g[['fwd10']].assign(symbol=s).reset_index())
    m=sig.merge(pd.concat(fr),on=['symbol','date']).dropna(subset=['exit_score','fwd10']); m['year']=m.date.dt.year
    return ss.spearmanr(m.exit_score,m.fwd10).correlation,{int(y):round(ss.spearmanr(g.exit_score,g.fwd10).correlation,3) for y,g in m.groupby('year')}
def main():
    con=psycopg2.connect(**PG)
    print("=== cs_rs20 + BIG exit model(66) — strip(IC/rawNAV) + full@K18 ===",flush=True)
    for name,(mid,ov) in V.items():
        tid=asyncio.run(make(name,mid,ov)); asyncio.run(async_engine.dispose())
        r=run_template_experiment(template_id=tid,seed=42); rid=r.get("run_id")
        tr=pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days from run_trades where run_id=%s and exit_date is not null",con,params=(rid,))
        cv=HERE/f"_k86_{name}.csv"; tr.to_csv(cv,index=False)
        nav=shuffle_stats(NavSim2(str(cv),date_lo="2020-01-01"),K=18,roundtrip=0.006,settle_lag=2,advance_fee=0.0008,n=20)["mean"]
        f22=shuffle_stats(NavSim2(str(cv),date_lo="2022-01-01"),K=18,roundtrip=0.006,settle_lag=2,advance_fee=0.0008,n=20)["mean"]
        ov,by=eic(rid,con)
        print(f"  {name:10s} ntr={len(tr):4d} hold={tr.holding_days.mean():4.1f} NAV=x{nav:.1f} f22=x{f22:.2f} | IC={ov:+.3f} 2022={by.get(2022)} 2024={by.get(2024)}",flush=True)
    con.close(); print("HB_86_DONE",flush=True)
if __name__=="__main__": main()
