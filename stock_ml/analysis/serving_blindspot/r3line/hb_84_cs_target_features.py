# -*- coding: utf-8 -*-
"""hb_84: SYNTHESIS gốc — cross_sectional_exit target + cross-sectional FEATURES (exit_vol_rs RS
ranks). Robust target + robust features -> co the vua regime-robust vua MANH hon cs-alone (x13.9).
Danh gia tren STRIP bench (v_sigonly) + IC. So base velocity (x16.3) va cs-alone (x13.9)."""
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
STRIP={"exit_priority":["signal"],"max_hold_bars":10000,"signal_exit_hold_ext_atr":None,"signal_exit_hold_rs_scale":0.0,"signal_exit_hold_mkt_scale":0.0,"signal_exit_hold_legage_scale":0.0,"signal_exit_hold_legamp_scale":0.0,"signal_exit_hold_min_score3_z":None,"signal_exit_protect_lo":None,"signal_exit_protect_hi":None,"signal_exit_protect_release_drop_k":None,"signal_exit_skip_if_score3_z":None,"exit_snr_extend_threshold":None,"signal_exit_skip_if_mkt_above_ma":None}
VARIANTS={  # (exit_target, exit_feature_set)
 "cs_rs10":({"type":"cross_sectional_exit","horizon":10},"exit_vol_rs"),
 "cs_rs20":({"type":"cross_sectional_exit","horizon":20},"exit_vol_rs"),
 "velo_rs":({"type":"velocity_exit_regression","horizon":20,"upside_horizon":8,"vol_normalize":True,"vol_window":40},"exit_vol_rs"),
}
async def make(name,tgt,fs):
    S=sessionmaker(async_engine,class_=AsyncSession,expire_on_commit=False)
    async with S() as s:
        repo=StrategyTemplateRepository(s); ex=await repo.get_by_name(name)
        if ex: return ex.id
        base=await repo.get_by_id(3102); slots=[]
        for sl in base.component_slots:
            if sl.slot_type=="exit":
                slots.append({"slot_type":"exit","ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,"feature_set_name":fs,"target_config":copy.deepcopy(tgt)})
            else:
                slots.append({"slot_type":sl.slot_type,"ml_component_id":sl.ml_component_id,"rule_component_id":sl.rule_component_id,"feature_set_name":sl.feature_set_name,"target_config":copy.deepcopy(sl.target_config)})
        ec=copy.deepcopy(base.engine_config); ec.update(STRIP)
        t=await repo.create(name=name,market=base.market,strategy=base.strategy,feature_set_id=base.feature_set_id,target_id=base.target_id,component_slots=slots,direction=base.direction,signal_mode=base.signal_mode,signal_threshold=base.signal_threshold,entry_threshold=base.entry_threshold,exit_threshold=base.exit_threshold,split_config=copy.deepcopy(base.split_config),engine_config=ec,validation_config=base.validation_config,seed=42,description=f"root synth {tgt} {fs}",hypothesis="cs target+features strong+robust",universe_slug=base.universe_slug,model_mode="ml_only")
        await s.commit(); return t.id
def exit_ic(rid,con):
    sig=pd.read_sql("select symbol,date,exit_score from run_signals where run_id=%s and exit_score is not null",con,params=(rid,))
    sig['date']=pd.to_datetime(sig['date']); syms=sig.symbol.unique().tolist()
    d=duckdb.connect(r"market_data/market.duckdb",read_only=True); ph=",".join("?"*len(syms))
    px=d.execute(f"select symbol,date,close from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date",syms).fetchdf(); d.close()
    px['date']=pd.to_datetime(px['date']); fr=[]
    for s,g in px.groupby('symbol'):
        g=g.set_index('date').sort_index(); g['fwd10']=g['close'].shift(-10)/g['close']-1; fr.append(g[['fwd10']].assign(symbol=s).reset_index())
    m=sig.merge(pd.concat(fr),on=['symbol','date']).dropna(subset=['exit_score','fwd10']); m['year']=m.date.dt.year
    ov=ss.spearmanr(m.exit_score,m.fwd10).correlation
    by={int(y):round(ss.spearmanr(g.exit_score,g.fwd10).correlation,3) for y,g in m.groupby('year')}
    return ov,by
def main():
    con=psycopg2.connect(**PG)
    print("=== ROOT synth cs-target+cs-features on STRIP — NAV_sig@K18 + IC ===",flush=True)
    for name,(tgt,fs) in VARIANTS.items():
        tid=asyncio.run(make(name,tgt,fs)); asyncio.run(async_engine.dispose())
        r=run_template_experiment(template_id=tid,seed=42); rid=r.get("run_id")
        tr=pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days from run_trades where run_id=%s and exit_date is not null",con,params=(rid,))
        cv=HERE/f"_k84_{name}.csv"; tr.to_csv(cv,index=False)
        nav=shuffle_stats(NavSim2(str(cv),date_lo="2020-01-01"),K=18,roundtrip=0.006,settle_lag=2,advance_fee=0.0008,n=20)["mean"]
        ov,by=exit_ic(rid,con)
        print(f"  {name:8s} ntr={len(tr):4d} hold={tr.holding_days.mean():4.1f} NAV=x{nav:5.1f} | IC={ov:+.3f} 2022={by.get(2022)} 2024={by.get(2024)} 2023={by.get(2023)}",flush=True)
    con.close(); print("HB_84_DONE",flush=True)
if __name__=="__main__": main()
