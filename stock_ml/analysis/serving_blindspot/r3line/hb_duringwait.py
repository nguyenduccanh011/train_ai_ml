# -*- coding: utf-8 -*-
"""Động lực TRONG quãng chờ (signal -> fill/expire): thời gian chờ, return during-wait, và 'nếu giữ từ
signal xuyên suốt' thì P&L bao nhiêu — để thấy champion đánh đổi gì khi đứng NGOÀI trong lúc chờ."""
import sys; from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd
RID="template/x2_struct_to_k16preempt_cssize-69338138"
con=psycopg2.connect(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev")
o=pd.read_sql("SELECT DISTINCT symbol,signal_date,outcome,limit_price,result_date FROM run_pending WHERE run_id=%s",con,params=(RID,)); con.close()
o["signal_date"]=pd.to_datetime(o["signal_date"]); o["result_date"]=pd.to_datetime(o["result_date"])
cx=duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb",read_only=True)
px=cx.execute("SELECT symbol,date,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"]=pd.to_datetime(px["date"]); CA={};HI={};LO={};BO={}
for s,g in px.groupby("symbol"):
    g=g.sort_values("date").reset_index(drop=True); CA[s]=g["close"].to_numpy();HI[s]=g["high"].to_numpy();LO[s]=g["low"].to_numpy();BO[s]={d:i for i,d in enumerate(g["date"])}
rows=[]
for x in o.itertuples():
    bS=BO.get(x.symbol,{}).get(x.signal_date); bR=BO.get(x.symbol,{}).get(x.result_date)
    if bS is None or bR is None or bR<bS: continue
    cS=CA[x.symbol][bS]
    seg_hi=HI[x.symbol][bS:bR+1]; seg_lo=LO[x.symbol][bS:bR+1]
    resolve_px = x.limit_price if x.outcome=="fill" else CA[x.symbol][bR]   # fill@limit ; expire@window-end close
    rows.append(dict(outcome=x.outcome, bars=bR-bS,
                     dw_ret=resolve_px/cS-1,                 # return signal->resolution (if held from signal)
                     maxup=seg_hi.max()/cS-1,                # best drawup during wait
                     maxdn=seg_lo.min()/cS-1))               # worst drawdown during wait
d=pd.DataFrame(rows)
print(f"tổng lệnh chờ: {len(d)}  (fill={ (d.outcome=='fill').sum() } expire={ (d.outcome=='expire').sum() })\n")
print("=== TRONG quãng chờ (signal -> khớp/hết hạn) ===")
print("nhóm    |  n   | bars chờ(median) | return giữ-từ-signal | max UP | max DOWN")
for oc in ["fill","expire"]:
    g=d[d.outcome==oc]
    print(f"{oc:7s} | {len(g):4d} | {g.bars.median():6.0f}          | {100*g.dw_ret.mean():+7.2f}%          | {100*g.maxup.mean():+6.1f}% | {100*g.maxdn.mean():+6.1f}%")
w=(d.outcome=='fill').mean()
print(f"\n'Giữ từ signal xuyên suốt quãng chờ' (market-entry): return trung bình = {100*d.dw_ret.mean():+.2f}%")
print(f"  = {w*100:.0f}%×(fill {100*d[d.outcome=='fill'].dw_ret.mean():+.1f}%) + {(1-w)*100:.0f}%×(expire {100*d[d.outcome=='expire'].dw_ret.mean():+.1f}%)")
print(f"\nchampion ĐỨNG NGOÀI trong quãng chờ -> né max-down fill ({100*d[d.outcome=='fill'].maxdn.mean():.1f}%), BỎ up expire ({100*d[d.outcome=='expire'].maxup.mean():+.1f}%)")
