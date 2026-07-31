# -*- coding: utf-8 -*-
"""Vì sao 'lệnh hết hạn (không khớp) lãi cao' NHƯNG mua chúng lại thua? So return TỪ SIGNAL (hindsight)
vs TỪ THỜI ĐIỂM MUA ĐƯỢC (window-end, khi mới biết nó hết hạn)."""
import sys; from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]; sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd
RID="template/x2_struct_to_k16preempt_cssize-69338138"
con=psycopg2.connect(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev")
o=pd.read_sql("SELECT DISTINCT symbol,signal_date,outcome,result_date FROM run_pending WHERE run_id=%s AND outcome='expire'",con,params=(RID,)); con.close()
o["signal_date"]=pd.to_datetime(o["signal_date"]); o["result_date"]=pd.to_datetime(o["result_date"])
cx=duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb",read_only=True)
px=cx.execute("SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"]=pd.to_datetime(px["date"]); CA={}; BO={}
for s,g in px.groupby("symbol"):
    g=g.sort_values("date").reset_index(drop=True); CA[s]=g["close"].to_numpy(); BO[s]={d:i for i,d in enumerate(g["date"])}
def px_b(sym,date):
    b=BO.get(sym,{}).get(pd.Timestamp(date)); ca=CA.get(sym); return (b, ca[b] if b is not None and ca is not None else None)
r_sig, r_end = [], []
for x in o.itertuples():
    bS,cS=px_b(x.symbol,x.signal_date); bE,cE=px_b(x.symbol,x.result_date)   # result_date = window-end for expire
    if bS is None or cS is None or bE is None or cE is None: continue
    ca=CA[x.symbol]; H=20
    if bS+H<len(ca): r_sig.append(ca[bS+H]/cS-1)      # from SIGNAL (hindsight — không mua được ở đây)
    if bE+H<len(ca): r_end.append(ca[bE+H]/cE-1)      # from WINDOW-END (mới biết hết hạn -> mua được ở ĐÂY)
print(f"lệnh hết hạn: {len(o)}")
print(f"return +20bar TỪ SIGNAL (hindsight, KHÔNG mua được): {np.mean(r_sig)*100:+.2f}%  win {100*np.mean(np.array(r_sig)>0):.0f}%")
print(f"return +20bar TỪ WINDOW-END (mới biết, MUA được ở đây):{np.mean(r_end)*100:+.2f}%  win {100*np.mean(np.array(r_end)>0):.0f}%")
print(f"giá đã CHẠY từ signal tới window-end trung bình: {np.mean([ (px_b(x.symbol,x.result_date)[1]/px_b(x.symbol,x.signal_date)[1]-1) for x in o.itertuples() if px_b(x.symbol,x.signal_date)[1] and px_b(x.symbol,x.result_date)[1]])*100:+.1f}%")
