# -*- coding: utf-8 -*-
import sys; from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd
PG=dict(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev")
RID="template/x2_struct_to_k16preempt_cssize-69338138"
con=psycopg2.connect(**PG)
tr=pd.read_sql("SELECT symbol,entry_signal_date,exit_reason,pnl_pct FROM run_trades WHERE run_id=%s AND exit_reason<>'preempt' AND entry_signal_date IS NOT NULL",con,params=(RID,)); con.close()
tr["d"]=pd.to_datetime(tr["entry_signal_date"])
tr["loser"]=((tr.exit_reason=="signal")&(tr.pnl_pct<0)).astype(int)
tr["winner"]=((tr.exit_reason.isin(["max_hold","overext_trail"]))&(tr.pnl_pct>0)).astype(int)
cx=duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb",read_only=True)
syms=",".join(repr(s) for s in tr.symbol.unique())
px=cx.execute(f"SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({syms}) AND date>='2018-01-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"]=pd.to_datetime(px["date"])
# WEEKLY trend features from daily close (resample W-FRI)
parts=[]
for s,g in px.groupby("symbol"):
    g=g.sort_values("date").set_index("date")
    wk=g["close"].resample("W-FRI").last()
    wtrend=(wk/wk.rolling(10).mean()-1)          # weekly close vs 10-week MA
    wslope=(wk/wk.shift(4)-1)                      # 4-week momentum
    wup=((wk>wk.rolling(10).mean())&(wk.rolling(10).mean().diff()>0)).astype(float)  # weekly uptrend flag
    df=pd.DataFrame({"wk_trend":wtrend,"wk_mom":wslope,"wk_up":wup}).reindex(g.index,method="ffill")
    df["symbol"]=s; df["date"]=g.index; parts.append(df.reset_index(drop=True))
W=pd.concat(parts,ignore_index=True)
sub=tr[(tr.loser==1)|(tr.winner==1)].merge(W,left_on=["symbol","d"],right_on=["symbol","date"],how="left")
print(f"losers={int(tr.loser.sum())} winners={int(tr.winner.sum())}")
print("weekly feature | winner | loser | |AUC-.5|")
for col in ["wk_trend","wk_mom","wk_up"]:
    w=sub[sub.winner==1][col].dropna(); lo=sub[sub.loser==1][col].dropna()
    if len(w)<30 or len(lo)<30: continue
    allv=pd.concat([w,lo]); rk=allv.rank(); n1=len(w)
    auc=(rk[:n1].sum()-n1*(n1+1)/2)/(n1*len(lo))
    print(f"{col:10s} | {w.mean():+.3f} | {lo.mean():+.3f} | {abs(auc-0.5):.3f}")
# winner-rate in weekly-up vs weekly-down entries
sub["wentry"]=np.where(sub.wk_up>=0.5,"wk_UP","wk_DOWN")
print("\nwinner-rate by weekly regime at entry:")
for r,g in sub.groupby("wentry"):
    print(f"  {r}: n={len(g)} winner%={100*(g.winner==1).mean():.0f}% (loser%={100*(g.loser==1).mean():.0f}%)")
print("WEEKLY_CHECK_DONE")
