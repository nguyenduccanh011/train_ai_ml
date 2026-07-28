import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd, duckdb, psycopg2
PG=dict(host="localhost",port=5433,dbname="stockml",user="stockml",password="stockml_dev")
MARKET="F:/PROJECTS/train_ai_ml/market_data/market.duckdb"; FRONT="template/x2_struct_to-69338138"
con=psycopg2.connect(**PG)
tr=pd.read_sql("SELECT symbol,entry_signal_date,pnl_pct FROM run_trades WHERE run_id=%s AND pnl_pct IS NOT NULL AND entry_signal_date IS NOT NULL",con,params=(FRONT,))
con.close()
tr["d"]=pd.to_datetime(tr["entry_signal_date"]); tr["yr"]=tr["d"].dt.year
cx=duckdb.connect(MARKET,read_only=True)
px=cx.execute("SELECT symbol,date,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf();cx.close()
px["date"]=pd.to_datetime(px["date"]); parts=[]
for s,g in px.groupby("symbol"):
    g=g.sort_values("date").copy()
    g["dist20low"]=g["close"]/g["low"].rolling(20).min()-1
    g["dist_ma20"]=g["close"]/g["close"].rolling(20).mean()-1
    parts.append(g[["symbol","date","dist20low","dist_ma20"]])
F=pd.concat(parts,ignore_index=True)
m=tr.merge(F,left_on=["symbol","d"],right_on=["symbol","date"],how="left")
print("year |  n  | IC(dist20low,pnl) | IC(dist_ma20,pnl) | mean pnl")
for yr in range(2020,2027):
    sub=m[m.yr==yr].dropna(subset=["pnl_pct"])
    if len(sub)<30: continue
    ic1=sub["dist20low"].corr(sub["pnl_pct"],method="spearman")
    ic2=sub["dist_ma20"].corr(sub["pnl_pct"],method="spearman")
    print(f"{yr} | {len(sub):4d} | {ic1:+.3f}            | {ic2:+.3f}            | {sub.pnl_pct.mean()*100:+.2f}%")
print("PERYEAR_DONE")
