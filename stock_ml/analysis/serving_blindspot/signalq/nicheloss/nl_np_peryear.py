import psycopg2, pandas as pd
con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
runs = {"np_atmkt":"template/np_atmkt-32a8dfee","np_pb02":"template/np_pb02-32a8dfee",
        "np_pb03":"template/np_pb03-32a8dfee","op_pb02":"template/op_pb02-32a8dfee",
        "op_pb03":"template/op_pb03-32a8dfee","gb_x08":"template/gb_x08-32a8dfee",
        "a2_nopb":"template/a2_nopb-32a8dfee","champ":"template/n2_2643_wavestruct_la05_lamp02-32a8dfee"}
tabs={}
for k,rid in runs.items():
    df = pd.read_sql("SELECT entry_date, pnl_pct FROM run_trades WHERE run_id=%s", con, params=(rid,))
    if len(df)==0: print("MISSING", k); continue
    df["entry_date"]=pd.to_datetime(df["entry_date"])
    tabs[k]=df.groupby(df.entry_date.dt.year).pnl_pct.sum()
con.close()
print(pd.DataFrame(tabs).fillna(0).round(2).to_string())
