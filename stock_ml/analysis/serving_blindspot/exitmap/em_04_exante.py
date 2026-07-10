# -*- coding: utf-8 -*-
"""exitmap step 4: separator do tai bar SIGNAL (ex-ante cho conviction-scaling)."""
import sqlite3

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"

tr = pd.read_csv(f"{OUT}\\gbx08_enriched_em.csv")
con = sqlite3.connect(DB)
px = pd.read_sql("select symbol,date,close from ohlcv order by symbol,date", con)
con.close()

A = {}
for s, g in px.groupby("symbol"):
    g = g.reset_index(drop=True)
    c = g.close.to_numpy(float)
    ret = np.diff(np.log(np.where(c > 0, c, np.nan)), prepend=np.nan)
    rm = pd.Series(ret).rolling(20).mean().to_numpy()
    rs = pd.Series(ret).rolling(20).std().to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    A[s] = dict(idx={d: i for i, d in enumerate(g.date)}, c=c,
                snr=rm / (rs + 1e-12), ma20=ma20)

snr_s, dma_s = [], []
for r in tr.itertuples():
    a = A[r.symbol]
    si = a["idx"][r.entry_signal_date]
    snr_s.append(a["snr"][si])
    dma_s.append(a["c"][si] / a["ma20"][si] - 1)
tr["snr_sig"] = snr_s
tr["dma20_sig"] = dma_s

for f in ["snr_sig", "dma20_sig"]:
    q = pd.qcut(tr[f], 5, labels=False, duplicates="drop")
    t = tr.groupby(q).agg(n=("pnl", "size"), u=("pnl", "sum"), mean=("pnl", "mean"),
                          WR=("pnl", lambda x: (x > 0).mean()))
    print(f"-- {f} (tai bar SIGNAL, ex-ante):")
    print(t.round(4).to_string())

# on nhat theo nam? Q4 snr_sig per year
q = pd.qcut(tr.snr_sig, 5, labels=False)
tr["snrq"] = q
pv = tr.pivot_table(index="year", columns="snrq", values="pnl", aggfunc="mean")
print("\nmean pnl theo nam x snr_sig quintile:")
print(pv.round(3).to_string())
tr.to_csv(f"{OUT}\\gbx08_enriched_em.csv", index=False)
