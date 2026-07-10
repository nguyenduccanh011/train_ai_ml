# -*- coding: utf-8 -*-
"""exitmap step 1: enrich 1378 trades gb_x08 voi dac trung tai bar signal + bar fill.

Features:
  runup20/60  : close[sig]/close[sig-k]-1 (run-up TRUOC tin hieu)
  dist_ma20/60: close[fill]/MA[fill]-1
  snr_sym     : mean(ret,20)/std(ret,20) tai bar fill (per-symbol trend/noise)
  vol_shock   : volume[fill]/mean(volume[fill-20..fill-1])
  fill_age    : so bar giao dich tu signal den fill (1 = khop ngay dau window)
  mae         : min(low[fill..exit])/entry_price - 1
  mfe         : max(high[fill..exit])/entry_price - 1
  depth10     : min(low[fill..fill+10])/entry_price - 1 (do sau cu giam ngay sau fill)
"""
import sqlite3

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"

tr = pd.read_csv(f"{OUT}\\gbx08_trades.csv")
con = sqlite3.connect(DB)
px = pd.read_sql("select symbol,date,open,high,low,close,volume from ohlcv order by symbol,date", con)
con.close()

A = {}
for s, g in px.groupby("symbol"):
    g = g.reset_index(drop=True)
    c = g.close.to_numpy(float)
    ret = np.diff(np.log(c), prepend=np.nan)
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    ma60 = pd.Series(c).rolling(60).mean().to_numpy()
    rm = pd.Series(ret).rolling(20).mean().to_numpy()
    rs = pd.Series(ret).rolling(20).std().to_numpy()
    v = g.volume.to_numpy(float)
    vma = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    A[s] = dict(idx={d: i for i, d in enumerate(g.date)}, dates=g.date.to_numpy(),
                c=c, l=g.low.to_numpy(float), h=g.high.to_numpy(float),
                ma20=ma20, ma60=ma60, snr=rm / (rs + 1e-12), v=v, vma=vma)

rows = []
miss = 0
for r in tr.itertuples():
    a = A.get(r.symbol)
    if a is None:
        miss += 1
        continue
    si = a["idx"].get(str(r.entry_signal_date))
    fi = a["idx"].get(str(r.entry_date))
    xi = a["idx"].get(str(r.exit_date)) if isinstance(r.exit_date, str) or not pd.isna(r.exit_date) else None
    if si is None or fi is None:
        miss += 1
        continue
    c = a["c"]
    d = dict(symbol=r.symbol, entry_signal_date=str(r.entry_signal_date),
             entry_date=str(r.entry_date), exit_date=str(r.exit_date),
             entry_price=r.entry_price, pnl=r.pnl_pct, hold=r.holding_days,
             exit_reason=r.exit_reason, year=int(str(r.entry_date)[:4]))
    d["fill_age"] = fi - si
    d["runup20"] = c[si] / c[si - 20] - 1 if si >= 20 else np.nan
    d["runup60"] = c[si] / c[si - 60] - 1 if si >= 60 else np.nan
    d["dist_ma20"] = c[fi] / a["ma20"][fi] - 1
    d["dist_ma60"] = c[fi] / a["ma60"][fi] - 1
    d["snr_sym"] = a["snr"][fi]
    d["vol_shock"] = a["v"][fi] / a["vma"][fi] if a["vma"][fi] > 0 else np.nan
    end10 = min(fi + 10, len(c) - 1)
    d["depth10"] = a["l"][fi:end10 + 1].min() / r.entry_price - 1
    if xi is not None and xi >= fi:
        d["mae"] = a["l"][fi:xi + 1].min() / r.entry_price - 1
        d["mfe"] = a["h"][fi:xi + 1].max() / r.entry_price - 1
    else:
        d["mae"] = np.nan
        d["mfe"] = np.nan
    rows.append(d)

df = pd.DataFrame(rows)
print("enriched:", len(df), "miss:", miss)
print(df[["fill_age", "runup20", "runup60", "dist_ma20", "dist_ma60", "snr_sym",
          "vol_shock", "depth10", "mae", "mfe"]].describe().to_string())
df.to_csv(f"{OUT}\\gbx08_enriched_em.csv", index=False)
