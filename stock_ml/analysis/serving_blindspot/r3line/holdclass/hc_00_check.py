# -*- coding: utf-8 -*-
"""hc_00_check: kiem tra nhat quan gia CSV trades vs ohlcv.db close
truoc khi lam counterfactual giu-tiep (hold-class oracle).

- CSV entry/exit price = close * (1 +/- S0) theo quy uoc nh_nav2 (S0=0.001).
- Neu db close da adjusted nhat quan voi engine -> ratio CSV/db ~ hang so
  trong tung trade -> extension bang close[new_i1]/close[i1] la hop le.
"""
import sqlite3
import pandas as pd
import numpy as np

DB = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
CSV = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r3line/r3_mh16_s42_trades.csv"
S0 = 0.001

df = pd.read_csv(CSV)
df["entry_date"] = df["entry_date"].astype(str).str[:10]
df["exit_date"] = df["exit_date"].astype(str).str[:10]
syms = sorted(set(df.symbol))
con = sqlite3.connect(DB)
px = pd.read_sql_query(
    "SELECT symbol,date,close,volume FROM ohlcv WHERE symbol IN (%s)"
    % ",".join("?" * len(syms)), con, params=syms)
con.close()

closes, idx = {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date")
    closes[s] = g["close"].to_numpy()
    idx[s] = {d: i for i, d in enumerate(g["date"])}

rows = []
for r in df.itertuples():
    s = r.symbol
    i0 = idx.get(s, {}).get(r.entry_date)
    i1 = idx.get(s, {}).get(r.exit_date)
    if i0 is None or i1 is None:
        rows.append((s, r.entry_date, np.nan, np.nan, np.nan))
        continue
    e_raw = r.entry_price / (1 + S0)
    x_raw = r.exit_price / (1 - S0)
    ratio0 = e_raw / closes[s][i0]
    ratio1 = x_raw / closes[s][i1]
    pnl_db = closes[s][i1] / closes[s][i0] - 1
    rows.append((s, r.entry_date, ratio0, ratio1, pnl_db))

chk = pd.DataFrame(rows, columns=["symbol", "entry_date", "ratio0", "ratio1", "pnl_db"])
chk["pnl_csv"] = (df.exit_price / df.entry_price - 1).values
chk["ratio_drift"] = chk.ratio1 / chk.ratio0 - 1

print("n trades:", len(chk), " missing db:", chk.ratio0.isna().sum())
print("\nratio0 (e_raw/close_entry) describe:")
print(chk.ratio0.describe())
print("\nratio_drift (ratio1/ratio0 - 1) describe:  # ~0 => close db khop engine adj")
print(chk.ratio_drift.abs().describe())
print("\nso trade |drift| > 1%:", (chk.ratio_drift.abs() > 0.01).sum())
print("so trade |drift| > 5%:", (chk.ratio_drift.abs() > 0.05).sum())
bad = chk[chk.ratio_drift.abs() > 0.05]
print(bad.head(15).to_string())

# du lieu extension: capped trades con bao nhieu bar sau exit
cap = df[df.exit_reason == "max_hold"].copy()
avail = []
for r in cap.itertuples():
    s = r.symbol
    i1 = idx.get(s, {}).get(r.exit_date)
    avail.append(len(closes[s]) - 1 - i1 if i1 is not None else -1)
cap["bars_after"] = avail
print("\ncapped:", len(cap))
print("bars_after >= 24 (cap40):", (cap.bars_after >= 24).sum())
print("bars_after >= 44 (cap60):", (cap.bars_after >= 44).sum())
print("holding_days capped describe:", cap.holding_days.describe().to_string())
