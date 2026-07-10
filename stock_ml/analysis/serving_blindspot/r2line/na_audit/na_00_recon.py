# -*- coding: utf-8 -*-
"""na_00_recon: kiem tra tinh toan ven du lieu dau vao truoc khi audit.
- Doi chieu pnl_pct vs entry/exit price (cost model nhung trong so): pnl = exit/entry - 1 - 0.004?
- Dem trade, hold, exit_reason mix cho c2 / base / gb.
- Kiem coverage ngay entry/exit trong ohlcv.db.
"""
import sqlite3

import pandas as pd

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
DB = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"

FEE = 0.004  # 2*0.0015 + 0.001


def probe(name, df):
    df = df.copy()
    recon = df.exit_price / df.entry_price - 1.0 - FEE
    err = (recon - df.pnl_pct).abs()
    print(f"[{name}] n={len(df)} dup(sym+entry)={df.duplicated(['symbol','entry_date']).sum()}")
    print(f"  pnl==exit/entry-1-0.004: max_err={err.max():.2e} n_err>1e-9={(err > 1e-9).sum()}")
    hd = pd.to_numeric(df.holding_days, errors="coerce")
    print(f"  hold: mean={hd.mean():.1f} median={hd.median():.0f} p95={hd.quantile(.95):.0f}")
    print(f"  pnl: mean={df.pnl_pct.mean()*100:.2f}% win%={(df.pnl_pct>0).mean()*100:.1f}")
    if "exit_reason" in df:
        print("  exit_reason:", df.exit_reason.value_counts().to_dict())
    yr = df.entry_date.astype(str).str[:4]
    print("  entries/nam:", yr.value_counts().sort_index().to_dict())
    return df


c2 = probe("c2_pb40snr", pd.read_csv(f"{R2}/r2_c2_pb40snr_s42_trades.csv"))
base = probe("r2_base", pd.read_csv(f"{R2}/r2_base_s42_trades.csv"))
gb_raw = pd.read_csv(GB)
print("\ngb columns run_id:", gb_raw.run_id.unique()[:3], "n=", len(gb_raw))
gb = probe("gb_x08", gb_raw)

# DB coverage
con = sqlite3.connect(DB)
tabs = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)
print("\nDB tables:", tabs.name.tolist())
cols = pd.read_sql_query("PRAGMA table_info(ohlcv)", con)
print("ohlcv cols:", cols.name.tolist())
for name, df in [("c2", c2), ("gb", gb)]:
    syms = sorted(set(df.symbol))
    px = pd.read_sql_query(
        "SELECT symbol, date FROM ohlcv WHERE symbol IN (%s)" % ",".join("?" * len(syms)),
        con, params=syms)
    have = set(zip(px.symbol, px.date.astype(str).str[:10]))
    miss_e = sum((s, str(d)[:10]) not in have for s, d in zip(df.symbol, df.entry_date))
    miss_x = sum((s, str(d)[:10]) not in have for s, d in zip(df.symbol, df.exit_date))
    print(f"[{name}] missing entry_date in db: {miss_e}, exit_date: {miss_x}")
# co VNINDEX khong?
idx = pd.read_sql_query("SELECT DISTINCT symbol FROM ohlcv WHERE symbol LIKE '%INDEX%' OR symbol LIKE 'VN%' LIMIT 20", con)
print("index-like symbols:", idx.symbol.tolist())
con.close()
