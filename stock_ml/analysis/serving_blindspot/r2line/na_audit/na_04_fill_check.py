# -*- coding: utf-8 -*-
"""na_04: fill/price consistency.
- entry_raw = entry_price/1.001 phai nam trong [low,high] ngay entry (limit fill kha thi).
- exit_raw = exit_price/0.999 doi chieu close ngay exit (exit ATC/close?) — phan phoi
  exit_raw/close toan bo trade: lech nhieu = adjustment mismatch hoac fill gia la.
- In 8 lenh doi chieu tay voi OHLC.
"""
import sqlite3

import pandas as pd

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
DB = "C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
S0 = 0.001


def check(name, csv):
    df = pd.read_csv(csv)
    df["ed"] = df.entry_date.astype(str).str[:10]
    df["xd"] = df.exit_date.astype(str).str[:10]
    syms = sorted(set(df.symbol))
    con = sqlite3.connect(DB)
    px = pd.read_sql_query(
        "SELECT symbol, date, open, high, low, close FROM ohlcv WHERE symbol IN (%s)"
        % ",".join("?" * len(syms)), con, params=syms)
    con.close()
    px["date"] = px.date.astype(str).str[:10]
    key = px.set_index(["symbol", "date"])

    df["entry_raw"] = df.entry_price / (1 + S0)
    df["exit_raw"] = df.exit_price / (1 - S0)
    e = df.join(key, on=["symbol", "ed"], rsuffix="_e")
    x = df.join(key, on=["symbol", "xd"], rsuffix="_x")

    in_range = ((e.entry_raw >= e.low * 0.9995) & (e.entry_raw <= e.high * 1.0005))
    print(f"[{name}] n={len(df)}")
    print(f"  entry_raw trong [low,high] ngay entry: {in_range.mean()*100:.1f}%"
          f" (ngoai bien: {(~in_range).sum()})")
    er_close = e.entry_raw / e.close
    print(f"  entry_raw/close_entry: p05={er_close.quantile(.05):.4f} med={er_close.median():.4f}"
          f" p95={er_close.quantile(.95):.4f}  (<1 = mua duoi close, passive limit)")
    er_low = (e.entry_raw - e.low).abs() / e.low
    print(f"  ty le entry_raw == low (±0.1%): {(er_low < 0.001).mean()*100:.1f}%  <- fill dung day = lac quan")
    xr_close = x.exit_raw / x.close
    print(f"  exit_raw/close_exit: p05={xr_close.quantile(.05):.4f} med={xr_close.median():.4f}"
          f" p95={xr_close.quantile(.95):.4f} | ==close(±0.1%): {((xr_close-1).abs()<0.001).mean()*100:.1f}%")
    xin = ((x.exit_raw >= x.low * 0.9995) & (x.exit_raw <= x.high * 1.0005))
    print(f"  exit_raw trong [low,high] ngay exit: {xin.mean()*100:.1f}%")
    if "exit_reason" in df:
        for reason, g in x.groupby(df.exit_reason):
            r = g.exit_raw / g.close
            print(f"    exit={reason:15s} n={len(g):4d} med(exit/close)={r.median():.4f}"
                  f" ==close: {((r-1).abs()<0.001).mean()*100:5.1f}%")
    return e, x


e_c2, x_c2 = check("c2_pb40snr", f"{R2}/r2_c2_pb40snr_s42_trades.csv")
e_gb, x_gb = check("gb_x08", GB)

print("\n=== 8 lenh c2 doi chieu tay ===")
samp = e_c2.sample(8, random_state=7)
cols = ["symbol", "ed", "entry_raw", "open", "high", "low", "close", "exit_reason"]
print(samp[cols].to_string(index=False))
print("\n(exit side)")
sx = x_c2.loc[samp.index]
print(sx[["symbol", "xd", "exit_raw", "open_x" if "open_x" in sx else "open", "low", "high", "close", "pnl_pct"]].head(8).to_string(index=False))
