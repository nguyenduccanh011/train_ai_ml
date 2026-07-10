# -*- coding: utf-8 -*-
"""Peek: duckdb intraday VN30F1M/F2M + enriched CSV columns."""
import duckdb
import pandas as pd

con = duckdb.connect(r"f:\PROJECTS\train_ai_ml\market_data\market.duckdb", read_only=True)
print("tables:", con.execute("SHOW TABLES").fetchall())
print("timeframes:", con.execute("SELECT DISTINCT timeframe FROM ohlcv").fetchall())
print(con.execute("DESCRIBE ohlcv").fetchall())
for tf in ("1m", "5m", "1"):
    try:
        r = con.execute(
            "SELECT symbol, timeframe, min(date), max(date), count(*) FROM ohlcv "
            f"WHERE timeframe='{tf}' GROUP BY 1,2 ORDER BY 1").fetchall()
        print(tf, r[:20])
    except Exception as e:
        print(tf, "ERR", e)
# sample intraday rows for VN30F1M
for sym in ("VN30F1M", "VN30F2M"):
    try:
        r = con.execute(
            "SELECT * FROM ohlcv WHERE symbol=? AND timeframe NOT IN ('1D') "
            "ORDER BY date LIMIT 5", [sym]).fetchdf()
        print(sym, "head:\n", r)
        r2 = con.execute(
            "SELECT * FROM ohlcv WHERE symbol=? AND timeframe NOT IN ('1D') "
            "ORDER BY date DESC LIMIT 5", [sym]).fetchdf()
        print(sym, "tail:\n", r2)
    except Exception as e:
        print(sym, "ERR", e)
# VN30 spot / VNINDEX daily available?
print("index syms:", con.execute(
    "SELECT DISTINCT symbol FROM ohlcv WHERE length(symbol)>3 OR symbol ILIKE 'VN%'").fetchall())
con.close()

for p in (r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap\gbx08_enriched2.csv",
          r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\exitmap\gbx08_enriched_em.csv"):
    df = pd.read_csv(p, nrows=3)
    print(p.split("\\")[-1], list(df.columns))
    print(df.head(2).to_string())
