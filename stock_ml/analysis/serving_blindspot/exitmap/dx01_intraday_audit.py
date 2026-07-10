# -*- coding: utf-8 -*-
"""Audit: is VN30F1M 'intraday' actually intraday, or 1 bar/day collapsed?"""
import duckdb

con = duckdb.connect(r"f:\PROJECTS\train_ai_ml\market_data\market.duckdb", read_only=True)
# rows per (symbol,timeframe,date) distribution
r = con.execute("""
SELECT symbol, timeframe, count(*) AS n, count(DISTINCT date) AS ndates,
       max(cnt) AS max_per_date
FROM (SELECT symbol, timeframe, date, count(*) OVER (PARTITION BY symbol, timeframe, date) AS cnt
      FROM ohlcv WHERE symbol IN ('VN30F1M','VN30F2M'))
GROUP BY 1,2 ORDER BY 1,2""").fetchall()
for row in r:
    print(row)
# dates per year for 1m
print(con.execute("""
SELECT year(date), count(*) FROM ohlcv
WHERE symbol='VN30F1M' AND timeframe='1m' GROUP BY 1 ORDER BY 1""").fetchall())
# is the single daily bar == daily bar? compare 1m vs 1D on same dates
print(con.execute("""
SELECT a.date, a.open, a.close, a.volume, b.open, b.close, b.volume
FROM ohlcv a JOIN ohlcv b ON a.date=b.date AND a.symbol=b.symbol
WHERE a.symbol='VN30F1M' AND a.timeframe='1m' AND b.timeframe='1D'
ORDER BY a.date DESC LIMIT 8""").fetchall())
# 1D coverage
print(con.execute("""
SELECT symbol, timeframe, min(date), max(date), count(*) FROM ohlcv
WHERE symbol IN ('VN30F1M','VN30F2M') AND timeframe='1D' GROUP BY 1,2""").fetchall())
con.close()
