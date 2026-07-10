import duckdb
con = duckdb.connect("market_data/market.duckdb", read_only=True)
print(con.execute("SELECT DISTINCT symbol FROM ohlcv WHERE symbol ILIKE '%INDEX%' OR symbol ILIKE '%VNI%' OR length(symbol)>3 LIMIT 30").fetchall())
print(con.execute("SELECT min(date), max(date), count(DISTINCT symbol) FROM ohlcv WHERE timeframe='1D'").fetchall())
print(con.execute("SELECT DISTINCT timeframe FROM ohlcv").fetchall())
con.close()
