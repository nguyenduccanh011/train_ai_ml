import duckdb
con = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
print(con.execute("SELECT date, open, high, close FROM ohlcv WHERE symbol='DPM' AND timeframe='1D' AND date BETWEEN '2020-04-06' AND '2020-04-09'").fetchall())
print(con.execute("SELECT DISTINCT timeframe FROM ohlcv").fetchall())
print(con.execute("SELECT count(DISTINCT symbol) FROM ohlcv WHERE timeframe='1D'").fetchall())
con.close()
