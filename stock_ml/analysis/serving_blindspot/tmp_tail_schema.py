import sqlite3
con = sqlite3.connect(r"C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db")
for name, sql in con.execute("select name, sql from sqlite_master where type='table'"):
    print(name)
    print(sql)
print(con.execute("select count(*) from ohlcv").fetchone() if any(
    r[0] == 'ohlcv' for r in con.execute("select name from sqlite_master where type='table'")) else "no ohlcv table")
