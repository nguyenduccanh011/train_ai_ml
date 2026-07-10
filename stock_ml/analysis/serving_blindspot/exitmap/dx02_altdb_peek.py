# -*- coding: utf-8 -*-
"""Check market.duckdb.bak & market_raw_api.duckdb for true intraday; ohlcv.db for index."""
import duckdb, sqlite3

for p in (r"f:\PROJECTS\train_ai_ml\market_data\market.duckdb.bak",
          r"f:\PROJECTS\train_ai_ml\market_data\market_raw_api.duckdb"):
    print("=" * 20, p)
    try:
        con = duckdb.connect(p, read_only=True)
        print("tables:", con.execute("SHOW TABLES").fetchall())
        for (t,) in con.execute("SHOW TABLES").fetchall():
            cols = [c[0] for c in con.execute(f"DESCRIBE {t}").fetchall()]
            print(t, cols, con.execute(f"SELECT count(*) FROM {t}").fetchone())
            if "timeframe" in cols and "symbol" in cols:
                r = con.execute(f"""
                    SELECT symbol, timeframe, count(*), min(date), max(date),
                           count(DISTINCT date)
                    FROM {t} WHERE symbol IN ('VN30F1M','VN30F2M')
                    GROUP BY 1,2 ORDER BY 1,2""").fetchall()
                for row in r:
                    print("  ", row)
                # date/time column types
                print("  date type:", [c for c in con.execute(f"DESCRIBE {t}").fetchall() if c[0] in ("date", "time", "ts", "timestamp")])
        con.close()
    except Exception as e:
        print("ERR", e)

print("=" * 20, "ohlcv.db index syms")
con = sqlite3.connect(r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db")
print([r for r in con.execute("select name from sqlite_master where type='table'")])
print([r[0] for r in con.execute("select distinct symbol from ohlcv where length(symbol)>3 or symbol like 'VN3%' or symbol like '%INDEX%'")])
con.close()
