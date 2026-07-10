import duckdb
con = duckdb.connect("market_data/market.duckdb", read_only=True)
print(con.execute("SHOW TABLES").fetchall())
for t in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
    print(t, con.execute(f"DESCRIBE {t}").fetchall()[:12])
    print(" rows:", con.execute(f"SELECT count(*) FROM {t}").fetchone())
# any index symbols?
try:
    print(con.execute("SELECT DISTINCT symbol FROM prices WHERE symbol ILIKE '%INDEX%' OR symbol ILIKE 'VN%' LIMIT 20").fetchall())
except Exception as e:
    print("ERR", e)
con.close()
