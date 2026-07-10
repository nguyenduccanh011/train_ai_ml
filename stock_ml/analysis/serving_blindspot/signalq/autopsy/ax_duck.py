import duckdb
con = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
print(con.execute("SHOW TABLES").fetchall())
for t in [r[0] for r in con.execute("SHOW TABLES").fetchall()][:5]:
    print(t, con.execute(f"DESCRIBE {t}").fetchall())
con.close()
