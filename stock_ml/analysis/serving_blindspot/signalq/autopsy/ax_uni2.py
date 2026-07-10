import psycopg2, json
con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("SELECT id, slug, name FROM universe_sets")
for r in cur.fetchall(): print(r)
cur.execute("""SELECT uv.universe_id, uv.version, uv.symbol_count FROM universe_versions uv
JOIN universe_sets us ON us.id=uv.universe_id WHERE us.slug='vn_stock_default' ORDER BY uv.version""")
for r in cur.fetchall(): print(r)
con.close()
