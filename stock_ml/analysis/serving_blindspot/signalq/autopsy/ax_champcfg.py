import psycopg2, json
con = psycopg2.connect(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cur = con.cursor()
cur.execute("SELECT engine_config FROM strategy_templates WHERE id=2646")
eng = cur.fetchone()[0]
eng = json.loads(eng) if isinstance(eng, str) else eng
for k in sorted(eng):
    if any(t in k for t in ("trail", "overext", "snr", "hold", "protect")):
        print(k, "=", eng[k])
con.close()
