# -*- coding: utf-8 -*-
"""R2 step 1: xem gia tri snr_extend/giveback cua gb_x08 (t2783) va cac template co key nay."""
import json
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
cur = con.cursor()
cur.execute("""
SELECT id, name,
       engine_config->>'exit_snr_extend_threshold',
       engine_config->>'exit_snr_extend_window',
       engine_config->>'exit_snr_min_gain',
       engine_config->>'exit_snr_defer_min_giveback'
FROM strategy_templates
WHERE engine_config->>'exit_snr_extend_threshold' IS NOT NULL
ORDER BY id DESC LIMIT 30""")
for r in cur.fetchall():
    print(r)
print("--- gb_x08 t2783 keys lien quan:")
cur.execute("SELECT engine_config FROM strategy_templates WHERE id=2783")
eng = cur.fetchone()[0]
eng = json.loads(eng) if isinstance(eng, str) else eng
for k in sorted(eng):
    if "snr" in k or "giveback" in k or "trail" in k or "overext" in k:
        print(f"  {k} = {eng[k]}")
con.close()
