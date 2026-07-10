# -*- coding: utf-8 -*-
"""nv_03_check_trades: ung vien nao con trades trong run_trades? (runs cu co the bi xoa)"""
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

CAND = {  # template_id: name
    2516: "n2_2429_maxhold20_cagr", 2531: "n2_2429_dyncsr88",
    2437: "n2_2429_exit_vol_market_downleg", 2234: "n2_velov_univ150c",
    2222: "sw_ox08_pb03", 2040: "qQ_rsdrop_d15", 1799: "n2_v19_fullwave",
    1410: "n2_am20_oxt03", 1306: "n2_mgz13_emgz09", 1644: "n2_dz_act05_m10_fl03_cap08",
    971: "n2_pb_xrule_bear3", 1194: "n2_1187_age4",
    2658: "n2_smac_v52_gruraw_vfast_ridepb", 625: "velexit_contE_h10_u3_e020_x040",
    200: "zzf6_zzpk_p06_t5_e48_x52",
}
con = psycopg2.connect(**PG)
runs = pd.read_sql(
    "select template_id, run_id, run_seed, composite_score, superseded, generated_at "
    "from leaderboard_runs where template_id = any(%s) and market='vn_stock'",
    con, params=(list(CAND),))
for tid, g in runs.groupby("template_id"):
    for _, r in g.iterrows():
        cur = con.cursor()
        cur.execute("select count(*) from run_trades where run_id=%s", (r.run_id,))
        n = cur.fetchone()[0]
        print(f"t{tid} {CAND[tid]:<34} seed={r.run_seed} comp={r.composite_score:.1f} "
              f"sup={r.superseded} trades_in_db={n} run_id={r.run_id}")
con.close()
print("NV03_DONE")
