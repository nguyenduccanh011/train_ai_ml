# -*- coding: utf-8 -*-
"""nv_10_dump: dump trades 16 ung vien tu run_trades -> nv_<name>_s<seed>_trades.csv.

Chon run KHONG superseded uu tien seed 42 (t2516 chi co s555).
"""
from pathlib import Path

import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent

CAND = [  # (template_id, name, family-ghi-chu)
    (2429, "n2_consw20_conv04_vg_combo_hb_nbpbw", "2429-base"),
    (2516, "n2_2429_maxhold20_cagr", "2429-maxhold"),
    (2531, "n2_2429_dyncsr88", "2429-dyncsr"),
    (2234, "n2_velov_univ150c", "velov-univ150"),
    (2222, "sw_ox08_pb03", "unmask-sw"),
    (2040, "qQ_rsdrop_d15", "unmask-rsdrop"),
    (1799, "n2_v19_fullwave", "v19"),
    (1410, "n2_am20_oxt03", "oxtrail-fast"),
    (1306, "n2_mgz13_emgz09", "mgz-gate"),
    (1644, "n2_dz_act05_m10_fl03_cap08", "deadzone-trail"),
    (971, "n2_pb_xrule_bear3", "champ958-bear"),
    (1194, "n2_1187_age4", "age-incub"),
    (2658, "n2_smac_v52_gruraw_vfast_ridepb", "smac"),
    (625, "velexit_contE_h10_u3_e020_x040", "cont-era"),
    (200, "zzf6_zzpk_p06_t5_e48_x52", "zigzag"),
]

con = psycopg2.connect(**PG)
for tid, name, fam in CAND:
    runs = pd.read_sql(
        "select run_id, run_seed, composite_score, total_pnl, pf, mdd_per_symbol, trades, avg_hold "
        "from leaderboard_runs where template_id=%s and market='vn_stock' and superseded=false "
        "order by (run_seed=42) desc, composite_score desc limit 1",
        con, params=(tid,))
    r = runs.iloc[0]
    seed = int(r.run_seed)
    csv = HERE / f"nv_{name}_s{seed}_trades.csv"
    tdf = pd.read_sql(
        "select symbol, entry_date, exit_date, entry_price, exit_price, holding_days, "
        "pnl_pct, exit_reason from run_trades where run_id=%s",
        con, params=(r.run_id,))
    tdf.to_csv(csv, index=False)
    print(f"t{tid:<5} {name:<36} fam={fam:<14} seed={seed} comp={r.composite_score:.1f} "
          f"pnl={r.total_pnl:.1f} pf={r.pf:.2f} tr={int(r.trades)} hold={r.avg_hold:.1f} "
          f"dumped={len(tdf)}")
con.close()
print("NV10_DONE")
