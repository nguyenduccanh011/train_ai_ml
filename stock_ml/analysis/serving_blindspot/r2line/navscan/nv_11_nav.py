# -*- coding: utf-8 -*-
"""nv_11_nav: cham NAV chuan v2 (nh_nav2) cho 15 ung vien + gb anchor.

Che do: settle_lag=2 + advance_fee=0.0008, R=0.006, K=25, shuffle-mean±sd 20 perm.
Frame: full (2020-01-01) + f22 (2022-01-01).
Ket qua: nv_11_results.csv (append tung dong, resume duoc).
"""
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "na_audit"))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"

CAND = [  # (label, family, csv)
    ("gb_x08_anchor", "anchor", GB),
    ("n2_2429_base", "2429-base", HERE / "nv_n2_consw20_conv04_vg_combo_hb_nbpbw_s42_trades.csv"),
    ("n2_2429_maxhold20", "2429-maxhold", HERE / "nv_n2_2429_maxhold20_cagr_s555_trades.csv"),
    ("n2_2429_dyncsr88", "2429-dyncsr", HERE / "nv_n2_2429_dyncsr88_s42_trades.csv"),
    ("n2_velov_univ150c", "velov-univ150", HERE / "nv_n2_velov_univ150c_s42_trades.csv"),
    ("sw_ox08_pb03", "unmask-sw", HERE / "nv_sw_ox08_pb03_s42_trades.csv"),
    ("qQ_rsdrop_d15", "unmask-rsdrop", HERE / "nv_qQ_rsdrop_d15_s42_trades.csv"),
    ("n2_v19_fullwave", "v19", HERE / "nv_n2_v19_fullwave_s42_trades.csv"),
    ("n2_am20_oxt03", "oxtrail-fast", HERE / "nv_n2_am20_oxt03_s42_trades.csv"),
    ("n2_mgz13_emgz09", "mgz-gate", HERE / "nv_n2_mgz13_emgz09_s42_trades.csv"),
    ("n2_dz_act05_cap08", "deadzone-trail", HERE / "nv_n2_dz_act05_m10_fl03_cap08_s42_trades.csv"),
    ("n2_pb_xrule_bear3", "champ958-bear", HERE / "nv_n2_pb_xrule_bear3_s42_trades.csv"),
    ("n2_1187_age4", "age-incub", HERE / "nv_n2_1187_age4_s42_trades.csv"),
    ("smac_v52_ridepb", "smac", HERE / "nv_n2_smac_v52_gruraw_vfast_ridepb_s42_trades.csv"),
    ("velexit_contE_h10", "cont-era", HERE / "nv_velexit_contE_h10_u3_e020_x040_s42_trades.csv"),
    ("zzf6_zzpk_p06", "zigzag", HERE / "nv_zzf6_zzpk_p06_t5_e48_x52_s42_trades.csv"),
]

OUT = HERE / "nv_11_results.csv"
done = set()
if OUT.exists():
    done = set(pd.read_csv(OUT).label)

for label, fam, csv in CAND:
    if label in done:
        print(f"skip (done): {label}", flush=True)
        continue
    t0 = time.time()
    row = dict(label=label, family=fam)
    for frame, lo in (("full", "2020-01-01"), ("f22", "2022-01-01")):
        sim = NavSim2(str(csv), date_lo=lo)
        st = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2,
                           advance_fee=0.0008, n=20)
        row[f"{frame}_mean"] = round(st["mean"], 3)
        row[f"{frame}_sd"] = round(st["sd"], 3)
        row[f"{frame}_dd_mean"] = round(st["dd_mean"], 4)
        row[f"{frame}_dd_worst"] = round(st["dd_worst"], 4)
        if frame == "full":
            row["skipped_db"] = sim.skipped_db
            row["n_trades"] = len(sim.trades)
            m0 = sim.run(K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008)
            row["fills"] = m0["fills"]
            row["skip_cash"] = m0["skip_cash"]
            row["turnover"] = round(m0["turnover"], 2)
            row["avg_open"] = round(m0["avg_open"], 1)
    df = pd.DataFrame([row])
    df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"{label:<20} fam={fam:<14} full x{row['full_mean']:.2f}±{row['full_sd']:.2f} "
          f"DD {row['full_dd_mean']*100:.1f}% (worst {row['full_dd_worst']*100:.1f}%) "
          f"f22 x{row['f22_mean']:.2f}±{row['f22_sd']:.2f} fills={row['fills']} "
          f"skipdb={row['skipped_db']} ({time.time()-t0:.0f}s)", flush=True)
print("NV11_DONE")
