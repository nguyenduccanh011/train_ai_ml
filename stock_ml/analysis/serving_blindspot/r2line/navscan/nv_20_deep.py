# -*- coding: utf-8 -*-
"""nv_20_deep: dao sau top ung vien (NAV >= gb+8%).

Moi ung vien: (a) f23 frame, (b) LOYO nhanh (bo 2021), (c) drop-top-20 winner,
(d) overlap co che voi fc_rule2 (r2c_oxt04_p42) va gb_x08: exact (symbol, entry_date)
va fuzzy ±5d — tra loi "cung cong thuc vong-quay-nhanh hay nguon alpha khac".

Usage: python nv_20_deep.py <label1> [label2 ...]
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "na_audit"))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
R2C = str(HERE.parent / "r2c_oxt04_p42_s42_trades.csv")

CSVS = {
    "n2_2429_base": HERE / "nv_n2_consw20_conv04_vg_combo_hb_nbpbw_s42_trades.csv",
    "n2_2429_maxhold20": HERE / "nv_n2_2429_maxhold20_cagr_s555_trades.csv",
    "n2_2429_dyncsr88": HERE / "nv_n2_2429_dyncsr88_s42_trades.csv",
    "n2_velov_univ150c": HERE / "nv_n2_velov_univ150c_s42_trades.csv",
    "sw_ox08_pb03": HERE / "nv_sw_ox08_pb03_s42_trades.csv",
    "qQ_rsdrop_d15": HERE / "nv_qQ_rsdrop_d15_s42_trades.csv",
    "n2_v19_fullwave": HERE / "nv_n2_v19_fullwave_s42_trades.csv",
    "n2_am20_oxt03": HERE / "nv_n2_am20_oxt03_s42_trades.csv",
    "n2_mgz13_emgz09": HERE / "nv_n2_mgz13_emgz09_s42_trades.csv",
    "n2_dz_act05_cap08": HERE / "nv_n2_dz_act05_m10_fl03_cap08_s42_trades.csv",
    "n2_pb_xrule_bear3": HERE / "nv_n2_pb_xrule_bear3_s42_trades.csv",
    "n2_1187_age4": HERE / "nv_n2_1187_age4_s42_trades.csv",
}


def nav(csv, lo, drop_year=None, drop_top=0):
    sim = NavSim2(str(csv), date_lo=lo)
    if drop_year or drop_top:
        tr = sim.trades
        if drop_year:
            tr = [t for t in tr if t["entry_date"][:4] != drop_year]
        if drop_top:
            # net per-trade da tinh o run(); dung pnl tho tu gia raw R0.6 xap xi
            for t in tr:
                t["_net"] = t["x_raw"] * (1 - 0.001) / (t["e_raw"] * (1 + 0.001)) - 1 - 0.004
            tr = sorted(tr, key=lambda t: t["_net"], reverse=True)[drop_top:]
        sim.trades = tr
    return shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)


def overlap(csv, ref_csv):
    a = pd.read_csv(csv)
    b = pd.read_csv(ref_csv)
    a["ed"] = a.entry_date.astype(str).str[:10]
    b["ed"] = b.entry_date.astype(str).str[:10]
    ka = set(zip(a.symbol, a.ed))
    kb = set(zip(b.symbol, b.ed))
    exact = len(ka & kb) / len(ka)
    bd = {}
    for s, d in kb:
        bd.setdefault(s, []).append(pd.Timestamp(d))
    fz = 0
    for s, d in ka:
        dts = bd.get(s)
        if dts is not None:
            d = pd.Timestamp(d)
            if any(abs((d - x).days) <= 5 for x in dts):
                fz += 1
    return exact, fz / len(ka)


for label in sys.argv[1:]:
    csv = CSVS[label]
    print(f"\n===== {label} =====", flush=True)
    st = nav(csv, "2023-01-01")
    print(f"f23:        x{st['mean']:.2f}±{st['sd']:.2f} DD {st['dd_mean']*100:.1f}%", flush=True)
    st = nav(csv, "2020-01-01", drop_year="2021")
    print(f"LOYO-2021:  x{st['mean']:.2f}±{st['sd']:.2f}", flush=True)
    st = nav(csv, "2020-01-01", drop_top=20)
    print(f"drop-top20: x{st['mean']:.2f}±{st['sd']:.2f}", flush=True)
    ex, fz = overlap(csv, R2C)
    print(f"overlap vs r2c_oxt04_p42: exact {ex*100:.1f}% fuzzy±5d {fz*100:.1f}%", flush=True)
    ex, fz = overlap(csv, GB)
    print(f"overlap vs gb_x08:        exact {ex*100:.1f}% fuzzy±5d {fz*100:.1f}%", flush=True)
    t = pd.read_csv(csv)
    t["yr"] = t.entry_date.astype(str).str[:4]
    print("per-year pnl:", t.groupby("yr").pnl_pct.sum().round(1).to_dict(), flush=True)
    print("hold median:", t.holding_days.median(), " trades:", len(t), flush=True)
print("NV20_DONE")
