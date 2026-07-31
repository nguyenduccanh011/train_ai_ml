# -*- coding: utf-8 -*-
"""hb_01: verify anchor — CSV re-dump tu DB phai tai lap so da cong bo
(AB_PROSECUTION.md / pr5_70_out.txt / xg). Neu lech >1% -> dump KHONG hop le."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

# (csv, date_lo, adv_ky_vong, noadv_ky_vong)
ANCHORS = [
    ("gb_t2783_s42_trades.csv", "2020-01-01", 13.78, 13.03),
    ("gb_t2783_s42_trades.csv", "2022-01-01", 3.43, 3.34),
    ("ab_noT_s42_trades.csv", "2020-01-01", 22.87, 20.99),
    ("ab_noT_s42_trades.csv", "2022-01-01", 4.43, 4.22),
    ("ab_noT_s555_trades.csv", "2020-01-01", 22.26, 20.69),
    ("xg_dyn_mh25_s42_trades.csv", "2020-01-01", 21.15, 19.39),
    ("xg_dyn_mh25_s42_trades.csv", "2022-01-01", 4.45, 4.57),
    ("pr5_dynclean_mh25_s42_trades.csv", "2020-01-01", 21.32, 19.48),
    ("pr5_dynclean_mh25_s42_trades.csv", "2022-01-01", 4.56, 4.60),
    ("pr5_dynclean_mh25_s42_trades.csv", "2025-01-01", 1.65, 1.68),
]

n_ok = 0
for csv, lo, ea, en in ANCHORS:
    sim = NavSim2(str(HERE / csv), date_lo=lo)
    a = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
    n = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, n=20)
    ok = abs(a["mean"] / ea - 1) < 0.01 and abs(n["mean"] / en - 1) < 0.01
    n_ok += ok
    print(f"{csv[:32]:34s} {lo[:4]}: adv x{a['mean']:.2f} (kv x{ea:.2f}) "
          f"noadv x{n['mean']:.2f} (kv x{en:.2f}) {'OK' if ok else '***FAIL***'}",
          flush=True)
print(f"\n{n_ok}/{len(ANCHORS)} anchor khop. HB_01_DONE")
