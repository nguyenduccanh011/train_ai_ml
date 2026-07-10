# -*- coding: utf-8 -*-
"""pr3_11: tội 1 (tiếp) — lát KHÔNG dùng khi chọn: f21 (từ 2021), f24 (từ 2024),
f25 (từ 2025). Selection dùng full+f22 (f23 check). n=20 perm, K25 adv R0.6.
Nếu p42 chỉ đẹp ở lát chọn lọc mà chết lát held-out -> tội thành lập."""
import sys
import math

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

R2 = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"

pts = {
    "gb_x08": GB,
    "r2c_oxt04_p42": f"{R2}/r2c_oxt04_p42_s42_trades.csv",
    "r2c_oxt04_p41": f"{R2}/r2c_oxt04_p41_s42_trades.csv",
    "r2c_oxt04_p43": f"{R2}/r2c_oxt04_p43_s42_trades.csv",
    "r2c_oxt03": f"{R2}/r2c_oxt03_s42_trades.csv",
    "r2_c2_pb40snr": f"{R2}/r2_c2_pb40snr_s42_trades.csv",
    "r2_base": f"{R2}/r2_base_s42_trades.csv",
}
frames = {"f21": "2021-01-01", "f24": "2024-01-01", "f25": "2025-01-01"}

res = {}
for fname, lo in frames.items():
    for name, csv in pts.items():
        s = shuffle_stats(NavSim2(csv, date_lo=lo), K=25, advance_fee=0.0008, n=20)
        res[(fname, name)] = s
        print(f"{fname} {name}: x{s['mean']:.3f}±{s['sd']:.3f} "
              f"(DD {s['dd_mean']*100:.1f}/{s['dd_worst']*100:.1f})", flush=True)

print("\n=== DELTA vs gb (sd lan truyền) ===")
for fname in frames:
    g = res[(fname, "gb_x08")]
    for name in pts:
        if name == "gb_x08":
            continue
        s = res[(fname, name)]
        d = (s["mean"] / g["mean"] - 1) * 100
        sd = math.sqrt((s["sd"] / g["mean"]) ** 2
                       + (g["sd"] * s["mean"] / g["mean"] ** 2) ** 2) * 100
        print(f"{fname} {name}: {d:+.1f}%±{sd:.1f} ({d/sd:+.1f}sd)")
print("DONE")
