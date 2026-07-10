# -*- coding: utf-8 -*-
"""r2c_30: kiem tra on dinh theo K (22/25/28, adv R0.6) cho cac ung vien —
f22-gai tai p42 co giu thu hang o K khac khong? (khong can run backtest moi)."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "na_audit"))
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

ADV = 0.0008
CANDS = ["r2b_oxtrail04", "r2c_oxt03", "r2c_oxt04_p41", "r2c_oxt04_p42", "r2c_oxt04_p43"]
GBCSV = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"

print("diem | K | full mean±sd (DDmean) | f22 mean±sd")
gb_f = NavSim2(GBCSV, "2020-01-01")
gb_2 = NavSim2(GBCSV, "2022-01-01")
for K in (22, 25, 28):
    sf = shuffle_stats(gb_f, K=K, roundtrip=0.006, settle_lag=2, advance_fee=ADV)
    s2 = shuffle_stats(gb_2, K=K, roundtrip=0.006, settle_lag=2, advance_fee=ADV)
    print(f"gb | {K} | x{sf['mean']:.2f}±{sf['sd']:.2f} ({sf['dd_mean']*100:.1f}%) | "
          f"x{s2['mean']:.2f}±{s2['sd']:.2f}", flush=True)
for name in CANDS:
    csv = HERE / f"{name}_s42_trades.csv"
    sim_f = NavSim2(str(csv), "2020-01-01")
    sim_2 = NavSim2(str(csv), "2022-01-01")
    for K in (22, 25, 28):
        sf = shuffle_stats(sim_f, K=K, roundtrip=0.006, settle_lag=2, advance_fee=ADV)
        s2 = shuffle_stats(sim_2, K=K, roundtrip=0.006, settle_lag=2, advance_fee=ADV)
        print(f"{name} | {K} | x{sf['mean']:.2f}±{sf['sd']:.2f} ({sf['dd_mean']*100:.1f}%) | "
              f"x{s2['mean']:.2f}±{s2['sd']:.2f}", flush=True)
print("DONE", flush=True)
