# -*- coding: utf-8 -*-
"""pr3_10: CONG TO tội 1 — phân phối TOÀN BỘ điểm đã thử (3 vòng r2/r2b/r2c)
trên thước nh_nav2 K25 adv (n=10 perm để quét; ứng viên + gb n=20).
Câu hỏi: r2c_oxt04_p42 là đuôi may mắn của phân phối hẹp hay outlier thật?"""
import glob
import os
import sys
import statistics

sys.path.insert(0, "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line/na_audit")
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

R2DIR = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/r2line"
GB = "f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap/gbx08_enriched2.csv"
OUT = os.path.join(R2DIR, "pr3", "pr3_10_scores.csv")

csvs = sorted(glob.glob(os.path.join(R2DIR, "*_s42_trades.csv")))
names = [os.path.basename(c).replace("_s42_trades.csv", "") for c in csvs]
print(f"{len(csvs)} config da thu (3 vong)")

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f.readlines()[1:]:
            done.add(line.split(",")[0])
else:
    with open(OUT, "w") as f:
        f.write("name,full_mean,full_sd,f22_mean,f22_sd\n")

rows = []
for name, csv in zip(names, csvs):
    if name in done:
        continue
    n = 20 if name == "r2c_oxt04_p42" else 10
    s_full = shuffle_stats(NavSim2(csv, date_lo="2020-01-01"), K=25,
                           advance_fee=0.0008, n=n)
    s_f22 = shuffle_stats(NavSim2(csv, date_lo="2022-01-01"), K=25,
                          advance_fee=0.0008, n=n)
    with open(OUT, "a") as f:
        f.write(f"{name},{s_full['mean']:.4f},{s_full['sd']:.4f},"
                f"{s_f22['mean']:.4f},{s_f22['sd']:.4f}\n")
    print(f"{name}: full x{s_full['mean']:.2f}±{s_full['sd']:.2f}  "
          f"f22 x{s_f22['mean']:.2f}±{s_f22['sd']:.2f}", flush=True)

# gb baseline n=20
if "gb_x08" not in done:
    s_full = shuffle_stats(NavSim2(GB, date_lo="2020-01-01"), K=25, advance_fee=0.0008, n=20)
    s_f22 = shuffle_stats(NavSim2(GB, date_lo="2022-01-01"), K=25, advance_fee=0.0008, n=20)
    with open(OUT, "a") as f:
        f.write(f"gb_x08,{s_full['mean']:.4f},{s_full['sd']:.4f},"
                f"{s_f22['mean']:.4f},{s_f22['sd']:.4f}\n")
    print(f"gb_x08: full x{s_full['mean']:.2f}±{s_full['sd']:.2f}  "
          f"f22 x{s_f22['mean']:.2f}±{s_f22['sd']:.2f}", flush=True)

# tổng kết phân phối
import pandas as pd  # noqa: E402
df = pd.read_csv(OUT)
r2 = df[df.name != "gb_x08"].sort_values("full_mean")
gb = df[df.name == "gb_x08"].iloc[0]
print("\n=== PHAN PHOI 3 VONG (delta % vs gb) ===")
for col in ("full_mean", "f22_mean"):
    d = (r2[col] / gb[col] - 1) * 100
    q = d.quantile
    print(f"{col}: n={len(d)} min {d.min():+.1f} q25 {q(.25):+.1f} med {d.median():+.1f} "
          f"q75 {q(.75):+.1f} max {d.max():+.1f} | p42 rank {(d < d[r2.name=='r2c_oxt04_p42'].iloc[0]).sum()+1}/{len(d)}")
print("\nTop-8 full:")
print(r2.sort_values("full_mean", ascending=False).head(8).to_string(index=False))
print("\nTop-8 f22:")
print(r2.sort_values("f22_mean", ascending=False).head(8).to_string(index=False))
print("DONE")
