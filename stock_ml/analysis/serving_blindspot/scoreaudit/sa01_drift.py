# -*- coding: utf-8 -*-
"""SCORE AUDIT buoc 1 — drift & saturation theo nam, RAW vs Z.

Voi moi chuoi (6 score):
  - RAW theo nam: mean/std/skew/q05/q50/q95; %sat_lo/%sat_hi = % ngay nam ngoai
    [q01, q99] pooled 2020-2023 (vung "bao hoa" so voi phan phoi ma threshold duoc tune).
  - Z theo nam: mean/std/skew/q95/q99; %|z|>2.5; % ngay vuot threshold van hanh.
  - Do on dinh hoa: dispersion cua mean theo nam (std cua yearly means) RAW-chuan-hoa
    (chia pooled std) vs Z. Neu Z khong ~ nho hon RAW -> z-norm khong on dinh hoa duoc.
Out: sa01_raw_stats.csv, sa01_z_stats.csv + bang in.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\scoreaudit"
SCORES = ["score", "score2", "score3", "score4", "score5", "exit_score"]
THR = {"score": -1.9, "score2": 0.9, "score3": 0.7, "score4": 0.7, "score5": 0.7,
       "exit_score": 2.0}

ph = pd.read_parquet(os.path.join(OUT, "sa_scores.parquet"))
ph["year"] = ph.date.str[:4]
ph.loc[ph.date >= "2026-01-01", "year"] = "2026H1"

ref = ph[(ph.date >= "2020-01-01") & (ph.date < "2024-01-01")]

rows_raw, rows_z = [], []
for c in SCORES:
    q01, q99 = ref[c].quantile([0.01, 0.99])
    for y, g in ph.groupby("year"):
        x = g[c].dropna()
        rows_raw.append(dict(
            score=c, year=y, n=len(x), mean=x.mean(), std=x.std(),
            skew=sps.skew(x), q05=x.quantile(0.05), q50=x.quantile(0.5),
            q95=x.quantile(0.95),
            sat_lo=(x < q01).mean() * 100, sat_hi=(x > q99).mean() * 100))
        z = g["z_" + c].dropna()
        thr = THR[c]
        cross = (z > thr).mean() * 100
        rows_z.append(dict(
            score=c, year=y, n=len(z), mean=z.mean(), std=z.std(),
            skew=sps.skew(z), q05=z.quantile(0.05), q95=z.quantile(0.95),
            q99=z.quantile(0.99), pct_abs_gt2p5=(z.abs() > 2.5).mean() * 100,
            pct_cross_thr=cross))

raw = pd.DataFrame(rows_raw)
zz = pd.DataFrame(rows_z)
raw.to_csv(os.path.join(OUT, "sa01_raw_stats.csv"), index=False)
zz.to_csv(os.path.join(OUT, "sa01_z_stats.csv"), index=False)

pd.set_option("display.width", 250)
for c in SCORES:
    print("\n==== %s RAW ====" % c)
    print(raw[raw.score == c].drop(columns="score").round(4).to_string(index=False))
    print("---- %s Z (threshold %.2f) ----" % (c, THR[c]))
    print(zz[zz.score == c].drop(columns="score").round(4).to_string(index=False))

# do on dinh hoa drift: dispersion yearly-mean (don vi pooled-std) RAW vs Z
print("\n==== DRIFT STABILIZATION (std cua yearly means / pooled std) ====")
for c in SCORES:
    r = raw[raw.score == c]
    z = zz[zz.score == c]
    disp_raw = r["mean"].std() / ph[c].std()
    disp_z = z["mean"].std() / ph["z_" + c].std()
    # std stability: range cua yearly std / pooled std
    sr = (r["std"].max() - r["std"].min()) / ph[c].std()
    sz = (z["std"].max() - z["std"].min()) / ph["z_" + c].std()
    print(" %-11s mean-disp RAW %.3f -> Z %.3f | std-range RAW %.3f -> Z %.3f" % (
        c, disp_raw, disp_z, sr, sz))
