# -*- coding: utf-8 -*-
"""P1-E1 buoc 1 — walk-forward theo nam, 3 model:
  (a) lgbm_reg  : LGBMRegressor tren fwd20_dm (label demeaned-by-date)
  (b) lgbm_rank : LGBMRanker lambdarank, group = ngay, relevance = quintile fwd20_dm per-date
  (c) momo      : baseline khong-ML = ret_20d (rank momentum)
Chuan: train <= y-1 voi PURGE label-window (label_end20 < test_start — khong co bar
label nao cham sang nam test); test = nam y (2021..2026H1). Seeds 42/7/99.
Sensitivity: lgbm_reg tren fwd10_dm (seed 42).
Out: p1_preds.parquet (symbol,date,fwd20,fwd20_dm,fwd10_dm + cot pred_*)
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml"
SEEDS = [42, 7, 99]
FOLDS = [(2021, "2021-01-01", "2022-01-01"),
         (2022, "2022-01-01", "2023-01-01"),
         (2023, "2023-01-01", "2024-01-01"),
         (2024, "2024-01-01", "2025-01-01"),
         (2025, "2025-01-01", "2026-01-01"),
         (2026, "2026-01-01", "2027-01-01")]

PARAMS = dict(num_leaves=15, learning_rate=0.03, n_estimators=600,
              min_child_samples=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=5,
              lambda_l1=0.1, lambda_l2=1.0, n_jobs=-1, verbosity=-1)

df = pd.read_parquet(os.path.join(OUT, "p1_dataset.parquet"))
meta = json.load(open(os.path.join(OUT, "p1_meta.json")))
FEAT = meta["feature_cols"]
df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
print("dataset:", df.shape, "| features:", len(FEAT))

out_cols = ["symbol", "date", "close", "fwd20", "fwd10", "fwd20_dm", "fwd10_dm", "ret_20d"]
preds = []

def relevance_bins(y, dates, k=5):
    """Quintile per-date -> relevance 0..k-1 (lambdarank labels)."""
    s = pd.Series(y)
    r = s.groupby(dates.values).rank(pct=True, method="average")
    return np.clip((r * k).astype(int).clip(upper=k - 1), 0, k - 1).to_numpy()

for year, ts, te in FOLDS:
    ts_d, te_d = pd.Timestamp(ts), pd.Timestamp(te)
    test = df[(df.date >= ts_d) & (df.date < te_d)].copy()
    if test.empty:
        continue
    o = test[out_cols].copy()
    o["year"] = year

    for h, label in [(20, "fwd20_dm"), (10, "fwd10_dm")]:
        tr = df[(df[f"label_end{h}"].notna()) & (df[f"label_end{h}"] < ts_d)
                & (df[label].notna()) & (df.n_universe >= 30)]
        Xtr, ytr = tr[FEAT], tr[label].to_numpy()
        if h == 20:
            print(f"[{year}] train rows={len(tr)} ({tr.date.min().date()}..{tr.date.max().date()}) "
                  f"test rows={len(test)}", flush=True)
        # (a) regression tren label demeaned
        seeds = SEEDS if h == 20 else [42]
        for sd in seeds:
            m = lgb.LGBMRegressor(random_state=sd, **PARAMS)
            m.fit(Xtr, ytr)
            o[f"pred_reg{'' if h == 20 else '_h10'}_s{sd}"] = m.predict(test[FEAT])
        # (b) lambdarank (chi cho h=20)
        if h == 20:
            trs = tr.sort_values(["date", "symbol"])
            rel = relevance_bins(trs[label].to_numpy(), trs["date"])
            grp = trs.groupby("date", sort=True).size().to_numpy()
            for sd in SEEDS:
                mr = lgb.LGBMRanker(objective="lambdarank", random_state=sd,
                                    label_gain=list(range(32)), **PARAMS)
                mr.fit(trs[FEAT], rel, group=grp)
                o[f"pred_rank_s{sd}"] = mr.predict(test[FEAT])
    preds.append(o)

pred = pd.concat(preds, ignore_index=True)
# (c) baseline momentum-rank khong-ML
pred["pred_momo"] = pred["ret_20d"]
pred.to_parquet(os.path.join(OUT, "p1_preds.parquet"), index=False)
print("saved p1_preds.parquet:", pred.shape)
print(pred.groupby("year").size())
print("P1_01_DONE")
