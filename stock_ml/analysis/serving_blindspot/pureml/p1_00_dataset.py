# -*- coding: utf-8 -*-
"""P1-E1 buoc 0 — dataset cho tuyen CROSS-SECTIONAL RANKING (screen kill/go).

Universe = 61 ma champion (manifest bundle gb_x08-model). KHONG mo universe.
Features = dung bo champion entry `entry_lvup126_recov` (45 per-symbol, FeatureResolver
  tu catalog — cung code pipeline) + 2 breadth 488-univ (breadth_pct_above_ma50,
  breadth_adv_pct, ham _load_market_breadth cua pipeline).
Labels (moi hoan toan voi ho champion):
  fwd20 = close[t+20]/close[t] - 1 (per symbol);  fwd10 tuong tu (sensitivity)
  fwd20_dm / fwd10_dm = demeaned theo NGAY tren universe (return tuong doi vs EW mean)
  -> loai beta thi truong khoi label. label_end (date cua bar t+20) de purge split.
Out: p1_dataset.parquet
"""
import os
import sys

import duckdb
import json
import numpy as np
import pandas as pd

ROOT = r"f:\PROJECTS\train_ai_ml"
OUT = os.path.join(ROOT, "stock_ml", "analysis", "serving_blindspot", "pureml")
BUNDLE = os.path.join(ROOT, "bundles", "bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf")
os.chdir(ROOT)  # _load_market_breadth dung duong dan tuong doi market_data/market.duckdb
sys.path.insert(0, ROOT)

from stock_ml.src.features.resolver import FeatureResolver  # noqa: E402
from stock_ml.src.pipeline.experiment import _load_market_breadth  # noqa: E402

FEATURE_SET = "entry_lvup126_recov"   # bo champion entry (45 per-symbol)
HORIZONS = [20, 10]

# ---- universe champion (61 ma, tu manifest bundle model-side gb_x08) ----
manifest = json.load(open(os.path.join(BUNDLE, "manifest.json")))
universe = sorted(manifest["universe"])
print("universe:", len(universe), universe[:5], "...")

con = duckdb.connect(os.path.join(ROOT, "market_data", "market.duckdb"), read_only=True)
px = con.execute(
    "SELECT symbol, CAST(date AS VARCHAR) AS date, open, high, low, close, volume "
    "FROM ohlcv WHERE timeframe='1D' AND symbol IN (%s) AND date >= '2017-01-01' "
    "ORDER BY symbol, date" % ",".join("'%s'" % s for s in universe)).fetchdf()
con.close()
px["date"] = pd.to_datetime(px["date"])
px = px[(px[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)
px = px.drop_duplicates(["symbol", "date"], keep="last")
print("panel:", px.shape, px.date.min().date(), "->", px.date.max().date(),
      "| syms:", px.symbol.nunique())

# ---- features champion (resolver tu catalog, cache off de khong ghi store) ----
resolver = FeatureResolver.from_catalog()
need_raw = resolver.required_raw_inputs([FEATURE_SET])
print("raw inputs:", sorted(need_raw))
market_df = None
if "market_close" in need_raw:
    # EW index tu universe (cung cach pipeline dung cho $market_close)
    piv = px.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    mret = piv.pct_change().clip(-0.5, 0.5).mean(axis=1)
    lvl = (1.0 + mret.fillna(0.0)).cumprod()
    market_df = pd.DataFrame({"date": lvl.index, "close": lvl.values})

feat, cols_by_set, _hits = resolver.resolve(
    px, [FEATURE_SET], market_df=market_df, cache=False)
FEAT_COLS = cols_by_set[FEATURE_SET]
print("per-symbol features:", len(FEAT_COLS))

# ---- breadth 488-univ (2 cot champion entry4 dung) ----
for m, col in [("pct_above_ma50", "breadth_pct_above_ma50"), ("adv_pct", "breadth_adv_pct")]:
    br = _load_market_breadth(m, 50)
    feat[col] = feat["date"].map(br)
FEAT_COLS = FEAT_COLS + ["breadth_pct_above_ma50", "breadth_adv_pct"]
print("total features:", len(FEAT_COLS))

# ---- labels forward-return + demeaned-by-date ----
feat = feat.sort_values(["symbol", "date"]).reset_index(drop=True)
g = feat.groupby("symbol", sort=False)
for h in HORIZONS:
    feat[f"fwd{h}"] = g["close"].shift(-h) / feat["close"] - 1.0
feat["label_end20"] = g["date"].shift(-20)   # bar cuoi cung label dung -> purge
feat["label_end10"] = g["date"].shift(-10)

# demean theo ngay (chi tren nhung ma co label hop le ngay do)
for h in HORIZONS:
    feat[f"fwd{h}_dm"] = feat[f"fwd{h}"] - feat.groupby("date")[f"fwd{h}"].transform("mean")

# so ma hop le moi ngay (de loc ngay qua mong o dau lich su)
feat["n_universe"] = feat.groupby("date")["close"].transform("count")

meta = {"feature_cols": FEAT_COLS, "feature_set": FEATURE_SET,
        "universe_n": len(universe), "universe": universe}
with open(os.path.join(OUT, "p1_meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
feat.to_parquet(os.path.join(OUT, "p1_dataset.parquet"), index=False)
print("saved p1_dataset.parquet:", feat.shape)

# sanity: demean dung nghia (mean per date ~ 0)
chk = feat.dropna(subset=["fwd20_dm"]).groupby("date")["fwd20_dm"].mean().abs().max()
print("max |mean fwd20_dm per date| =", chk)
for y, gy in feat.groupby(feat.date.dt.year):
    print(y, "rows", len(gy), "syms", gy.symbol.nunique(),
          "median n_universe", int(gy.n_universe.median()))
