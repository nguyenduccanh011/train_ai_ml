# -*- coding: utf-8 -*-
"""SCORE AUDIT buoc 0 — kiem ke + dump chuoi score tho cua gb_x08 (template 2783).

Nguon: bundles/bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf/prediction_history.parquet
  - model-side TRUNG voi 2783 (cung entry/exit target, 4 ensemble, seed 42, WF 2y/1y gap85);
    2783 chi khac cac engine knob exit_snr_* (khong cham score).
Chuoi: score (entry triple_barrier h30 pt15 sl8), exit_score (velocity_exit h20/u8/vol40),
  score2 (reversal_entry h10), score3 (continuation_entry h10), score4 (mfe h20),
  score5 (fwd_return_penalized h20). RAW + z-norm causal 252/60 per-symbol (dung ham pipeline).
Gate/force state (tai lap tu experiment.py, cung token config 2783):
  gate_entry (upleg_abovema20), gate_exit (cons2_w20), f_dl12 (downleg12),
  f_bma20p2 (belowma20p2), nonbull (ma35 p2), f_dl6 (downleg6), lowbreadth (pctma50<0.25),
  breadth (gia tri pct_above_ma50 488-univ).
Quyet dinh tai lap: buy_main = zE>-1.9 & gate_entry; buy2..5 = z>0.9/0.7/0.7/0.7;
  sell_ml = zX>2.0 & gate_exit; sell_force = f_dl12 | (f_bma20p2 & nonbull) | (f_dl6 & lowbreadth).
Out: sa_scores.parquet
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = r"f:\PROJECTS\train_ai_ml"
BUNDLE = os.path.join(ROOT, "bundles", "bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf")
OUT = os.path.join(ROOT, "stock_ml", "analysis", "serving_blindspot", "scoreaudit")
os.chdir(ROOT)  # _load_market_breadth dung duong dan tuong doi market_data/market.duckdb
sys.path.insert(0, os.path.join(ROOT, "stock_ml"))

from src.pipeline.experiment import (  # noqa: E402
    _causal_zscore_by_symbol, _entry_gate_mask, _exit_gate_mask,
    _exit_force_mask, _market_nonbull_mask, _market_breadth_mask,
    _load_market_breadth,
)

SCORES = ["score", "score2", "score3", "score4", "score5", "exit_score"]

ph = pd.read_parquet(os.path.join(BUNDLE, "prediction_history.parquet"))
ph["date"] = pd.to_datetime(ph["date"]).dt.strftime("%Y-%m-%d")
ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)
print("ph:", ph.shape, ph.date.min(), "->", ph.date.max(), "| syms:", ph.symbol.nunique())

universe = sorted(ph.symbol.unique())
con = duckdb.connect(os.path.join(ROOT, "market_data", "market.duckdb"), read_only=True)
px = con.execute(
    "SELECT symbol, CAST(date AS VARCHAR) AS date, open, high, low, close, volume "
    "FROM ohlcv WHERE timeframe='1D' AND symbol IN (%s) AND date >= '2019-01-01' "
    "ORDER BY symbol, date" % ",".join("'%s'" % s for s in universe)).fetchdf()
con.close()
ph = ph.merge(px, on=["symbol", "date"], how="left")
assert ph.close.isna().sum() == 0, "thieu gia: %d" % ph.close.isna().sum()

# z-norm causal 252/60 (chinh xac ham pipeline dung o recombine)
for c in SCORES:
    ph["z_" + c] = _causal_zscore_by_symbol(ph[c], ph["symbol"], 252, 60)

# gates + force states
ph["gate_entry"] = _entry_gate_mask(ph, "upleg_abovema20")
ph["gate_exit"] = _exit_gate_mask(ph, "cons2_w20")
ph["f_dl12"] = _exit_force_mask(ph, "downleg12")
ph["f_bma20p2"] = _exit_force_mask(ph, "belowma20p2")
ph["f_dl6"] = _exit_force_mask(ph, "downleg6")
ph["nonbull"] = _market_nonbull_mask(ph, 35, 2)
ph["lowbreadth"] = _market_breadth_mask(ph, "pct_above_ma50", 0.25, 50, "level", 60)
br = _load_market_breadth("pct_above_ma50", 50)
ph["breadth"] = ph["date"].map({d.strftime("%Y-%m-%d"): v for d, v in br.items()})
# downleg z cua thi truong (exit_market_drop: z cua 5d-return EW index, lookback 60)
piv = ph.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
mret = piv.pct_change().clip(-0.5, 0.5).mean(axis=1)
lvl = (1.0 + mret.fillna(0.0)).cumprod()
r5 = lvl.pct_change(5)
mz = (r5 - r5.rolling(60, min_periods=60).mean()) / r5.rolling(60, min_periods=60).std()
ph["mkt_drop_z"] = ph["date"].map(mz.to_dict())
# bma20 (khong persist) de do corr trang thai
g = ph.groupby("symbol", sort=False)
ma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
ph["bma20"] = (ph["close"] < ma20).where(ma20.notna(), False)

# quyet dinh tai lap (dung threshold 2783: entry -1.9, ens 0.9/.7/.7/.7, exit z 2.0)
ph["buy_main"] = (ph.z_score > -1.9) & ph.gate_entry
ph["buy2"] = ph.z_score2 > 0.9
ph["buy3"] = ph.z_score3 > 0.7
ph["buy4"] = ph.z_score4 > 0.7
ph["buy5"] = ph.z_score5 > 0.7
ph["buy_union"] = ph.buy_main | ph.buy2 | ph.buy3 | ph.buy4 | ph.buy5
ph["sell_ml_raw"] = ph.z_exit_score > 2.0
ph["sell_ml"] = ph.sell_ml_raw & ph.gate_exit
ph["sell_force"] = ph.f_dl12 | (ph.f_bma20p2 & ph.nonbull) | (ph.f_dl6 & ph.lowbreadth)

ph.to_parquet(os.path.join(OUT, "sa_scores.parquet"), index=False)
print("saved sa_scores.parquet:", ph.shape)
print("\nrates (pooled): buy_main %.3f buy_union %.3f sell_ml %.3f sell_force %.3f" % (
    ph.buy_main.mean(), ph.buy_union.mean(), ph.sell_ml.mean(), ph.sell_force.mean()))
for y, gy in ph.groupby(ph.date.str[:4]):
    print(" %s buy_union %.4f (main %.4f e2 %.4f e3 %.4f e4 %.4f e5 %.4f) sell_ml %.4f raw %.4f force %.4f" % (
        y, gy.buy_union.mean(), gy.buy_main.mean(), gy.buy2.mean(), gy.buy3.mean(),
        gy.buy4.mean(), gy.buy5.mean(), gy.sell_ml.mean(), gy.sell_ml_raw.mean(), gy.sell_force.mean()))
