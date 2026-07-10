# -*- coding: utf-8 -*-
"""Signal-starvation step 3: tran thu hoach neu bat them song-chua-trade 2024-26H1.

Co che gia dinh: phat tin hieu tai bar buy-recon dau tien trong ±10 quanh chan song
(voi song da co signal nhung khong trade: bar +1 dau tien), dat limit pullback 4.5%
duoi close bar do, fill khi low <= limit trong 40 bar (KHONG doi co che fill).
Gia tri thu ve = (peak_close/fill_price - 1) x deflator_nam, voi deflator_nam =
Σu_thuc(cac song da trade) / Σ(fill-to-peak tiem nang cua chinh cac song do, proxy
gain_pct) — tuc gia dinh song moi bat duoc chuyen hoa u voi HIEU SUAT THUC TE cua
champion nam do (exit som, giveback, cost nhu thuong).
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = os.path.join(BASE, "exitmap")
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
BUNDLE = r"C:\Users\DUC CANH PC\Desktop\stock-serving\bundles\bundle_n2_2643_wavestruct_la05_lamp02_top150_2025-01-01_wf"
sys.path.insert(0, r"f:\PROJECTS\train_ai_ml\stock_ml")
from src.pipeline.experiment import _causal_zscore_by_symbol, _entry_gate_mask  # noqa: E402

PRE, POST = 10, 10
DEPTH = 0.045
WIN = 40
THR = {"score2": 0.9, "score3": 0.7, "score4": 0.7, "score5": 0.7}

ph = pd.read_parquet(os.path.join(BUNDLE, "prediction_history.parquet"))
ph["date"] = pd.to_datetime(ph["date"]).dt.date.astype(str)
ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)
universe = sorted(ph.symbol.unique())
con = sqlite3.connect(DB)
q = "select symbol, date, close, high, low from ohlcv where symbol in (%s) and date >= '2020-01-01' order by symbol, date" % (
    ",".join("?" * len(universe)))
px = pd.read_sql(q, con, params=universe)
con.close()
ph = ph.merge(px, on=["symbol", "date"], how="left")
for c in ["score", "score2", "score3", "score4", "score5"]:
    ph["z_" + c] = _causal_zscore_by_symbol(ph[c], ph["symbol"], 252, 60)
ph["gate"] = _entry_gate_mask(ph, "upleg_abovema20")
buy = (ph.z_score > -1.9) & ph.gate
for c, t in THR.items():
    buy |= ph["z_" + c] > t
ph["buy_recon"] = buy
sig = pd.read_csv(os.path.join(BASE, "signals.csv"), usecols=["symbol", "date", "signal"])
ph = ph.merge(sig, on=["symbol", "date"], how="left")
ph["signal"] = ph["signal"].fillna(0).astype(int)
arr_by_sym = {s: g.reset_index(drop=True) for s, g in ph.groupby("symbol")}
idx_by_sym = {s: pd.Series(np.arange(len(g)), index=g.date.to_numpy())
              for s, g in arr_by_sym.items()}

ep = pd.read_csv(os.path.join(OUT, "ss_catch_w1.csv"))
ep["year"] = ep.foot_date.str[:4]
ep = ep[(ep.year >= "2024")].reset_index(drop=True)

# deflator theo nam tu cac song DA trade
tra = ep[ep.has_trade]
defl = (tra.groupby("year").u_sum.sum() / (tra.groupby("year").gain_pct.sum() / 100))
print("deflator (u thuc / tiem nang foot-to-peak) theo nam:")
print(defl.round(3).to_string())

rows = []
for r in ep[~ep.has_trade].itertuples():
    g = arr_by_sym[r.symbol]
    im = idx_by_sym[r.symbol]
    if r.foot_date not in im.index:
        continue
    fi = int(im[r.foot_date])
    lo_i, hi_i = max(0, fi - PRE), min(len(g) - 1, fi + POST)
    w = g.iloc[lo_i:hi_i + 1]
    cand = w.index[(w.signal > 0) | w.buy_recon]
    if len(cand) == 0:
        rows.append(dict(year=r.year, status="no-source"))
        continue
    b = int(cand[0])
    limit = g.close.iloc[b] * (1 - DEPTH)
    fut = g.iloc[b + 1:min(len(g), b + 1 + WIN)]
    hit = fut.index[fut.low <= limit]
    if len(hit) == 0:
        rows.append(dict(year=r.year, status="no-fill"))
        continue
    fb = int(hit[0])
    # peak cua song trong frame prediction
    pk_date = r.peak_date
    if pk_date in im.index:
        pi = int(im[pk_date])
    else:
        pi = -1
    if pi <= fb:
        rows.append(dict(year=r.year, status="fill-after-peak"))
        continue
    pot = g.close.iloc[pi] / limit - 1
    rows.append(dict(year=r.year, status="filled", pot=pot))

sim = pd.DataFrame(rows)
print("\ntrang thai song-chua-trade 2024-26H1:")
print(sim.groupby(["year", "status"]).size().unstack(fill_value=0).to_string())
f = sim[sim.status == "filled"]
pot_y = f.groupby("year").pot.agg(["size", "sum", "mean"])
pot_y["u_ceiling_100"] = pot_y["sum"] * defl
pot_y["u_ceiling_50"] = pot_y.u_ceiling_100 * 0.5
pot_y["u_ceiling_25"] = pot_y.u_ceiling_100 * 0.25
print("\n=== TRAN thu hoach (deflator hieu suat thuc champion cung nam) ===")
print(pot_y.round(2).to_string())
print("\nde so sanh: u thuc champion (gb_x08) 2024=+2.35, 2025=+25.42, 2026H1=-0.55; noise seed ±2-3u")
print("DONE ss_03")
