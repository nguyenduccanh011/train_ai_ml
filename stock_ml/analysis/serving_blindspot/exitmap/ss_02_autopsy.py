# -*- coding: utf-8 -*-
"""Signal-starvation step 2: autopsy song-bi-mu 2024-26H1 (W1, khong co buy-signal ±10 bar).

Tai lap chain buy cua champion 2643 tu prediction_history.parquet (score moi bar):
  buy = (zE > -1.9 & gate upleg_abovema20) | z2>0.9 | z3>0.7 | z4>0.7 | z5>0.7
  z = causal rolling 252/60 per symbol (dung _causal_zscore_by_symbol cua pipeline).
Sanity: so khop voi signals.csv (+1 = buy & ~sell).
Phan loai song mu (uu tien): SELL-VETO > NEAR-MISS (max ensemble margin >= -0.25)
  > GATE-BLOCKED (zE qua nguong nhung gate dong suot cua so) > TRUE-SILENCE.
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
THR = {"score2": 0.9, "score3": 0.7, "score4": 0.7, "score5": 0.7}
Z_MAIN = -1.9
NEAR = 0.25  # near-miss band duoi threshold

ph = pd.read_parquet(os.path.join(BUNDLE, "prediction_history.parquet"))
ph["date"] = pd.to_datetime(ph["date"]).dt.date.astype(str)
ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)

# join close (can cho gate mask)
universe = sorted(ph.symbol.unique())
con = sqlite3.connect(DB)
q = "select symbol, date, close, high, low from ohlcv where symbol in (%s) and date >= '2020-01-01' order by symbol, date" % (
    ",".join("?" * len(universe)))
px = pd.read_sql(q, con, params=universe)
con.close()
ph = ph.merge(px, on=["symbol", "date"], how="left")
assert ph.close.isna().sum() == 0, ph.close.isna().sum()

# causal z 252/60
for c in ["score", "score2", "score3", "score4", "score5"]:
    ph["z_" + c] = _causal_zscore_by_symbol(ph[c], ph["symbol"], 252, 60)

gate = _entry_gate_mask(ph, "upleg_abovema20")
ph["gate"] = gate
ph["buy_main"] = (ph.z_score > Z_MAIN) & gate
ens_margin = np.full(len(ph), -np.inf)
for c, thr in THR.items():
    ens_margin = np.maximum(ens_margin, ph["z_" + c].to_numpy() - thr)
ph["ens_margin"] = ens_margin
ph["buy_recon"] = ph.buy_main | (ens_margin > 0)

# ---- sanity vs signals.csv ----
sig = pd.read_csv(os.path.join(BASE, "signals.csv"), usecols=["symbol", "date", "signal"])
ph = ph.merge(sig, on=["symbol", "date"], how="left")
ph["signal"] = ph["signal"].fillna(0).astype(int)
plus = ph.signal > 0
print("sanity: n(+1 signals.csv)=", plus.sum(),
      "| trong do buy_recon=True: %.2f%%" % (100 * ph.buy_recon[plus].mean()))
recon_not_sell = ph.buy_recon & (ph.signal >= 0)
print("sanity: n(buy_recon & not-sell)=", recon_not_sell.sum(),
      "| trong do signals.csv=+1: %.2f%%" % (100 * plus[recon_not_sell].mean()))

# ---- autopsy blind waves ----
ep = pd.read_csv(os.path.join(OUT, "ss_catch_w1.csv"))
ep["year"] = ep.foot_date.str[:4]

# bar index trong frame prediction (per symbol) — map bang date
idx_by_sym = {s: pd.Series(np.arange(len(g)), index=g.date.to_numpy())
              for s, g in ph.groupby("symbol")}
arr_by_sym = {s: g.reset_index(drop=True) for s, g in ph.groupby("symbol")}

rows = []
for r in ep[ep.year >= "2020"].itertuples():
    g = arr_by_sym[r.symbol]
    im = idx_by_sym[r.symbol]
    if r.foot_date not in im.index:
        continue
    fi = int(im[r.foot_date])
    lo_i, hi_i = max(0, fi - PRE), min(len(g) - 1, fi + POST)
    w = g.iloc[lo_i:hi_i + 1]
    m_ens = w.ens_margin.max()
    any_gate = bool(w.gate.any())
    any_buy = bool(w.buy_recon.any())
    any_plus = bool((w.signal > 0).any())
    zE_max = w.z_score.max()
    # gate mo lan dau sau foot (trong +40 bar)
    fut = g.iloc[fi:min(len(g), fi + 41)]
    go = np.nonzero(fut.gate.to_numpy())[0]
    gate_open_lag = int(go[0]) if len(go) else -1
    if any_plus:
        cls = "CAUGHT"
    elif any_buy:
        cls = "SELL-VETO"
    elif m_ens >= -NEAR:
        cls = "NEAR-MISS"
    elif zE_max > Z_MAIN and not any_gate:
        cls = "GATE-BLOCKED"
    else:
        cls = "TRUE-SILENCE"
    rows.append(dict(symbol=r.symbol, foot_date=r.foot_date, year=r.year,
                     gain_pct=r.gain_pct, bars_to_15=r.bars_to_15,
                     below_ma20=r.below_ma20, drawdown_pre=r.drawdown_pre,
                     adv20_bil=r.adv20_bil, has_trade=r.has_trade,
                     cls=cls, ens_margin=round(float(m_ens), 3),
                     zE_max=round(float(zE_max), 3), any_gate=any_gate,
                     gate_open_lag=gate_open_lag))
au = pd.DataFrame(rows)
au.to_csv(os.path.join(OUT, "ss_autopsy_w1.csv"), index=False)

print("\n=== phan loai song theo nam (W1) ===")
tab = au.pivot_table(index="year", columns="cls", values="symbol", aggfunc="size").fillna(0).astype(int)
tab["n"] = tab.sum(1)
for c in [c for c in ["SELL-VETO", "NEAR-MISS", "GATE-BLOCKED", "TRUE-SILENCE"] if c in tab]:
    tab[c + "_%"] = (100 * tab[c] / tab.n).round(1)
print(tab.to_string())

bl = au[(au.year >= "2024") & (au.cls != "CAUGHT")]
ca = au[(au.year >= "2024") & (au.cls == "CAUGHT")]
print("\n=== blind 2024-26H1: n=%d | phan ra: ===" % len(bl))
print(bl.cls.value_counts().to_string())
print("\nens_margin cua blind (deciles):")
print(bl.ens_margin.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
print("\n=== trait: blind vs caught (2024-26H1, median) ===")
for c in ["gain_pct", "bars_to_15", "drawdown_pre", "adv20_bil"]:
    print(f"{c:14s} blind={bl[c].median():8.2f}  caught={ca[c].median():8.2f}")
print("below_ma20@foot: blind=%.0f%% caught=%.0f%%" % (100 * bl.below_ma20.mean(), 100 * ca.below_ma20.mean()))
print("gate_open_lag (median bars den khi gate mo): blind=%.1f caught=%.1f" %
      (bl.gate_open_lag.replace(-1, np.nan).median(), ca.gate_open_lag.replace(-1, np.nan).median()))
print("gate khong bao gio mo trong +40 bar: blind=%.0f%% caught=%.0f%%" %
      (100 * (bl.gate_open_lag < 0).mean(), 100 * (ca.gate_open_lag < 0).mean()))
tr61 = set(pd.read_csv(os.path.join(OUT, "gbx08_trades.csv")).symbol.unique())
print("ma chua tung duoc trade (ngoai 61): blind=%.0f%% caught=%.0f%%" %
      (100 * (~bl.symbol.isin(tr61)).mean(), 100 * (~ca.symbol.isin(tr61)).mean()))
print("\nDONE ss_02")
