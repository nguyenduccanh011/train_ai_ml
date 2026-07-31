# -*- coding: utf-8 -*-
"""DECISIVE: entry_price có = close[ngày khớp] không? Quyết định same-day market filter là causal hay look-ahead.
 - nếu entry_price ≈ close[entry_date] => khớp TẠI ĐÓNG CỬA => biết return mkt ngày đó => filter causal => ĐĂNG KÝ được.
 - nếu entry_price < close (nằm giữa low..close) => khớp limit INTRADAY => same-day = look-ahead."""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.ERROR)
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd, numpy as np, duckdb
from scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
cx = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
px = cx.execute("SELECT symbol,date,open,high,low,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); OP = {}; HI = {}; LO = {}; CLO = {}; DIDX = {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date"); OP[s] = g["open"].values; HI[s] = g["high"].values; LO[s] = g["low"].values; CLO[s] = g["close"].values
    DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}

con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
cvtr = pd.read_sql("select symbol,entry_date,entry_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(rid,)); con.close()
cvtr["ed"] = cvtr["entry_date"].astype(str)
rows = []
for r in cvtr.itertuples():
    di = DIDX.get(r.symbol, {}); i = di.get(r.ed)
    if i is None:
        continue
    c, o, lo, hi = CLO[r.symbol][i], OP[r.symbol][i], LO[r.symbol][i], HI[r.symbol][i]
    rows.append(dict(ep=r.entry_price, vs_close=r.entry_price / c - 1, vs_open=r.entry_price / o - 1,
                     at_close=abs(r.entry_price / c - 1) < 1e-4, at_open=abs(r.entry_price / o - 1) < 1e-4,
                     in_lohi=(lo * 0.999 <= r.entry_price <= hi * 1.001), below_close=r.entry_price < c * 0.999))
D = pd.DataFrame(rows)
print(f"=== fill convention — {len(D)} lệnh seed42 ===", flush=True)
print(f"  entry_price / close[fill] - 1 : mean {100*D.vs_close.mean():+.3f}% | median {100*D.vs_close.median():+.3f}% | std {100*D.vs_close.std():.3f}%", flush=True)
print(f"  entry_price / open[fill]  - 1 : mean {100*D.vs_open.mean():+.3f}% | median {100*D.vs_open.median():+.3f}%", flush=True)
print(f"  % khớp ĐÚNG close (|Δ|<0.01%) : {100*D.at_close.mean():.1f}%", flush=True)
print(f"  % khớp ĐÚNG open  (|Δ|<0.01%) : {100*D.at_open.mean():.1f}%", flush=True)
print(f"  % entry nằm trong [low,high]  : {100*D.in_lohi.mean():.1f}%", flush=True)
print(f"  % entry < close (limit dưới)  : {100*D.below_close.mean():.1f}%", flush=True)
print("FILLCONV_DONE", flush=True)
