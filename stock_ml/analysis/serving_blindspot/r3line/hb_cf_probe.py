# -*- coding: utf-8 -*-
"""Probe: vì sao MWAIT/RBREAK ra NaN/0. Kiểm tra key-match sgd vs IDX + phân bố BELOW50."""
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
px = cx.execute("SELECT symbol,date,close FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"])
piv = px.pivot_table(index="date", columns="symbol", values="close").sort_index()
ewd = piv.pct_change().mean(axis=1)
idx_s = (1.0 + ewd.fillna(0.0)).cumprod()
ma50 = idx_s.rolling(50).mean()
IDX = {d.strftime("%Y-%m-%d"): float(v) for d, v in idx_s.items()}
BELOW50 = {d.strftime("%Y-%m-%d"): (bool(v < m) if pd.notna(m) else False) for (d, v), (_, m) in zip(idx_s.items(), ma50.items())}
print(f"idx_s: n={len(idx_s)} min={idx_s.min():.3f} max={idx_s.max():.3f} last={idx_s.iloc[-1]:.3f}", flush=True)
print(f"ewd: %neg={100*(ewd<0).mean():.1f}% mean={ewd.mean():+.5f} min={ewd.min():+.4f} max={ewd.max():+.4f}", flush=True)
print(f"BELOW50: %True={100*np.mean(list(BELOW50.values())):.1f}%", flush=True)

con = psycopg2.connect(**PG)
rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
cvtr = pd.read_sql("select symbol,entry_date,entry_signal_date from run_trades where run_id=%s and exit_date is not null limit 2000", con, params=(rid,)); con.close()
cvtr["sigd"] = pd.to_datetime(cvtr["entry_signal_date"]); cvtr["ed"] = cvtr["entry_date"].astype(str)
nsig_null = cvtr["entry_signal_date"].isna().sum()
sgd_in = sum(1 for r in cvtr.itertuples() if str(r.sigd.date()) in IDX)
ed_in = sum(1 for r in cvtr.itertuples() if r.ed in IDX)
print(f"trades={len(cvtr)} sig_null={nsig_null} sgd_in_IDX={sgd_in} ed_in_IDX={ed_in}", flush=True)
r0 = cvtr.iloc[0]
print(f"sample: ed={r0.ed!r} sgd={str(r0.sigd.date())!r} | ed_in={r0.ed in IDX} sgd_in={str(r0.sigd.date()) in IDX}", flush=True)
print(f"IDX sample keys: {list(IDX.keys())[:3]} ... {list(IDX.keys())[-3:]}", flush=True)
print("PROBE_DONE", flush=True)
