# -*- coding: utf-8 -*-
"""Step 1: regenerate full-universe signals + trades for the active bundle (local store only).

Runs the REAL engine twice: once via the installed wheel (baseline, = serving derive_trades)
and once via the instrumented copy (engine_instr.py) that records skipped/missed/filled
pullback events and the per-bar effective conviction-scaled pullback depth.
Asserts trade-level parity between the two, then writes:
  signals.csv     — all non-zero signal rows (+score columns)
  trades_raw.csv  — trades_to_dataframe output (raw engine ledger)
  events.csv      — instrumented entry-path events
  depths.parquet  — per (symbol, date) effective pullback depth
"""
import os
import sys

SERVING = r"C:\Users\DUC CANH PC\Desktop\stock-serving"
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
BUNDLE = "bundles/bundle_n2_2643_wavestruct_la05_lamp02_top150_2025-01-01_wf"

os.chdir(SERVING)
sys.path.insert(0, SERVING)
sys.path.insert(0, OUT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from serving.engine import SignalEngine  # noqa: E402
from serving.trades import _build_engine_config  # noqa: E402

eng = SignalEngine(BUNDLE, warm=False, write_cache=False, prewarm_days=0)
signals, ohlcv = eng.snapshot()
print("signals shape:", signals.shape, "cols:", list(signals.columns))
print("ohlcv shape:", ohlcv.shape)

# --- signals.csv (non-zero rows only, keep score columns) ---
score_cols = [c for c in ["score", "score2", "score3", "score4", "score5", "score6",
                          "exit_score", "entry_csr", "swing_top_prob"] if c in signals.columns]
sig_nz = signals[signals["signal"] != 0][["symbol", "date", "signal"] + score_cols].copy()
sig_nz["date"] = pd.to_datetime(sig_nz["date"]).dt.date.astype(str)
sig_nz.to_csv(os.path.join(OUT, "signals.csv"), index=False, encoding="utf-8")
print("signals.csv rows:", len(sig_nz), "buy:", int((sig_nz['signal'] > 0).sum()),
      "sell:", int((sig_nz['signal'] < 0).sum()))

# --- engine config (identical to serving derive_trades) ---
engine_cfg = getattr(eng, "engine_cfg", None) or eng.bundle.config["engine"]
ec = _build_engine_config(engine_cfg)

# --- baseline run (installed wheel) ---
from stock_ml.src.backtest.engine import run_backtest as rb0, trades_to_dataframe  # noqa: E402
t0 = trades_to_dataframe(rb0(signals, ohlcv, ec))
print("baseline trades:", len(t0))

# --- instrumented run ---
import engine_instr  # noqa: E402
engine_instr.instr_reset()
t1 = engine_instr.trades_to_dataframe(engine_instr.run_backtest(signals, ohlcv, ec))
print("instrumented trades:", len(t1))

# --- parity check ---
a = t0.sort_values(["symbol", "entry_date"]).reset_index(drop=True)
b = t1.sort_values(["symbol", "entry_date"]).reset_index(drop=True)
assert len(a) == len(b), f"trade count mismatch {len(a)} vs {len(b)}"
for col in ["symbol", "entry_date", "exit_date", "exit_reason"]:
    assert (a[col].astype(str) == b[col].astype(str)).all(), f"mismatch col {col}"
for col in ["entry_price", "exit_price", "pnl_pct"]:
    assert np.allclose(a[col].astype(float), b[col].astype(float)), f"mismatch col {col}"
print("PARITY OK: instrumented engine reproduces the wheel engine exactly")

t1s = t1.copy()
for c in ["entry_date", "exit_date", "entry_signal_date"]:
    t1s[c] = pd.to_datetime(t1s[c]).dt.date.astype(str)
t1s.to_csv(os.path.join(OUT, "trades_raw.csv"), index=False, encoding="utf-8")
print("trades_raw.csv rows:", len(t1s))
print("exit_reason counts:\n", t1s["exit_reason"].value_counts())

# --- events.csv ---
rows = []
for sym, kind, i, limit, aux in engine_instr.INSTR_EVENTS:
    d = engine_instr.INSTR_DATES.get(sym)
    date = str(pd.Timestamp(d[i]).date()) if d is not None else None
    aux_date = str(pd.Timestamp(d[aux]).date()) if (d is not None and aux >= 0) else None
    rows.append((sym, kind, i, date, limit, aux, aux_date))
ev = pd.DataFrame(rows, columns=["symbol", "kind", "bar_idx", "date", "limit_price",
                                 "aux_idx", "aux_date"])
ev.to_csv(os.path.join(OUT, "events.csv"), index=False, encoding="utf-8")
print("events.csv rows:", len(ev))
print(ev["kind"].value_counts())

# --- depths.parquet ---
dep_frames = []
for sym, dep in engine_instr.INSTR_DEPTH.items():
    dts = engine_instr.INSTR_DATES[sym]
    dep_frames.append(pd.DataFrame({"symbol": sym,
                                    "date": pd.to_datetime(dts).astype(str).str[:10],
                                    "eff_depth": dep}))
depth_df = pd.concat(dep_frames, ignore_index=True)
depth_df.to_parquet(os.path.join(OUT, "depths.parquet"), index=False)
print("depths.parquet rows:", len(depth_df))

# --- pending orders reference (for daily_activity reconciliation) ---
try:
    pend = eng.pending_orders()
    pend.to_csv(os.path.join(OUT, "pending_orders_ref.csv"), index=False, encoding="utf-8")
    print("pending_orders (serving view):", len(pend))
except Exception as e:  # noqa: BLE001
    print("pending_orders failed:", e)

print("DONE step 1")
