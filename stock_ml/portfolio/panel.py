"""Market panel: full-market cs5_ma50 conviction (cross-sectional rank) + ret7 + price arrays.

Pure frame-in functions — no DB access here; the PortfolioContext fetches frames
(keeps duckdb/sqlite out of the wheel dependency surface).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml.portfolio.constants import CS4


def build_market_panel(px: pd.DataFrame):
    """px: columns symbol,date,low,close,high (full market, from market_start).
    Returns CLO/LO/DIDX/INV price arrays + CSm (conviction) / R5 (ret7) maps."""
    px = px.copy()
    px["date"] = pd.to_datetime(px["date"])
    CLO, LO, DIDX, INV, parts = {}, {}, {}, {}, []
    for s, g in px.groupby("symbol"):
        g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
        CLO[s] = c.to_numpy(dtype=float); LO[s] = l.to_numpy(dtype=float)
        DIDX[s] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(g["date"])}
        INV[s] = {i: d.strftime("%Y-%m-%d") for i, d in enumerate(g["date"])}
        dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
        g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
        g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
        tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1
        g["ret5"] = c / c.shift(7) - 1  # ret7 window
        parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50", "ret5"]])
    P = pd.concat(parts, ignore_index=True)
    for col in CS4 + ["atrpct", "dist_ma50"]:
        P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
    P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
    CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}
    R5 = {(r.symbol, str(r.date.date())): (r.ret5 if pd.notna(r.ret5) else np.nan) for r in P.itertuples()}
    return CLO, LO, DIDX, INV, CSm, R5


def build_price_panel(px: pd.DataFrame):
    """px: columns symbol,date,close (NAV-mark store slice). -> (sym_close, sym_idx, calendar)."""
    px = px.copy()
    px["date"] = px["date"].astype(str).str[:10]
    sym_close, sym_idx = {}, {}
    for s, g in px.groupby("symbol"):
        g = g.sort_values("date")
        sym_close[s] = g["close"].to_numpy(dtype=float)
        sym_idx[s] = {d: i for i, d in enumerate(g["date"].tolist())}
    calendar = sorted(set(px["date"]))
    return sym_close, sym_idx, calendar
