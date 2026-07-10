"""Build the meta dataset for a LEARNED conv-combo prototype (offline, champion 2646 s42).

One row per champion trade. X = head z's at the signal bar (z of score, score2..score5,
exit_score; 252/60 rolling per-symbol, exactly the engine recipe), the champion's
reconstructed conv-combo strength + its price components, and causal context features.
y = trade net pnl_pct (+ win / big_win variants).

Signal-bar join: trade entry = pullback-limit fill after a buy-signal bar. Among buy
bars (signal==1) strictly before entry_date within 40 bars, pick the bar whose
reconstructed limit = close_sig * (1 - 0.045 * conv_mult) is PRICE-CONSISTENT with the
recorded entry_price (entry <= limit * (1 + slippage 0.001), i.e. limit fill or gap-down
open), preferring the latest such bar; if none is consistent, the closest. Measured
accuracy: 87.4% of trades exactly price-consistent (vs 37.6% for a naive last-bar join);
zE corr between the two joins is 0.93, so features are robust to residual ambiguity.

Champion 2646 conv config (from strategy_templates, Postgres):
  use_combo=True, combo_w=(0.25, 0.2, 0.3, 0.25) over (rsi_s, ext_s, eff_s, rpos),
  head_w=0.5 (sigmoid of escore_z), k=0.4, floor=0.5, vol_z=1.0 (lb=40),
  rsi_lo=50, ext_cap=0.10 (defaults). entry_pullback_pct=0.045, window=40.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SQ = HERE.parent
PARITY = SQ.parent / "wavestart" / "parity" / "bch_before"
SIGNALS_CSV = PARITY / "signals_n2_2643_wavestruct_la05_lamp02.csv"
TRADES_CSV = SQ / "st_champ2646_s42_trades.csv"
OUT = HERE / "meta_dataset.csv"

CW = (0.25, 0.20, 0.30, 0.25)   # (rsi, ext, eff, rpos)
HEAD_W = 0.5
CONV_K = 0.4
CONV_FLOOR = 0.5
RSI_LO = 50.0
EXT_CAP = 0.10
VOL_Z = 1.0
VOL_LB = 40
PB_PCT = 0.045
PB_WINDOW = 40


def _roll_z(s: pd.Series) -> pd.Series:
    mu = s.rolling(252, min_periods=60).mean()
    sd = s.rolling(252, min_periods=60).std()
    return (s - mu) / (sd + 1e-9)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def per_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    c = g["close"].astype(float)
    h = g["high"].astype(float)
    lo = g["low"].astype(float)

    # head z's (engine recipe: rolling 252 min 60)
    for col, zname in [("score", "zE"), ("score2", "z2"), ("score3", "z3"),
                       ("score4", "z4"), ("score5", "z5"), ("exit_score", "zX")]:
        g[zname] = _roll_z(g[col].astype(float))

    # engine conv price components
    d = c.diff()
    gain = d.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-9))
    ma20 = c.rolling(20, min_periods=20).mean()
    ext = c / ma20 - 1.0
    g["rsi_s"] = np.clip((rsi - RSI_LO) / (100.0 - RSI_LO + 1e-9), 0.0, 1.0)
    g["ext_s"] = np.clip(ext / (EXT_CAP + 1e-9), 0.0, 1.0)
    eff = (c - c.shift(10)).abs() / (d.abs().rolling(10, min_periods=10).sum() + 1e-9)
    g["eff_s"] = np.clip(eff, 0.0, 1.0)
    lo20 = lo.rolling(20, min_periods=10).min()
    hi20 = h.rolling(20, min_periods=10).max()
    g["rpos"] = np.clip((c - lo20) / (hi20 - lo20 + 1e-9), 0.0, 1.0)

    # champion strength / depth multiplier (incl. vol_z gate + warmup rule)
    price_combo = (CW[0] * g["rsi_s"] + CW[1] * g["ext_s"]
                   + CW[2] * g["eff_s"] + CW[3] * g["rpos"])
    price_combo = pd.Series(np.nan_to_num(price_combo.to_numpy(), nan=0.0), index=g.index)
    g["price_combo"] = price_combo
    strength = ((1.0 - HEAD_W) * price_combo
                + HEAD_W * _sigmoid(np.nan_to_num(g["zE"].to_numpy(), nan=0.0)))
    g["champ_strength"] = strength
    mult = np.clip(1.0 - CONV_K * strength, CONV_FLOOR, 1.0)
    mult[np.isnan(rsi.to_numpy()) | np.isnan(ext.to_numpy())] = 1.0
    rv = c.pct_change().rolling(20, min_periods=10).std()
    rvm = rv.rolling(VOL_LB, min_periods=10).mean()
    rvs = rv.rolling(VOL_LB, min_periods=10).std()
    volz = ((rv - rvm) / (rvs + 1e-9)).to_numpy()
    mult[volz > VOL_Z] = 1.0
    g["conv_mult"] = mult

    # causal context
    g["ret5"] = c.pct_change(5)
    g["ret20"] = c.pct_change(20)
    g["dist_ma20"] = ext
    tr = pd.concat([(h - lo), (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atr_pct = atr14 / c
    g["atr_ratio"] = atr_pct / (atr_pct.rolling(60, min_periods=30).mean() + 1e-9)
    g["snr20"] = ((c - c.shift(20)).abs()
                  / (d.abs().rolling(20, min_periods=20).sum() + 1e-9))
    g["above_ma20"] = (c > ma20).astype(float)
    g["bar_idx"] = np.arange(len(g))
    return g


def main() -> None:
    sig = pd.read_csv(SIGNALS_CSV, parse_dates=["date"])
    trades = pd.read_csv(TRADES_CSV, parse_dates=["entry_date", "exit_date"])
    print(f"signals: {len(sig)} rows / {sig.symbol.nunique()} symbols; trades: {len(trades)}")

    feats = (sig.groupby("symbol", group_keys=False)
             .apply(per_symbol_features)
             .reset_index(drop=True))

    # cross-sectional (universe) context per date — causal: uses same-date bar info only,
    # and the signal bar strictly precedes the trade entry.
    daily = feats.groupby("date").agg(breadth=("above_ma20", "mean"),
                                      uni_snr20=("snr20", "median"))
    feats = feats.merge(daily, on="date", how="left")

    keep = ["symbol", "date", "bar_idx", "signal", "score",
            "zE", "z2", "z3", "z4", "z5", "zX",
            "rsi_s", "ext_s", "eff_s", "rpos", "price_combo",
            "champ_strength", "conv_mult", "close",
            "ret5", "ret20", "dist_ma20", "atr_ratio", "snr20",
            "breadth", "uni_snr20"]
    feats = feats[keep]

    # join: last buy-signal bar strictly before entry_date, within PB_WINDOW bars
    buys = feats[feats["signal"] == 1].copy()
    rows = []
    unmatched = 0
    for t in trades.itertuples(index=False):
        f = feats[(feats.symbol == t.symbol) & (feats.date == t.entry_date)]
        entry_idx = int(f.bar_idx.iloc[0]) if len(f) else None
        cand = buys[(buys.symbol == t.symbol) & (buys.date < t.entry_date)]
        if entry_idx is not None:
            cand = cand[cand.bar_idx >= entry_idx - PB_WINDOW]
        if cand.empty:
            unmatched += 1
            continue
        limits = (cand.close * (1.0 - PB_PCT * cand.conv_mult)).to_numpy()
        ratios = t.entry_price / limits
        dist = np.where(ratios <= 1.0015, 0.0, ratios - 1.0015)
        j = int(np.argmin(dist + 1e-6 * (len(cand) - 1 - np.arange(len(cand)))))
        s = cand.iloc[j]
        lag = (entry_idx - int(s.bar_idx)) if entry_idx is not None else np.nan
        limit = float(limits[j])
        rows.append({
            "symbol": t.symbol, "entry_date": t.entry_date,
            "entry_year": t.entry_date.year, "signal_date": s.date,
            "bar_lag": lag, "recon_limit": limit,
            "entry_price": t.entry_price,
            "limit_ratio": t.entry_price / limit,
            "pnl_pct": t.pnl_pct, "holding_days": t.holding_days,
            "exit_reason": t.exit_reason,
            **{k: s[k] for k in ["zE", "z2", "z3", "z4", "z5", "zX",
                                 "rsi_s", "ext_s", "eff_s", "rpos", "price_combo",
                                 "champ_strength", "conv_mult",
                                 "ret5", "ret20", "dist_ma20", "atr_ratio",
                                 "snr20", "breadth", "uni_snr20"]},
        })
    meta = pd.DataFrame(rows)
    meta["win"] = (meta.pnl_pct > 0).astype(int)
    meta["big_win"] = (meta.pnl_pct >= 0.10).astype(int)

    print(f"matched {len(meta)}/{len(trades)} trades (unmatched {unmatched})")
    print("bar_lag:", meta.bar_lag.describe()[["mean", "50%", "max"]].round(2).to_dict())
    # join accuracy: entry at reconstructed limit (entry_price includes cost markup, so
    # ratio should sit in a tight band slightly above 1.0) or a gap-down open below it.
    r = meta.limit_ratio
    print("limit_ratio quantiles:", r.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(4).to_dict())
    tight = ((r > 0.995) & (r < 1.015)).mean()
    print(f"share of trades with entry within [-0.5%,+1.5%] of reconstructed limit: {tight:.1%}")
    print("nan counts:", meta.isna().sum()[lambda x: x > 0].to_dict())
    print("y: mean pnl {:.4f}, win {:.3f}, big_win {:.3f}".format(
        meta.pnl_pct.mean(), meta.win.mean(), meta.big_win.mean()))
    print(meta.groupby("entry_year").size().to_dict())

    meta.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(meta)} rows)")


if __name__ == "__main__":
    main()
