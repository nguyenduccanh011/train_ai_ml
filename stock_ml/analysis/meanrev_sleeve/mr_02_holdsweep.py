"""ROOT question: is there ANY daily-OHLCV long edge in the dead years 2024/2026, or is it a
genuine no-edge regime? Sweep HOLD length + reversal flavor on the ML mean-rev sleeve, measure
per-year (focus 2024/2026). If a short-hold / deep-oversold reversal turns 2024/2026 POSITIVE,
that's a live decorrelated source; if every flavor fails there, dead-year is a structural no-edge tax.
Reuses mr_01 load; recomputes label/setup per config. Standalone diagnostic (100-univ, WF 2022-26).
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stock_ml.analysis.meanrev_sleeve.mr_01_model import load, FEATCOLS, feats as _feats0

COST = 0.004
TOPQ = 0.80


def build_cfg(oh, hold, setup_kind):
    """Recompute per-symbol feats + label/pnl/setup for a given HOLD and setup flavor."""
    parts = []
    for _, g in oh.groupby("symbol"):
        g = g.sort_values("date")
        c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"]
        ma20, ma50, ma100 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(100).mean()
        std20 = c.rolling(20).std()
        d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + up / (dn + 1e-9))
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        ret1 = c.pct_change(); ret5 = c / c.shift(5) - 1; ret10 = c / c.shift(10) - 1
        downdays = (ret1 < 0).astype(int).groupby((ret1 >= 0).cumsum()).cumsum()
        dd20 = c / c.rolling(20).max() - 1; dd60 = c / c.rolling(60).max() - 1
        lowwick = (np.minimum(o, c) - l) / (h - l + 1e-9)
        rng = (h - l) / c
        volz = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
        df = pd.DataFrame({
            "rsi": rsi, "rsi_chg": rsi.diff(3),
            "dist_ma20": c / ma20 - 1, "dist_ma50": c / ma50 - 1, "dist_ma100": c / ma100 - 1,
            "bb_pct": (c - ma20) / (2 * std20 + 1e-9), "bb_width": (4 * std20) / ma20,
            "ret1": ret1, "ret5": ret5, "ret10": ret10, "downdays": downdays, "dd20": dd20, "dd60": dd60,
            "atr_ratio": atr / c, "lowwick": lowwick, "rng": rng, "volz": volz,
            "ma100_slope": ma100 / ma100.shift(20) - 1,
        })
        entry = c.shift(-1)
        fwd = c.shift(-1 - hold) / entry - 1
        df["pnl"] = fwd - COST
        fmax = pd.concat([h.shift(-k) / entry - 1 for k in range(1, hold + 1)], axis=1).max(axis=1)
        fmin = pd.concat([l.shift(-k) / entry - 1 for k in range(1, hold + 1)], axis=1).min(axis=1)
        # bounce label scaled to hold (shorter hold -> smaller target)
        tgt = 0.03 if hold <= 4 else 0.04
        df["label"] = ((fmax >= tgt) & (fmin > -0.06)).astype(int)
        if setup_kind == "dip_uptrend":       # mr_01 base
            df["setup"] = ((ret5 < -0.05) | (rsi < 45)) & (c > ma100) & (ma100 > ma100.shift(20))
        elif setup_kind == "deep_oversold":   # RSI<30 snapback, still above long MA
            df["setup"] = (rsi < 30) & (c > ma100)
        elif setup_kind == "oversold_any":    # oversold regardless of trend (catch bear bounces)
            df["setup"] = (rsi < 30)
        df["date"] = g["date"].values; df["close"] = c.values; df["symbol"] = g["symbol"].values
        parts.append(df)
    D = pd.concat(parts, ignore_index=True); D["yr"] = D["date"].dt.year
    return D


def wf(D, hold):
    D = D[D["setup"]].dropna(subset=FEATCOLS + ["label", "pnl"]).copy()
    trades = []
    for ty in range(2022, 2027):
        cut = pd.Timestamp(f"{ty}-01-01") - pd.Timedelta(days=hold + 3)
        tr = D[D["date"] < cut]; te = D[D["date"].dt.year == ty]
        if len(tr) < 500 or len(te) < 20:
            continue
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(tr[FEATCOLS], tr["label"])
        thr = np.quantile(m.predict_proba(tr[FEATCOLS])[:, 1], TOPQ)
        te = te.copy(); te["pred"] = m.predict_proba(te[FEATCOLS])[:, 1]
        sel = te[te["pred"] >= thr]
        for _, r in sel.iterrows():
            trades.append((r["symbol"], r["date"], r["yr"], r["pnl"]))
    T = pd.DataFrame(trades, columns=["symbol", "date", "yr", "pnl"]).sort_values(["symbol", "date"])
    keep = []; last = {}
    for _, r in T.iterrows():
        ld = last.get(r["symbol"])
        if ld is None or (r["date"] - ld).days > hold * 1.6:
            keep.append(True); last[r["symbol"]] = r["date"]
        else:
            keep.append(False)
    return T[keep]


if __name__ == "__main__":
    oh = load(100)
    Y = list(range(2022, 2027))
    print("cfg                    | Σtot  WR  |  " + "  ".join(f"{y}" for y in Y))
    for setup_kind in ("dip_uptrend", "deep_oversold", "oversold_any"):
        for hold in (3, 5, 8):
            D = build_cfg(oh, hold, setup_kind)
            T = wf(D, hold)
            if len(T) == 0:
                print(f"{setup_kind:14s} h{hold:<2d} | no trades"); continue
            yr = {y: g["pnl"].sum() for y, g in T.groupby("yr")}
            wr = 100 * (T.pnl > 0).mean()
            per = "  ".join(f"{yr.get(y,0):+5.1f}" for y in Y)
            print(f"{setup_kind:14s} h{hold:<2d} | {T.pnl.sum():+5.1f} {wr:3.0f}% | {per}  (n={len(T)})", flush=True)
    print("HOLDSWEEP_DONE")
