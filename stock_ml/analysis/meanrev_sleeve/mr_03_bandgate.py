"""oversold_any bounce carries a REAL dead-year edge (2023 +3.2, 2024 +2.9) but is coupled to a
2022 deep-bear blowup (-9.2, falling-knife). Test a MARKET-DRAWDOWN gate (equal-weight index dd
from 252d high) — a common-factor single-book gate (legal, like rgskip) — to skip deep-bear days
and keep the choppy-year bounces. If gating 2022 leaves 2023/2024 positive and turns net positive,
that's a live decorrelated dead-year source worth engine integration.
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stock_ml.analysis.meanrev_sleeve.mr_01_model import load, FEATCOLS
from stock_ml.analysis.meanrev_sleeve.mr_02_holdsweep import build_cfg

TOPQ = 0.80


def market_dd(oh):
    """Equal-weight index drawdown from trailing 252d high, per date."""
    px = oh.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    idx = (px / px.shift(1)).fillna(1.0).cumprod().mean(axis=1)   # equal-weight index level
    dd = idx / idx.rolling(252, min_periods=60).max() - 1.0
    return dd


def wf_gated(D, hold, dd, dd_floor):
    D = D[D["setup"]].dropna(subset=FEATCOLS + ["label", "pnl"]).copy()
    ddmap = dd.to_dict()
    D["mkt_dd"] = D["date"].map(ddmap)
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
        sel = te[(te["pred"] >= thr) & ((te["mkt_dd"] >= dd_floor) | te["mkt_dd"].isna())]
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
    dd = market_dd(oh)
    Y = list(range(2022, 2027))
    print("cfg                          | Σtot  WR  |  " + "  ".join(f"{y}" for y in Y))
    for hold in (5, 8):
        for dd_floor in (-1.0, -0.20, -0.15, -0.10):   # -1.0 = no gate (baseline)
            D = build_cfg(oh, hold, "oversold_any")
            T = wf_gated(D, hold, dd, dd_floor)
            if len(T) == 0:
                print(f"oversold_any h{hold} gate{dd_floor:+.2f} | no trades"); continue
            yr = {y: g["pnl"].sum() for y, g in T.groupby("yr")}
            wr = 100 * (T.pnl > 0).mean()
            per = "  ".join(f"{yr.get(y,0):+5.1f}" for y in Y)
            tag = "NOGATE" if dd_floor == -1.0 else f"dd>{dd_floor:+.2f}"
            print(f"oversold_any h{hold} {tag:9s} | {T.pnl.sum():+5.1f} {wr:3.0f}% | {per}  (n={len(T)})", flush=True)
    print("BANDGATE_DONE")
