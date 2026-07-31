"""Stage-2a: train the 'remaining-run' head walk-forward on the FULL (symbol,date) grid and save
causal OOF run-scores -> parquet. The engine loads this as a hold modulator (like score3 but fed by
explicit leg-structure + momentum + volume). Causal: for test year Y, train only on bars dated
< (Jan-1-Y minus GAP days); target = forward 40-bar MFE (NaN-tailed, so no leak).
"""

from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

PCT = 0.06
HOR = 40
GAP_DAYS = 85
TRAIN_YEARS = 2
OUT = REPO / "results" / "_research_2429" / "runscore.parquet"
UNIVERSE = None  # all symbols traded by the champion


def causal_leg(close):
    n = len(close)
    d = 0
    ei = 0
    ep = close[0]
    prev = 0
    conf = {}
    for i in range(1, n):
        p = close[i]
        if d >= 0 and p > ep:
            ep = p
            ei = i
            d = 1
        elif d <= 0 and p < ep:
            ep = p
            ei = i
            d = -1
        elif d == 1 and p <= ep * (1 - PCT):
            conf[i] = (ei, +1, ei - prev, abs(close[ei] / close[prev] - 1) if close[prev] else None)
            prev = ei
            d = -1
            ep = p
            ei = i
        elif d == -1 and p >= ep * (1 + PCT):
            conf[i] = (ei, -1, None, None)
            prev = ei
            d = 1
            ep = p
            ei = i
    active = np.zeros(n, int)
    tl = np.full(n, np.nan)
    ta = np.full(n, np.nan)
    last = 0
    uls = []
    uas = []
    ctl = np.nan
    cta = np.nan
    for i in range(n):
        if i in conf:
            pid, k, ul, ua = conf[i]
            last = pid
            if k == +1 and ul:
                uls.append(ul)
                ctl = np.median(uls)
            if k == +1 and ua:
                uas.append(ua)
                cta = np.median(uas)
        active[i] = last
        tl[i] = ctl
        ta[i] = cta
    age = np.arange(n) - active
    lan = np.nan_to_num(age / np.where(tl > 0, tl, np.nan) - 1, nan=0.0)
    camp = np.abs(close / np.where(close[active] == 0, np.nan, close[active]) - 1)
    lamp = np.nan_to_num(camp / np.where(ta > 0, ta, np.nan) - 1, nan=0.0)
    return lan, lamp, age.astype(float), camp


FEATS = [
    "legage",
    "legamp",
    "cur_amp",
    "cur_age",
    "ext20",
    "ext50",
    "ret20",
    "ret60",
    "volr20",
    "accdist",
    "clpos",
]


def build_panel(symbols):
    con = duckdb.connect(str(REPO / "market_data" / "market.duckdb"), read_only=True)
    ph = ",".join(["?"] * len(symbols))
    px = con.execute(
        f"SELECT symbol,date,close,high,low,volume FROM ohlcv WHERE timeframe='1D' AND symbol IN ({ph}) ORDER BY symbol,date",
        symbols,
    ).df()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    out = []
    for sym, g in px.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        cl = g.close.to_numpy(float)
        hi = g.high.to_numpy(float)
        lo = g.low.to_numpy(float)
        vol = g.volume.to_numpy(float)
        n = len(cl)
        if n < 120:
            continue
        lan, lamp, age, camp = causal_leg(cl)
        ma20 = pd.Series(cl).rolling(20).mean().to_numpy()
        ma50 = pd.Series(cl).rolling(50).mean().to_numpy()
        vma20 = pd.Series(vol).rolling(20).mean().to_numpy()
        rng = np.where((hi - lo) <= 0, 1e-9, hi - lo)
        clpos = (cl - lo) / rng
        volr = vol / np.where(vma20 > 0, vma20, np.nan)
        accb = ((volr > 1.2) & (clpos > 0.55)).astype(float)
        distb = ((volr > 1.2) & (clpos < 0.45)).astype(float)
        adbal = (pd.Series(accb).rolling(20).sum() - pd.Series(distb).rolling(20).sum()).to_numpy()
        ret20 = cl / np.concatenate([[np.nan] * 20, cl[:-20]]) - 1
        ret60 = cl / np.concatenate([[np.nan] * 60, cl[:-60]]) - 1
        # forward 40-bar MFE target (NaN-tailed)
        fwd = np.full(n, np.nan)
        for i in range(n - 1):
            j = min(i + HOR, n)
            fwd[i] = hi[i + 1 : j].max() / cl[i] - 1 if j > i + 1 else np.nan
        df = pd.DataFrame(
            dict(
                symbol=sym,
                date=g.date.to_numpy(),
                legage=lan,
                legamp=lamp,
                cur_amp=camp,
                cur_age=age,
                ext20=cl / np.where(ma20 > 0, ma20, np.nan) - 1,
                ext50=cl / np.where(ma50 > 0, ma50, np.nan) - 1,
                ret20=ret20,
                ret60=ret60,
                volr20=np.nan_to_num(volr, nan=1.0),
                accdist=np.nan_to_num(adbal, nan=0.0),
                clpos=clpos,
                target=fwd,
            )
        )
        out.append(df)
    panel = pd.concat(out, ignore_index=True)
    return panel.replace([np.inf, -np.inf], np.nan)


def main():
    import json

    # champion universe symbols
    t = pd.read_csv(REPO / "_tmp_analysis" / "champ2646_trades.csv")
    symbols = sorted(t.symbol.unique())
    print(f"building panel for {len(symbols)} symbols...", flush=True)
    panel = build_panel(symbols)
    panel["year"] = panel.date.dt.year
    print(f"panel rows={len(panel)}", flush=True)
    panel["runscore"] = np.nan
    for ty in range(2020, 2027):
        cut = pd.Timestamp(year=ty, month=1, day=1) - pd.Timedelta(days=GAP_DAYS)
        tr = panel[
            (panel.date < cut)
            & (panel.date >= pd.Timestamp(year=ty - TRAIN_YEARS, month=1, day=1))
            & panel.target.notna()
        ].dropna(subset=FEATS)
        te = panel[panel.year == ty]
        if len(tr) < 500 or te.empty:
            continue
        m = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=60,
            verbose=-1,
            random_state=42,
            deterministic=True,
            force_col_wise=True,
        )
        m.fit(tr[FEATS], tr["target"])
        te_ok = te.dropna(subset=FEATS)
        panel.loc[te_ok.index, "runscore"] = m.predict(te_ok[FEATS])
        print(f"  test {ty}: train={len(tr)} pred={len(te_ok)}", flush=True)
    res = panel[panel.runscore.notna()][["symbol", "date", "runscore"]].copy()
    # z-score runscore per test-year cross-section (so the engine gate is regime-stable)
    res["year"] = res.date.dt.year
    res["runscore_z"] = res.groupby("year").runscore.transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)
    )
    res[["symbol", "date", "runscore", "runscore_z"]].to_parquet(OUT)
    print(f"saved {len(res)} run-scores -> {OUT}")
    print("TRAIN_RUNSCORE_DONE")


if __name__ == "__main__":
    main()
