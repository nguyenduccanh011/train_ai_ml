"""Meta-priority: walk-forward LGBM on completed base trades (t_pnl target).

Decision timepoint = fill-date (sorts same-day fills); verified causal/immaterial.
Deterministic: LGBM deterministic=True + force_col_wise=True + random_state=1.
"""

from __future__ import annotations

import pandas as pd

from stock_ml.portfolio.constants import FCOLS


def meta_features(q: pd.DataFrame) -> pd.DataFrame:
    """q: columns symbol,date,open,high,low,close,volume (trade symbols only)."""
    q = q.copy()
    q["date"] = pd.to_datetime(q["date"])
    out = []
    for s, g in q.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        c = g["close"]
        g["ret5"] = c.pct_change(5)
        g["ret20"] = c.pct_change(20)
        g["ret60"] = c.pct_change(60)
        g["vol20"] = c.pct_change().rolling(20).std()
        g["dist_h20"] = c / c.rolling(20).max() - 1
        g["dist_h63"] = c / c.rolling(63).max() - 1
        g["dist_l20"] = c / c.rolling(20).min() - 1
        g["dist_l63"] = c / c.rolling(63).min() - 1
        g["ma20r"] = c / c.rolling(20).mean() - 1
        g["ma50r"] = c / c.rolling(50).mean() - 1
        tr = pd.concat(
            [g["high"] - g["low"], (g["high"] - c.shift()).abs(), (g["low"] - c.shift()).abs()],
            axis=1,
        ).max(axis=1)
        g["atr_pct"] = tr.rolling(14).mean() / c
        g["updays10"] = (c.diff() > 0).rolling(10).sum()
        g["volr"] = g["volume"] / g["volume"].rolling(20).mean()
        out.append(
            g[
                [
                    "symbol",
                    "date",
                    "ret5",
                    "ret20",
                    "ret60",
                    "vol20",
                    "dist_h20",
                    "dist_h63",
                    "dist_l20",
                    "dist_l63",
                    "ma20r",
                    "ma50r",
                    "atr_pct",
                    "updays10",
                    "volr",
                ]
            ]
        )
    df = pd.concat(out)
    df["rs_mom20"] = df.groupby("date")["ret20"].rank(pct=True)
    df["rs_mom60"] = df.groupby("date")["ret60"].rank(pct=True)
    return df


def meta_priority(
    base_trades: pd.DataFrame, signals: pd.DataFrame, ohlcv_frame: pd.DataFrame
) -> dict:
    """Walk-forward LGBM meta-priority (t_pnl) on completed base trades -> {(symbol, edkey): prio}."""
    from lightgbm import LGBMRegressor

    feat = meta_features(ohlcv_frame)
    tr = base_trades[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].copy()
    tr = tr[tr.exit_date.notna()]
    sig = signals[(signals.signal == 1) & signals.score.notna()][
        ["symbol", "date", "score", "exit_score"]
    ].copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    tr["t_pnl"] = tr.exit_price / tr.entry_price - 1.0
    tr["yr"] = tr.entry_date.dt.year
    sig["date"] = pd.to_datetime(sig["date"])
    parts = []
    for s, tg in tr.groupby("symbol"):
        sg = sig[sig.symbol == s].sort_values("date")
        m = pd.merge_asof(
            tg.sort_values("entry_date"),
            sg[["date", "score", "exit_score"]].rename(columns={"date": "entry_date"}),
            on="entry_date",
            direction="backward",
        )
        parts.append(m)
    tr = pd.concat(parts).merge(
        feat.rename(columns={"date": "entry_date"}), on=["symbol", "entry_date"], how="left"
    )
    tr["edkey"] = tr.entry_date.dt.strftime("%Y-%m-%d")
    pm = {}
    last_year = int(tr.yr.max())
    for ty in range(2021, last_year + 1):
        train = tr[tr.exit_date < f"{ty}-01-01"].dropna(subset=FCOLS + ["t_pnl"])
        test = tr[tr.yr == ty].dropna(subset=FCOLS)
        if len(train) < 100 or not len(test):
            continue
        mdl = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            min_data_in_leaf=30,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l2=1.0,
            verbose=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=1,
        )
        mdl.fit(train[FCOLS], train["t_pnl"])
        for (_, row), p in zip(test.iterrows(), mdl.predict(test[FCOLS])):
            pm[(row.symbol, row.edkey)] = float(p)
    return pm
