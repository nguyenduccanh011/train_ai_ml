from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityRegimeBlock:
    """ATR percentile, compression duration, post-expansion failure."""

    name = "volatility_regime"
    requires = ["open", "high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        o = df["open"]

        tr = pd.concat(
            [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14).mean()

        df["atr_percentile_60d"] = atr14.rolling(60, min_periods=20).rank(pct=True)
        df["true_range_percentile"] = tr.rolling(60, min_periods=20).rank(pct=True)
        df["overnight_gap_pct"] = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)

        if "bb_width" in df.columns:
            bb_w = df["bb_width"]
        else:
            sma20 = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            bb_w = (4 * std20) / sma20.replace(0, np.nan)
        bb_pct20 = bb_w.rolling(60, min_periods=20).rank(pct=True)
        squeeze = (bb_pct20 < 0.2).astype(float)
        compress = np.zeros(len(df))
        cnt = 0
        for i in range(len(df)):
            if squeeze.iloc[i] == 1.0:
                cnt += 1
            else:
                cnt = 0
            compress[i] = cnt
        df["compression_duration"] = compress

        ret1d = c.pct_change(1)
        vol5 = ret1d.rolling(5).std()
        vol20 = ret1d.rolling(20).std()
        expansion = (vol5 > vol20 * 1.5).astype(float)
        post_fail = np.zeros(len(df))
        for i in range(3, len(df)):
            if expansion.iloc[i - 3] == 1.0 or expansion.iloc[i - 2] == 1.0:
                recent_ret = (c.iloc[i] - c.iloc[i - 2]) / c.iloc[i - 2] if c.iloc[i - 2] > 0 else 0
                if abs(recent_ret) < 0.01:
                    post_fail[i] = 1.0
        df["post_expansion_failure"] = post_fail

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "atr_percentile_60d",
            "true_range_percentile",
            "overnight_gap_pct",
            "compression_duration",
            "post_expansion_failure",
        ]
