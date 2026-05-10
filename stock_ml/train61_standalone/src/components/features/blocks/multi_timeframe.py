from __future__ import annotations

import pandas as pd


class MultiTimeframeBlock:
    """Weekly returns, weekly MAs, trend alignment, weekly swing proximity."""

    name = "multi_timeframe"
    requires = ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]

        df["weekly_return_1w"] = c.pct_change(5)
        df["weekly_return_4w"] = c.pct_change(20)

        wma20 = c.rolling(100).mean()
        wma50 = c.rolling(250).mean()
        df["price_vs_wma20"] = c / wma20.replace(0, None) - 1
        df["price_vs_wma50"] = c / wma50.replace(0, None) - 1

        sma20_daily = c.rolling(20).mean()
        daily_trend = (c > sma20_daily).astype(int)
        weekly_trend = (c > wma20).astype(int)
        df["weekly_trend_alignment"] = (daily_trend == weekly_trend).astype(float)

        high_20w = h.rolling(100, min_periods=20).max()
        low_20w = l.rolling(100, min_periods=20).min()
        df["weekly_swing_proximity"] = (c - low_20w) / (high_20w - low_20w).replace(0, None)

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "weekly_return_1w",
            "weekly_return_4w",
            "price_vs_wma20",
            "price_vs_wma50",
            "weekly_trend_alignment",
            "weekly_swing_proximity",
        ]
