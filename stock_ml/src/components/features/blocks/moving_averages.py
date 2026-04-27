from __future__ import annotations

import pandas as pd


class MovingAverageBlock:
    """SMA, EMA, price-to-MA ratios, crossovers."""

    name = "moving_averages"
    requires = ["close"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]

        for w in [5, 10, 20, 50]:
            df[f"sma_{w}"] = c.rolling(w).mean()
        for w in [10, 20]:
            df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

        df["price_to_sma20"] = c / df["sma_20"] - 1
        df["price_to_sma50"] = c / df["sma_50"] - 1
        df["sma10_cross_sma20"] = (df["sma_10"] > df["sma_20"]).astype(int)
        return df

    def get_feature_names(self) -> list[str]:
        return [
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_10",
            "ema_20",
            "price_to_sma20",
            "price_to_sma50",
            "sma10_cross_sma20",
        ]
