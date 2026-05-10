from __future__ import annotations

import pandas as pd


class VolatilityBlock:
    """Bollinger Bands, ATR, Keltner Channels."""

    name = "volatility"
    requires = ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        high = df["high"]
        low = df["low"]

        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_upper"] = (sma20 + 2 * std20 - c) / c
        df["bb_lower"] = (c - sma20 + 2 * std20) / c
        df["bb_width"] = (4 * std20) / sma20.replace(0, None)
        df["bb_pctb"] = (c - (sma20 - 2 * std20)) / (4 * std20).replace(0, None)

        tr = pd.concat(
            [high - low, (high - c.shift(1)).abs(), (low - c.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr_14"] / c

        ema20 = c.ewm(span=20, adjust=False).mean()
        df["keltner_upper"] = (ema20 + 2 * df["atr_14"] - c) / c
        df["keltner_lower"] = (c - ema20 + 2 * df["atr_14"]) / c

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_pctb",
            "atr_14",
            "atr_ratio",
            "keltner_upper",
            "keltner_lower",
        ]
