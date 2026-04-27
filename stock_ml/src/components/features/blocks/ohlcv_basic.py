from __future__ import annotations

import numpy as np
import pandas as pd


class OhlcvBasicBlock:
    """Price action basics: returns, volatility, OHLC ratios."""

    name = "ohlcv_basic"
    requires = ["open", "high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"].replace(0, np.nan)

        df["return_1d"] = c.pct_change(1)
        df["return_5d"] = c.pct_change(5)
        df["return_10d"] = c.pct_change(10)
        df["return_20d"] = c.pct_change(20)
        df["log_return_1d"] = np.log(c / c.shift(1))
        df["volatility_10d"] = df["return_1d"].rolling(10).std()
        df["volatility_20d"] = df["return_1d"].rolling(20).std()
        df["high_low_range"] = (h - l) / c
        df["close_to_high"] = (h - c) / c
        df["close_to_low"] = (c - l) / c

        df["volume_ratio_5d"] = v / v.rolling(5).mean()
        df["volume_ratio_20d"] = v / v.rolling(20).mean()
        if "traded_value" in df.columns:
            tv = df["traded_value"].replace(0, np.nan)
            df["traded_value_ratio_5d"] = tv / tv.rolling(5).mean()

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "return_1d",
            "return_5d",
            "return_10d",
            "return_20d",
            "log_return_1d",
            "volatility_10d",
            "volatility_20d",
            "high_low_range",
            "close_to_high",
            "close_to_low",
            "volume_ratio_5d",
            "volume_ratio_20d",
        ]
