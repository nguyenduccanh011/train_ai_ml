from __future__ import annotations

import numpy as np
import pandas as pd


class TrendBlock:
    """MACD, ADX, CCI, Aroon."""

    name = "trend"
    requires = ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        high = df["high"]
        low = df["low"]

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat(
            [high - low, (high - c.shift(1)).abs(), (low - c.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14).mean()
        df["plus_di"] = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        df["minus_di"] = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        dx = (
            (df["plus_di"] - df["minus_di"]).abs()
            / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
            * 100
        )
        df["adx_14"] = dx.rolling(14).mean()

        tp = (high + low + c) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df["cci_20"] = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)

        df["aroon_up"] = high.rolling(25).apply(lambda x: x.argmax() / 24 * 100, raw=True)
        df["aroon_down"] = low.rolling(25).apply(lambda x: x.argmin() / 24 * 100, raw=True)

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "macd",
            "macd_signal",
            "macd_histogram",
            "plus_di",
            "minus_di",
            "adx_14",
            "cci_20",
            "aroon_up",
            "aroon_down",
        ]
