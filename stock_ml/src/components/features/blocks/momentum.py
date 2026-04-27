from __future__ import annotations

import numpy as np
import pandas as pd


class MomentumBlock:
    """RSI, Stochastic, Williams %R, ROC, momentum."""

    name = "momentum"
    requires = ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]

        for period in [7, 14]:
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        low14 = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        df["williams_r"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

        df["roc_10"] = c.pct_change(10) * 100
        df["roc_20"] = c.pct_change(20) * 100
        df["momentum_10"] = c - c.shift(10)

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "rsi_7",
            "rsi_14",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "roc_10",
            "roc_20",
            "momentum_10",
        ]
