from __future__ import annotations

import numpy as np
import pandas as pd


class VolumeAdvancedBlock:
    """OBV, VWAP ratio, MFI, CMF."""

    name = "volume_advanced"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        v = df["volume"]
        high = df["high"]
        low = df["low"]

        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_slope_10"] = obv.diff(10) / obv.rolling(10).mean().replace(0, np.nan)

        cum_vp = (c * v).rolling(20).sum()
        cum_v = v.rolling(20).sum()
        vwap = cum_vp / cum_v.replace(0, np.nan)
        df["vwap_ratio"] = c / vwap - 1

        tp = (high + low + c) / 3
        mf = tp * v
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        mfr = pos_mf / neg_mf.replace(0, np.nan)
        df["mfi_14"] = 100 - (100 / (1 + mfr))

        clv = ((c - low) - (high - c)) / (high - low).replace(0, np.nan)
        df["cmf_20"] = (clv * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

        return df

    def get_feature_names(self) -> list[str]:
        return ["obv", "obv_slope_10", "vwap_ratio", "mfi_14", "cmf_20"]
