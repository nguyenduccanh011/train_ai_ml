from __future__ import annotations

import numpy as np
import pandas as pd


class ExhaustionBlock:
    """Upthrust, spring, climax volume, gaps, reversal bars."""

    name = "exhaustion"
    requires = ["open", "high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        v = df["volume"].values.astype(float)
        n = len(df)

        vol_ma20 = pd.Series(v).rolling(20).mean().values

        upthrust = np.zeros(n)
        spring = np.zeros(n)
        for i in range(1, n):
            rng = h[i] - l[i]
            if rng <= 0:
                continue
            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            prev_high = np.max(h[max(0, i - 20) : i]) if i >= 1 else h[i]
            prev_low = np.min(l[max(0, i - 20) : i]) if i >= 1 else l[i]
            if h[i] > prev_high and c[i] < o[i] and upper_wick > 0.5 * rng:
                upthrust[i] = 1.0
            if l[i] < prev_low and c[i] > o[i] and lower_wick > 0.5 * rng:
                spring[i] = 1.0
        df["upthrust"] = upthrust
        df["spring"] = spring

        climax_up = np.zeros(n)
        climax_down = np.zeros(n)
        for i in range(20, n):
            vol_ratio = v[i] / vol_ma20[i] if vol_ma20[i] > 0 else 1.0
            if vol_ratio < 2.0:
                continue
            rng = h[i] - l[i]
            if rng <= 0:
                continue
            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            if c[i] > c[i - 1] and upper_wick > 0.4 * rng:
                climax_up[i] = vol_ratio
            if c[i] < c[i - 1] and lower_wick > 0.4 * rng:
                climax_down[i] = vol_ratio
        df["climax_volume_up"] = climax_up
        df["climax_volume_down"] = climax_down

        gap_up = np.zeros(n)
        gap_down = np.zeros(n)
        gap_filled = np.zeros(n)
        for i in range(1, n):
            gap = (o[i] - c[i - 1]) / c[i - 1] if c[i - 1] > 0 else 0
            if gap > 0.005:
                gap_up[i] = gap
                if c[i] < o[i]:
                    gap_filled[i] = 1.0
            elif gap < -0.005:
                gap_down[i] = abs(gap)
                if c[i] > o[i]:
                    gap_filled[i] = -1.0
        df["gap_up_pct"] = gap_up
        df["gap_down_pct"] = gap_down
        df["gap_filled"] = gap_filled

        reversal = np.zeros(n)
        for i in range(3, n):
            vol_ok = (
                v[i] > vol_ma20[i] * 1.2 if not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0 else False
            )
            if c[i - 3] > c[i - 2] > c[i - 1] and c[i] > c[i - 1] and vol_ok:
                reversal[i] = 1.0
            elif c[i - 3] < c[i - 2] < c[i - 1] and c[i] < c[i - 1] and vol_ok:
                reversal[i] = -1.0
        df["reversal_3bar"] = reversal

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "upthrust",
            "spring",
            "climax_volume_up",
            "climax_volume_down",
            "gap_up_pct",
            "gap_down_pct",
            "gap_filled",
            "reversal_3bar",
        ]
