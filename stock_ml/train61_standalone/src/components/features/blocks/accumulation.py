from __future__ import annotations

import numpy as np
import pandas as pd


class AccumulationBlock:
    """Accumulation detection: vol contraction, range compression, dist-to-high, vol dry/spike."""

    name = "accumulation"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"].replace(0, np.nan)

        ret1 = c.pct_change(1)
        std5 = ret1.rolling(5).std()
        std20 = ret1.rolling(20).std()
        df["acc_vol_contraction"] = std5 / std20.replace(0, np.nan)

        rng = (h - l) / c.replace(0, np.nan)
        rng5 = rng.rolling(5).mean()
        rng20 = rng.rolling(20).mean()
        df["acc_range_compression"] = rng5 / rng20.replace(0, np.nan)

        for w in [10, 20, 50]:
            roll_high = h.rolling(w, min_periods=max(3, w // 3)).max()
            df[f"acc_dist_to_high_{w}"] = (roll_high - c) / c.replace(0, np.nan)

        vol5 = v.rolling(5).mean()
        vol20 = v.rolling(20).mean()
        df["acc_vol_dry_ratio"] = vol5 / vol20.replace(0, np.nan)
        df["acc_vol_spike_ratio"] = v / vol20.replace(0, np.nan)
        df["acc_vol_dry_then_spike"] = (df["acc_vol_dry_ratio"] < 0.85).astype(float) * (
            df["acc_vol_spike_ratio"] > 1.5
        ).astype(float)

        small_day = (ret1.abs() < 0.005).astype(float)
        df["acc_sideway_score_10"] = small_day.rolling(10).mean()

        sma20 = c.rolling(20).mean()
        std20p = c.rolling(20).std()
        bbw = (4 * std20p) / sma20.replace(0, np.nan)
        df["acc_bbw_pct_60d"] = bbw.rolling(60, min_periods=20).rank(pct=True)

        big_move = (ret1.abs() > 0.03).astype(int)
        days_since = np.zeros(len(df))
        cnt = 0
        bm = big_move.fillna(0).values
        for i in range(len(df)):
            cnt = 0 if bm[i] == 1 else cnt + 1
            days_since[i] = cnt
        df["acc_days_since_big_move"] = days_since

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "acc_vol_contraction",
            "acc_range_compression",
            "acc_dist_to_high_10",
            "acc_dist_to_high_20",
            "acc_dist_to_high_50",
            "acc_vol_dry_ratio",
            "acc_vol_spike_ratio",
            "acc_vol_dry_then_spike",
            "acc_sideway_score_10",
            "acc_bbw_pct_60d",
            "acc_days_since_big_move",
        ]
