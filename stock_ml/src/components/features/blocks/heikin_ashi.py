from __future__ import annotations

import numpy as np
import pandas as pd


class HeikinAshiBlock:
    """Heikin-Ashi wave position, shadow ratios, reversal signals."""

    name = "heikin_ashi"
    requires = ["open", "high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        n = len(df)

        ha_close = (o + h + l + c) / 4.0
        ha_open = np.zeros(n)
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

        ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(l, np.minimum(ha_open, ha_close))
        ha_body = ha_close - ha_open
        ha_range = ha_high - ha_low
        ha_range = np.where(ha_range == 0, 1e-8, ha_range)

        ha_green = (ha_body > 0).astype(float)
        green_streak = np.zeros(n)
        red_streak = np.zeros(n)
        for i in range(n):
            if ha_green[i] == 1:
                green_streak[i] = (green_streak[i - 1] + 1) if i > 0 else 1
                red_streak[i] = 0
            else:
                red_streak[i] = (red_streak[i - 1] + 1) if i > 0 else 1
                green_streak[i] = 0

        df["ha_green"] = ha_green
        df["ha_green_streak"] = green_streak
        df["ha_red_streak"] = red_streak
        df["ha_color_switch"] = (np.diff(ha_green, prepend=ha_green[0]) != 0).astype(float)

        upper_shadow = ha_high - np.maximum(ha_open, ha_close)
        lower_shadow = np.minimum(ha_open, ha_close) - ha_low
        df["ha_upper_shadow_ratio"] = upper_shadow / ha_range
        df["ha_lower_shadow_ratio"] = lower_shadow / ha_range
        df["ha_no_lower_shadow"] = (lower_shadow < ha_range * 0.05).astype(float)
        df["ha_no_upper_shadow"] = (upper_shadow < ha_range * 0.05).astype(float)

        idx = df.index
        upper_s = pd.Series(upper_shadow / ha_range, index=idx)
        df["ha_upper_shadow_growing"] = (
            upper_s.rolling(3, min_periods=3).mean() > upper_s.rolling(10, min_periods=5).mean()
        ).astype(float)
        lower_s = pd.Series(lower_shadow / ha_range, index=idx)
        df["ha_lower_shadow_growing"] = (
            lower_s.rolling(3, min_periods=3).mean() > lower_s.rolling(10, min_periods=5).mean()
        ).astype(float)

        df["ha_body_ratio"] = np.abs(ha_body) / ha_range
        body_s = pd.Series(np.abs(ha_body) / ha_range, index=idx)
        df["ha_body_shrinking"] = (
            body_s.rolling(3, min_periods=3).mean() < body_s.rolling(10, min_periods=5).mean()
        ).astype(float)

        win = 20
        green_streak_s = pd.Series(green_streak, index=idx)
        streak_max = green_streak_s.rolling(win, min_periods=5).max().replace(0, 1)
        df["ha_streak_position"] = (green_streak_s / streak_max).fillna(0.0)

        df["ha_doji"] = (
            (np.abs(ha_body) / ha_range < 0.15)
            & (upper_shadow / ha_range > 0.2)
            & (lower_shadow / ha_range > 0.2)
        ).astype(float)

        df["ha_bearish_reversal_signal"] = (
            (green_streak >= 4)
            & (df["ha_upper_shadow_growing"] == 1)
            & (df["ha_body_shrinking"] == 1)
        ).astype(float)

        prev_red_streak = pd.Series(red_streak, index=idx).shift(1).fillna(0).values
        df["ha_bullish_reversal_signal"] = (
            (prev_red_streak >= 3)
            & (df["ha_lower_shadow_growing"] == 1)
            & (df["ha_color_switch"] == 1)
        ).astype(float)

        df["ha_early_wave"] = (
            (green_streak <= 2)
            & (green_streak >= 1)
            & (df["ha_no_lower_shadow"] == 1)
            & (df["ha_body_ratio"] > 0.5)
        ).astype(float)

        df["ha_late_wave"] = (
            (green_streak >= 5)
            & (df["ha_upper_shadow_ratio"] > 0.3)
            & (df["ha_body_shrinking"] == 1)
        ).astype(float)

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "ha_green",
            "ha_green_streak",
            "ha_red_streak",
            "ha_color_switch",
            "ha_upper_shadow_ratio",
            "ha_lower_shadow_ratio",
            "ha_no_lower_shadow",
            "ha_no_upper_shadow",
            "ha_upper_shadow_growing",
            "ha_lower_shadow_growing",
            "ha_body_ratio",
            "ha_body_shrinking",
            "ha_streak_position",
            "ha_doji",
            "ha_bearish_reversal_signal",
            "ha_bullish_reversal_signal",
            "ha_early_wave",
            "ha_late_wave",
        ]
