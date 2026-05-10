from __future__ import annotations

import numpy as np
import pandas as pd


class LeadingSignalsBlock:
    """Early detection features: volume precursors, squeeze, accumulation, S/R proximity."""

    name = "leading_signals"
    requires = ["high", "low", "close", "volume"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"].replace(0, np.nan)

        vol_ma5 = v.rolling(5).mean()
        vol_ma20 = v.rolling(20).mean()
        df["vol_surge_ratio"] = vol_ma5 / vol_ma20
        df["pv_divergence"] = df.get("volume_ratio_20d", v / vol_ma20) - abs(
            df.get("return_5d", c.pct_change(5))
        )

        if "bb_width" not in df.columns:
            sma20 = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            bb_w = (4 * std20) / sma20.replace(0, np.nan)
        else:
            bb_w = df["bb_width"]
        df["bb_width_percentile"] = bb_w.rolling(60).rank(pct=True)

        tr_parts = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(
            axis=1
        )
        atr5 = tr_parts.rolling(5).mean()
        atr20 = tr_parts.rolling(20).mean()
        df["atr_contraction"] = atr5 / atr20.replace(0, np.nan)

        df["close_position_in_range"] = (c - l) / (h - l).replace(0, np.nan)
        df["close_pos_ma5"] = df["close_position_in_range"].rolling(5).mean()

        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        obv_slope = obv.diff(10) / obv.rolling(10).mean().abs().replace(0, np.nan)
        price_slope = c.pct_change(10)
        df["obv_price_divergence"] = obv_slope - price_slope

        high_20 = h.rolling(20).max()
        low_20 = l.rolling(20).min()
        df["dist_to_resistance"] = (high_20 - c) / c
        df["dist_to_support"] = (c - low_20) / c
        df["range_position_20d"] = (c - low_20) / (high_20 - low_20).replace(0, np.nan)

        lows = l.values
        hl_count = pd.Series(0.0, index=df.index)
        for i in range(4, len(lows)):
            cnt = 0
            for j in range(1, 5):
                if i - j >= 0 and lows[i - j + 1] > lows[i - j]:
                    cnt += 1
            hl_count.iloc[i] = cnt
        df["higher_lows_count"] = hl_count

        daily_range_pct = (h - l) / c
        df["consolidation_score"] = (daily_range_pct < 0.02).rolling(10).sum()

        if "rsi_14" in df.columns:
            rsi = df["rsi_14"]
        else:
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

        df["rsi_slope_5d"] = rsi.diff(5)
        df["price_rsi_divergence"] = rsi.diff(10) - (c.pct_change(10) * 100)

        df["breakout_setup_score"] = (
            (df["vol_surge_ratio"] > 1.2).astype(float)
            + (df["bb_width_percentile"] < 0.3).astype(float)
            + (df["dist_to_resistance"] < 0.02).astype(float)
            + (df["close_pos_ma5"] > 0.6).astype(float)
            + (df["higher_lows_count"] >= 3).astype(float)
        )

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "vol_surge_ratio",
            "pv_divergence",
            "bb_width_percentile",
            "atr_contraction",
            "close_position_in_range",
            "close_pos_ma5",
            "obv_price_divergence",
            "dist_to_resistance",
            "dist_to_support",
            "range_position_20d",
            "higher_lows_count",
            "consolidation_score",
            "rsi_slope_5d",
            "price_rsi_divergence",
            "breakout_setup_score",
        ]
