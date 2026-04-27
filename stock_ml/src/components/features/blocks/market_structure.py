from __future__ import annotations

import numpy as np
import pandas as pd


class MarketStructureBlock:
    """Pivot points, BOS, CHoCH, HH/HL regime detection."""

    name = "market_structure"
    requires = ["high", "low", "close"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(df)

        for order in [3, 5, 7]:
            ph = np.zeros(n)
            pl = np.zeros(n)
            for i in range(order, n - order):
                if all(h[i] >= h[i - j] for j in range(1, order + 1)) and all(
                    h[i] >= h[i + j] for j in range(1, min(order + 1, n - i))
                ):
                    ph[i] = 1.0
                if all(l[i] <= l[i - j] for j in range(1, order + 1)) and all(
                    l[i] <= l[i + j] for j in range(1, min(order + 1, n - i))
                ):
                    pl[i] = 1.0
            df[f"pivot_high_{order}"] = ph
            df[f"pivot_low_{order}"] = pl

        last_swing_h = np.full(n, np.nan)
        last_swing_l = np.full(n, np.nan)
        sh_val, sl_val = np.nan, np.nan
        ph5 = df["pivot_high_5"].values
        pl5 = df["pivot_low_5"].values
        for i in range(n):
            if ph5[i] == 1.0:
                sh_val = h[i]
            if pl5[i] == 1.0:
                sl_val = l[i]
            last_swing_h[i] = sh_val
            last_swing_l[i] = sl_val
        df["dist_to_last_swing_high"] = (last_swing_h - c) / np.where(c > 0, c, 1.0)
        df["dist_to_last_swing_low"] = (c - last_swing_l) / np.where(c > 0, c, 1.0)

        bos_up = np.zeros(n)
        bos_down = np.zeros(n)
        choch = np.zeros(n)
        prev_sh = np.nan
        prev_sl = np.nan
        last_direction = 0
        for i in range(1, n):
            if ph5[i] == 1.0 and not np.isnan(prev_sh):
                if h[i] > prev_sh:
                    bos_up[i] = 1.0
                    if last_direction == -1:
                        choch[i] = 1.0
                    last_direction = 1
            if pl5[i] == 1.0 and not np.isnan(prev_sl):
                if l[i] < prev_sl:
                    bos_down[i] = 1.0
                    if last_direction == 1:
                        choch[i] = -1.0
                    last_direction = -1
            if ph5[i] == 1.0:
                prev_sh = h[i]
            if pl5[i] == 1.0:
                prev_sl = l[i]
        df["bos_up"] = bos_up
        df["bos_down"] = bos_down
        df["choch"] = choch

        for window in [20, 40]:
            regime = np.zeros(n)
            for i in range(window, n):
                seg_h = h[i - window : i + 1]
                seg_l = l[i - window : i + 1]
                hh_count = sum(1 for j in range(1, len(seg_h)) if seg_h[j] > seg_h[j - 1])
                hl_count = sum(1 for j in range(1, len(seg_l)) if seg_l[j] > seg_l[j - 1])
                lh_count = sum(1 for j in range(1, len(seg_h)) if seg_h[j] < seg_h[j - 1])
                ll_count = sum(1 for j in range(1, len(seg_l)) if seg_l[j] < seg_l[j - 1])
                total = window
                bull_score = (hh_count + hl_count) / total
                bear_score = (lh_count + ll_count) / total
                if bull_score > 0.6:
                    regime[i] = 1
                elif bear_score > 0.6:
                    regime[i] = -1
            df[f"hh_hl_regime_{window}"] = regime

        return df

    def get_feature_names(self) -> list[str]:
        return [
            "pivot_high_3",
            "pivot_low_3",
            "pivot_high_5",
            "pivot_low_5",
            "pivot_high_7",
            "pivot_low_7",
            "dist_to_last_swing_high",
            "dist_to_last_swing_low",
            "bos_up",
            "bos_down",
            "choch",
            "hh_hl_regime_20",
            "hh_hl_regime_40",
        ]
