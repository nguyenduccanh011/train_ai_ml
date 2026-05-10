from __future__ import annotations

import numpy as np
import pandas as pd

from src.components.targets.early_wave import _early_exit_signal


class EarlyWaveV2Target:
    """Widened early-wave target (V35a).

    Relaxed accumulation check + MACD/MA20 rule-trigger branch.
    Mirrors legacy TargetGenerator(target_type="early_wave_v2").
    """

    name = "early_wave_v2"

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 20,
        forward_window: int = 10,
        gain_threshold: float = 0.05,
        loss_threshold: float = 0.03,
        n_classes: int = 3,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.forward_window = forward_window
        self.gain_threshold = gain_threshold
        self.loss_threshold = loss_threshold
        self.n_classes = n_classes

    @property
    def num_classes(self) -> int:
        return self.n_classes

    @property
    def supports_exit_labels(self) -> bool:
        return True

    def generate_entry_labels(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        open_ = df["open"].values if "open" in df.columns else close
        n = len(close)
        fw = self.forward_window
        back = self.short_window
        down_win = self.long_window
        gain_thresh = self.gain_threshold
        loss_thresh = self.loss_threshold

        close_s = pd.Series(close)
        ema12 = close_s.ewm(span=12, adjust=False).mean()
        ema26 = close_s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_signal).values
        ma20 = close_s.rolling(20).mean().values

        targets = np.full(n, 0.0)

        for i in range(n):
            if i + fw >= n:
                targets[i] = np.nan
                continue
            future = close[i + 1 : i + 1 + fw]
            max_gain = (np.max(future) - close[i]) / close[i] if close[i] > 0 else 0
            max_loss = (np.min(future) - close[i]) / close[i] if close[i] > 0 else 0

            if i >= back:
                past_h = np.max(high[i - back : i + 1])
                past_l = np.min(low[i - back : i + 1])
                past_range = (past_h - past_l) / close[i] if close[i] > 0 else 1
                is_accumulating = past_range < 0.15
            else:
                is_accumulating = False

            rule_trigger = (
                i >= 26
                and not np.isnan(macd_hist[i])
                and not np.isnan(ma20[i])
                and macd_hist[i] > 0
                and close[i] > ma20[i]
                and close[i] > open_[i]
            )

            if i >= down_win:
                long_ret = (
                    (close[i] - close[i - down_win]) / close[i - down_win]
                    if close[i - down_win] > 0
                    else 0
                )
                is_downtrend = long_ret < -0.10
            else:
                is_downtrend = False

            if (
                is_accumulating
                and max_gain >= gain_thresh
                and max_loss > -loss_thresh
                or rule_trigger
                and max_gain >= gain_thresh * 0.7
                and max_loss > -loss_thresh * 1.3
            ):
                targets[i] = 1.0
            elif self.n_classes == 3 and is_downtrend and max_gain < 0.03:
                targets[i] = -1.0
            else:
                targets[i] = 0.0

        return pd.Series(targets, index=df.index)

    def generate_exit_labels(
        self,
        df: pd.DataFrame,
        forward_window: int,
        loss_threshold: float,
    ) -> pd.Series | None:
        sell = _early_exit_signal(df["close"].values, forward_window, loss_threshold)
        return pd.Series(sell, index=df.index)
