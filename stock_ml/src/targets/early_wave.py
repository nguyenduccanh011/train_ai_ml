"""early_wave_v2 entry target + companion exit target (3-class classification).

Faithful port of the "early_wave_v2" label from the source recipe. For each bar i
(per symbol), with back=short_window, fw=forward_window:

    future   = close[i+1 : i+1+fw]
    max_gain = (max(future) - close[i]) / close[i]
    max_loss = (min(future) - close[i]) / close[i]
    past_range      = (max(high[i-back:i+1]) - min(low[i-back:i+1])) / close[i]
    is_accumulating = past_range < 0.15
    rule_trigger    = i>=26 and macd_hist[i]>0 and close[i]>ma20[i] and close[i]>open[i]
    is_downtrend    = (close[i]-close[i-long_window])/close[i-long_window] < -0.10

    target = +1  (BUY) if (is_accumulating and max_gain>=gain and max_loss>-loss)
                       or (rule_trigger and max_gain>=gain*0.7 and max_loss>-loss*1.3)
           = -1  (AVOID) elif classes==3 and is_downtrend and max_gain<0.03
           =  0  (NEUTRAL) otherwise

Bars without a complete forward window (i+fw>=n) get NaN so the walk-forward
splitter drops them from training — this + gap_days blocks forward-label leakage.

The exit companion (EarlyWaveExitTarget) labels a bar -1 ("sell") when the forward
window contains a drawdown worse than -loss_threshold, else 0. The dual-ML exit
classifier is trained on (target == -1), so its positive class = "a drop is coming".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_OHLC = ("open", "high", "low", "close")


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).to_numpy()


@dataclass(frozen=True)
class EarlyWaveV2Target:
    short_window: int = 5
    long_window: int = 20
    forward_window: int = 21
    gain_threshold: float = 0.033125
    loss_threshold: float = 0.0165625
    classes: int = 3

    def _labels(self, g: pd.DataFrame) -> pd.Series:
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        open_ = g["open"].to_numpy(dtype=float)
        n = len(close)
        out = np.full(n, np.nan, dtype=np.float64)

        c = pd.Series(close)
        macd_hist = _macd_hist(c)
        ma20 = c.rolling(self.long_window, min_periods=self.long_window).mean().to_numpy()

        back, fw = self.short_window, self.forward_window
        gain, loss = self.gain_threshold, self.loss_threshold
        for i in range(n):
            if i + fw >= n:
                continue  # incomplete forward window -> NaN (dropped by splitter)
            ci = close[i]
            future = close[i + 1 : i + 1 + fw]
            max_gain = (future.max() - ci) / ci
            max_loss = (future.min() - ci) / ci

            lo = i - back if i - back > 0 else 0
            past_range = (high[lo : i + 1].max() - low[lo : i + 1].min()) / ci
            is_accumulating = past_range < 0.15

            rule_trigger = (
                i >= 26 and macd_hist[i] > 0 and ci > ma20[i] and ci > open_[i]
            )
            is_downtrend = (
                i >= self.long_window
                and (ci - close[i - self.long_window]) / close[i - self.long_window] < -0.10
            )

            if (is_accumulating and max_gain >= gain and max_loss > -loss) or (
                rule_trigger and max_gain >= gain * 0.7 and max_loss > -loss * 1.3
            ):
                out[i] = 1.0
            elif self.classes == 3 and is_downtrend and max_gain < 0.03:
                out[i] = -1.0
            else:
                out[i] = 0.0
        return pd.Series(out, index=g.index)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in _OHLC if c not in df.columns]
        if missing:
            raise ValueError(f"EarlyWaveV2Target needs OHLC columns, missing {missing}")
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        parts = [self._labels(g) for _, g in out.groupby("symbol", sort=False)]
        out["target"] = pd.concat(parts).reindex(out.index)
        return out


@dataclass(frozen=True)
class EarlyWaveExitTarget:
    forward_window: int = 21
    loss_threshold: float = 0.03725
    # sell_high=True emits +1 (drop coming) / 0 instead of -1 / 0, so a REGRESSION exit
    # head's prediction ~ P(drop) is HIGH = sell — the convention the dual-ML recombine
    # z(exit) sell band expects (velocity/risk_exit/downleg all emit high = sell). The
    # legacy default (-1/0) keeps the classifier path (trained on target == -1) intact.
    sell_high: bool = False

    def _labels(self, g: pd.DataFrame) -> pd.Series:
        close = g["close"].to_numpy(dtype=float)
        n = len(close)
        out = np.full(n, np.nan, dtype=np.float64)
        fw, loss = self.forward_window, self.loss_threshold
        pos = 1.0 if self.sell_high else -1.0
        for i in range(n):
            if i + fw >= n:
                continue
            ci = close[i]
            max_loss = (close[i + 1 : i + 1 + fw].min() - ci) / ci
            out[i] = pos if max_loss <= -loss else 0.0
        return pd.Series(out, index=g.index)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        parts = [self._labels(g) for _, g in out.groupby("symbol", sort=False)]
        out["target"] = pd.concat(parts).reindex(out.index)
        return out
