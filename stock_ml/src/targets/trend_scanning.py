"""Trend-Scanning EXIT regression target (Lopez de Prado trend-scanning, sell-side adaptation).

Standard fixed-horizon / min-max exit labels are path- and regime-blind. Trend Scanning labels a bar by
the STATISTICAL STRENGTH of the forward trend: fit a linear regression of log-price over several forward
windows L, take the window with the largest |t-value| of the slope (the most statistically-significant
forward trend), and label by that t-value. For an EXIT head (we are long), we want HIGH = sell, so:

    target(t) = -t_value_of_the_dominant_forward_trend

  - forward trend strongly DOWN (t very negative) -> target HIGH  -> SELL (the move is turning over)
  - forward trend strongly UP   (t very positive) -> target LOW   -> HOLD (let the trend run)
  - choppy / no significant trend (t ~ 0)          -> target ~0   -> neutral

The t-value is intrinsically vol/regime-aware (a 3% move in low vol is significant, in high vol is noise),
which targets the project's bottleneck: the velocity exit (IC ~0.04) is a wash, and the exit gives back
~77% of each peak. The label spans c[t+1 .. t+max_window] (forward span = max_window). Tail-NaN where the
longest forward window runs past the series end (fail-loud require_no_nan guard intact).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendScanningExitTarget:
    min_window: int = 5
    max_window: int = 20
    step: int = 5
    use_log: bool = True
    windows: tuple = field(default=())

    def _wins(self):
        if self.windows:
            return [int(w) for w in self.windows]
        return list(range(self.min_window, self.max_window + 1, self.step))

    def _forward_tvalue(self, y: np.ndarray, L: int) -> np.ndarray:
        """t-value of the slope of y over each window of length L STARTING at s; returned indexed by s."""
        n = len(y)
        cs = np.concatenate([[0.0], np.cumsum(y)])
        cs2 = np.concatenate([[0.0], np.cumsum(y * y)])
        kk = np.arange(n, dtype=float)
        csk = np.concatenate([[0.0], np.cumsum(kk * y)])
        s = np.arange(0, n - L + 1)
        Sy = cs[s + L] - cs[s]
        Syy = cs2[s + L] - cs2[s]
        Skc = csk[s + L] - csk[s]
        Sxy = Skc - s * Sy                       # sum of (j*y) with j = position in window
        Sx = L * (L - 1) / 2.0
        xc = np.arange(L) - (L - 1) / 2.0
        Sxx_c = float((xc * xc).sum())
        b = (Sxy - Sx * Sy / L) / Sxx_c
        Syy_c = Syy - Sy * Sy / L
        SSE = np.clip(Syy_c - b * b * Sxx_c, 1e-12, None)
        se_b = np.sqrt(SSE / (L - 2) / Sxx_c)
        return b / (se_b + 1e-12)                # indexed by s (window start)

    def _label(self, close: pd.Series) -> pd.Series:
        c = close.to_numpy(float)
        n = len(c)
        if n < self.min_window + 2:
            return pd.Series(np.nan, index=close.index)
        y = np.log(np.clip(c, 1e-9, None)) if self.use_log else c
        mats = []
        for L in self._wins():
            arr = np.full(n, np.nan)
            if n >= L + 1 and L >= 3:
                tval = self._forward_tvalue(y, L)        # indexed by window-start s
                s = np.arange(0, n - L + 1)
                arr[s[1:] - 1] = tval[1:]                # bar t = s-1 (forward window starts at t+1)
            mats.append(arr)
        M = np.vstack(mats).T                            # (n, num_L)
        allnan = np.all(np.isnan(M), axis=1)
        absf = np.where(np.isnan(M), -1.0, np.abs(M))
        bi = absf.argmax(axis=1)
        best = M[np.arange(n), bi]
        best[allnan] = np.nan
        return pd.Series(-best, index=close.index)       # high = forward downtrend = SELL

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._label)
        return out
