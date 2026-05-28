"""Forward-return classification target.

For each row `t`:
  fwd_return(t) = close[t + horizon] / close[t] - 1
  target(t) = +1 if fwd_return >= gain_threshold
            = -1 if fwd_return <= -loss_threshold
            =  0 otherwise

Rows where the future window is incomplete get target = NaN — the splitter
drops them from training. This — combined with the splitter's `gap_days` trim
— is what blocks forward-label leakage across the train/test boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForwardReturnTarget:
    horizon: int = 5
    gain_threshold: float = 0.04
    loss_threshold: float = 0.04

    def fwd_return(self, close: pd.Series) -> pd.Series:
        return close.shift(-self.horizon) / close - 1.0

    def labels(self, fwd: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=fwd.index, dtype="float64")
        mask = fwd.notna()
        vals = fwd.where(mask)
        out[mask & (vals >= self.gain_threshold)] = 1.0
        out[mask & (vals <= -self.loss_threshold)] = -1.0
        out[
            mask & (vals.between(-self.loss_threshold, self.gain_threshold, inclusive="neither"))
        ] = 0.0
        return out

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["fwd_return"] = out.groupby("symbol")[close_col].transform(self.fwd_return)
        out["target"] = self.labels(out["fwd_return"])
        return out
