"""Forward-return regression target.

For each row `t`:
  fwd_return(t) = close[t + horizon] / close[t] - 1

Output raw float return (not classified). Signal generation uses threshold:
  signal = +1 if pred_return > +threshold
         = -1 if pred_return < -threshold
         =  0 otherwise
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForwardReturnRegressionTarget:
    """Predict forward return as regression (float ∈ ℝ)."""

    horizon: int = 5

    def fwd_return(self, close: pd.Series) -> pd.Series:
        return close.shift(-self.horizon) / close - 1.0

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["fwd_return"] = out.groupby("symbol")[close_col].transform(self.fwd_return)
        out["target"] = out["fwd_return"]
        return out
