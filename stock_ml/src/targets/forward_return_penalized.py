"""Drawdown-penalized forward-return regression target (entry-side).

For each row `t` over a forward window of `horizon` bars:
  fwd_return(t)   = close[t + horizon] / close[t] - 1
  fwd_downside(t) = 1 - min_{k=1..h}( close[t+k] / close[t] )     # deepest coming drop, >= 0
  target(t)       = fwd_return(t) - penalty * fwd_downside(t)

A plain forward-return target rewards catching an oversold dip that *eventually* bounces
even when the path plunges first (a 'falling knife': the model learns oversold -> buy and
keeps buying into structural crashes). Subtracting the forward drawdown makes a high
return reached through a deep intra-window plunge score BADLY, so the entry regressor is
trained to down-weight oversold-in-downtrend setups instead of chasing the rebound.

  penalty = 0  -> identical to ForwardReturnRegressionTarget (plain forward return)
  penalty = 1  -> a 10% intra-window drawdown fully offsets a 10% forward gain

Same horizon tail-NaN as ForwardReturnRegressionTarget / ForwardDrawdownRegressionTarget
(the last `horizon` rows per symbol are NaN — the window is not yet observable), so train
folds never see the tail and the fail-loud ``require_no_nan`` guard stays intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForwardReturnPenalizedRegressionTarget:
    """Predict forward return penalized by the forward drawdown (float ∈ ℝ)."""

    horizon: int = 10
    penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.penalty < 0:
            raise ValueError(f"penalty must be >= 0, got {self.penalty}")

    def _penalized(self, close: pd.Series) -> pd.Series:
        fwd_return = close.shift(-self.horizon) / close - 1.0
        shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        fwd_min = pd.concat(shifts, axis=1).min(axis=1)
        # NaN once the full window runs past the series end (mirrors forward-return).
        fwd_min = fwd_min.where(close.shift(-self.horizon).notna())
        fwd_downside = 1.0 - fwd_min / close
        return fwd_return - self.penalty * fwd_downside

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._penalized)
        return out
