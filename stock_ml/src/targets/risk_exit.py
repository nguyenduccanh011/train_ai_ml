"""Risk-exit regression target (sell-side).

Teaches the exit regressor to sell when the forward path is *risk-dominated* — an
imminent decline, or a flat/sideways drift with no upside left — instead of selling
at swing peaks. The zigzag-peak target sells at the top of every blow-off, cutting a
continuation move short. This target keeps holding while upside remains and only
flags a sell when downside outweighs upside.

For each row `t` over a forward window of `horizon` bars:
  fwd_upside(t)   = max_{k=1..h}( close[t+k] / close[t] ) - 1     # best coming gain, >= 0
  fwd_downside(t) = 1 - min_{k=1..h}( close[t+k] / close[t] )     # deepest coming drop, >= 0
  target(t)       = fwd_downside(t) - fwd_upside(t)

High value = sell (downside dominates). Behaviour:
  - strong continuation ahead  -> upside high  -> target negative -> HOLD (don't sell the run)
  - imminent drop              -> downside high -> target high     -> SELL
  - sideways accumulation      -> both small    -> target ~0       -> sell only as risk creeps in

The dual-ML dispatch sells when ``pred_exit > exit_threshold`` (higher = stronger
sell), matching the convention of the other exit targets.

Mirrors ForwardDrawdownRegressionTarget's tail-NaN (last `horizon` rows per symbol
are NaN — the window is not yet observable), keeping the fail-loud ``require_no_nan``
guard intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskExitRegressionTarget:
    """Predict forward downside minus forward upside as regression (float ∈ ℝ)."""

    horizon: int = 10

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

    def _label(self, close: pd.Series) -> pd.Series:
        shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        window = pd.concat(shifts, axis=1)
        observable = close.shift(-self.horizon).notna()
        fwd_min = window.min(axis=1).where(observable)
        fwd_max = window.max(axis=1).where(observable)
        fwd_downside = 1.0 - fwd_min / close
        fwd_upside = fwd_max / close - 1.0
        return fwd_downside - fwd_upside

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._label)
        return out
