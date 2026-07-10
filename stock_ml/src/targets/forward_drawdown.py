"""Forward-downside regression target (sell-side).

For each row `t` over a forward window of `horizon` bars:
  fwd_min_return(t) = min_{k=1..h}( close[t+k] / close[t] - 1 )
  target(t)         = -fwd_min_return(t)            # = 1 - min_{k} close[t+k] / close[t]

i.e. the magnitude of the deepest close-to-close drop within the next `horizon`
bars. High value = an imminent decline → a good time to SELL. Output is a raw
float (regression). The dual-ML dispatch sells when ``pred_exit > exit_threshold``.

Mirror of ``ForwardReturnRegressionTarget`` (which the entry slot uses): both look
`horizon` bars ahead and both leave the last `horizon` rows per symbol as NaN (the
window is not yet observable), so train folds never see the tail and the fail-loud
``require_no_nan`` guard stays intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForwardDrawdownRegressionTarget:
    """Predict the forward downside (deepest coming drop) as regression (float ∈ ℝ)."""

    horizon: int = 10
    # VOL-NORMALIZE (project deep-finding: "only VOL predicts MAGNITUDE"): divide the raw
    # forward-downside label by the symbol's own trailing return-vol (std of daily returns over
    # vol_window, causal/backward-looking). The head then predicts ABNORMAL coming downside
    # relative to this stock's normal noise — so it HOLDS through ordinary high-vol wiggles and
    # SELLS only on a genuine regime break (mirrors VelocityExitRegressionTarget.vol_normalize).
    # The normalizer uses only PAST returns (no leak); the forward span is unchanged (horizon).
    # False = plain downside ratio (backward-compatible default).
    vol_normalize: bool = False
    vol_window: int = 20

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {self.vol_window}")

    def fwd_downside(self, close: pd.Series) -> pd.Series:
        # Deepest forward close over [t+1 .. t+horizon]; NaN once the full window
        # runs past the series end (mirrors forward-return's tail-NaN).
        shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        fwd_min = pd.concat(shifts, axis=1).min(axis=1)
        fwd_min = fwd_min.where(close.shift(-self.horizon).notna())
        label = 1.0 - fwd_min / close
        if self.vol_normalize:
            # Causal trailing return-vol of THIS symbol (past returns only -> no leak).
            vol = close.pct_change().rolling(self.vol_window, min_periods=2).std()
            label = label / (vol + 1e-4)
        return label

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["fwd_downside"] = out.groupby("symbol")[close_col].transform(self.fwd_downside)
        out["target"] = out["fwd_downside"]
        return out
