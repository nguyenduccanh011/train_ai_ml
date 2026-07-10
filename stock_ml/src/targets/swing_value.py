"""Swing-value regression target (path-dependent, asymmetric) — user's swing-penalty objective.

Encodes the swing trade objective directly as a per-bar label:
  ENTRY  (direction='buy'):  value(t) = (fwd_max - close)/close  -  alpha * (close - fwd_min)/close
      high = price RISES ahead (you can sell higher) with little adverse DROP first
      (penalizes buying a KNIFE or a TOP — exactly "buy so a later sell is higher").
  EXIT   (direction='sell'): value(t) = (close - fwd_min)/close  -  alpha * (fwd_max - close)/close
      high = price DROPS ahead (you can rebuy LOWER) with little missed RISE
      (penalizes selling into CONTINUATION — exactly "sell to rebuy lower, penalize sell-then-rebuy-higher").

fwd_max / fwd_min are over the forward window [t+1, t+horizon]. `alpha` is the asymmetric penalty weight
on the wrong-side excursion (the regime "action threshold" knob: raise it to demand the wrong-side move be
small before acting). Tail-NaN at `horizon` (window not yet observable) — mirrors ForwardDrawdownRegression.
Optional `vol_normalize` divides the label by the symbol's trailing return-vol so the head predicts the
ABNORMAL swing magnitude relative to its own noise (project finding: only VOL predicts magnitude).

Leakage note: fwd_max/fwd_min are intentionally future-derived LABELS; features stay past-only and the
train/test gap is sized from `horizon` via registry._HORIZON_TARGETS.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SwingValueRegressionTarget:
    """Asymmetric forward swing value (regression, float ∈ ℝ). direction='buy' (entry) or 'sell' (exit)."""

    horizon: int = 10
    direction: str = "buy"
    alpha: float = 1.0
    vol_normalize: bool = False
    vol_window: int = 20

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.direction not in ("buy", "sell"):
            raise ValueError(f"direction must be 'buy' or 'sell', got {self.direction!r}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {self.vol_window}")

    def _value(self, close: pd.Series) -> pd.Series:
        shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        fwd = pd.concat(shifts, axis=1)
        fwd_max = fwd.max(axis=1)
        fwd_min = fwd.min(axis=1)
        observable = close.shift(-self.horizon).notna()
        rise = fwd_max / close - 1.0
        drop = 1.0 - fwd_min / close
        if self.direction == "buy":
            label = rise - self.alpha * drop
        else:  # "sell"
            label = drop - self.alpha * rise
        label = label.where(observable)
        if self.vol_normalize:
            vol = close.pct_change().rolling(self.vol_window, min_periods=2).std()
            label = label / (vol + 1e-4)
        return label

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        if close_col not in df.columns:
            raise ValueError(f"df must contain '{close_col}'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._value)
        return out
