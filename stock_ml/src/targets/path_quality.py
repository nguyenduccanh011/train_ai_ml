"""Path-aware entry-quality regression targets (entry-side).

Three scalar regression targets that score "is this a good entry" by the SHAPE of
the forward path, not just its endpoint return. All look `horizon` bars ahead and
leave the last `horizon` rows per symbol as NaN (window not yet observable), so
train folds never see the tail and the fail-loud ``require_no_nan`` guard holds.
All use close-to-close (consistent with ForwardDrawdownRegressionTarget) and write
ONLY the ``target`` column — features stay past-only.

  * mfe_regression       — max favorable excursion: the best up-move reachable in
                           the next h bars. High = strong upside potential.
  * reward_risk_regression — MFE / (MAE + floor): big upside vs small drawdown. This
                           is the "Entry B beats Entry A" quality signal (both end +5%
                           but B never drew down). Clipped to keep the scale sane.
  * multi_horizon_return — mean forward return across several horizons. High = price
                           rises CONSISTENTLY across 3/5/10/20 bars (smooth uptrend,
                           a proxy for trend_smoothness) rather than one lucky spike.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _fwd_max(close: pd.Series, h: int) -> pd.Series:
    """Max forward close over [t+1 .. t+h]; NaN once the window runs past the end."""
    shifts = [close.shift(-k) for k in range(1, h + 1)]
    fwd = pd.concat(shifts, axis=1).max(axis=1)
    return fwd.where(close.shift(-h).notna())


def _fwd_min(close: pd.Series, h: int) -> pd.Series:
    shifts = [close.shift(-k) for k in range(1, h + 1)]
    fwd = pd.concat(shifts, axis=1).min(axis=1)
    return fwd.where(close.shift(-h).notna())


@dataclass(frozen=True)
class MFERegressionTarget:
    """Max favorable excursion over the next `horizon` bars (close-based), as regression."""

    horizon: int = 10

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

    def _mfe(self, close: pd.Series) -> pd.Series:
        return _fwd_max(close, self.horizon) / close - 1.0

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._mfe)
        return out


@dataclass(frozen=True)
class RewardRiskRegressionTarget:
    """MFE / (MAE + floor) over the next `horizon` bars — path-aware reward/risk.

    floor stops the ratio exploding when the drawdown is ~0; clip caps the upper tail
    so a handful of zero-drawdown bars don't dominate the regression.
    """

    horizon: int = 10
    floor: float = 0.03
    clip: float = 10.0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.floor <= 0:
            raise ValueError(f"floor must be > 0, got {self.floor}")
        if self.clip <= 0:
            raise ValueError(f"clip must be > 0, got {self.clip}")

    def _rr(self, close: pd.Series) -> pd.Series:
        mfe = (_fwd_max(close, self.horizon) / close - 1.0).clip(lower=0.0)
        mae = (1.0 - _fwd_min(close, self.horizon) / close).clip(lower=0.0)  # drop magnitude >=0
        rr = mfe / (mae + self.floor)
        return rr.clip(upper=self.clip)

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._rr)
        return out


@dataclass(frozen=True)
class MultiHorizonReturnTarget:
    """Mean forward return across several horizons — rewards consistent multi-bar rises."""

    horizons: tuple[int, ...] = (3, 5, 10, 20)

    def __post_init__(self) -> None:
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError(f"horizons must be non-empty positive ints, got {self.horizons}")

    def _mhr(self, close: pd.Series) -> pd.Series:
        cols = [close.shift(-h) / close - 1.0 for h in self.horizons]
        mean = pd.concat(cols, axis=1).mean(axis=1)
        # tail-NaN at the LONGEST horizon so no row uses an unobservable window
        return mean.where(close.shift(-max(self.horizons)).notna())

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._mhr)
        return out
