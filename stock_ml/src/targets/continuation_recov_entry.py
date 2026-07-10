"""Continuation-entry target with a recovery override (two-sided buy-side).

ContinuationEntryRegressionTarget rewards buying only when ``close > sma(trend_window)``
and clips every positive reward to 0 otherwise. Trained on a trending market this makes
the model score the established-uptrend right shoulder (near 52w-high, post-climax) high and
score washed-out bottoms structurally LOW — the model is literally trained to NOT score a
genuine reversal bottom, even one that rebounds cleanly (the documented "coincident chaser"
blind spot; corr(score, dist_to_20d_high) ~ +0.36, forward-IC ~ 0).

This target keeps the continuation reward intact but adds an OR-gate so a *confirmed,
washed-out recovery* is no longer clipped. The drawdown penalty in ``base`` already makes a
falling knife score negative, so the override only un-clips bottoms that actually rebound.

For each row `t` over a forward window of `horizon` bars:
  fwd_return(t)   = close[t + horizon] / close[t] - 1
  fwd_downside(t) = 1 - min_{k=1..h}( close[t+k] / close[t] )     # deepest coming drop, >= 0
  base(t)         = fwd_return(t) - penalty * fwd_downside(t)     # a "clean rise" score

Gates (all past-only except the forward-confirmation, like every regression label):
  trend_ok(t)   = close[t] > sma(close, trend_window)                       # continuation side
  washed_out(t) = close[t] / max(close, recov_window) - 1 <= -recov_dd      # deep pullback
  confirmed(t)  = max_{k=1..h}( close[t+k] ) / close[t] - 1 >= min_fwd_rally # real rebound
  keep(t)       = trend_ok(t)  OR  ( washed_out(t) AND confirmed(t) )
  target(t)     = base(t)              if keep(t)
                = min( base(t), 0 )    otherwise

So an uptrend pullback/breakout (trend_ok) AND a confirmed rebound off a washed-out bottom
both keep the full penalized return; extended tops and unconfirmed knives still have their
positive part clipped. The model can now learn a high entry score for BOTH continuation and
confirmed-reversal setups instead of suppressing the latter.

The gates are label-shaping (NOT runtime rules); inference still uses past-only features.
Mirrors the tail-NaN of the other forward targets (last `horizon` rows per symbol NaN),
keeping the fail-loud ``require_no_nan`` guard intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ContinuationRecovEntryRegressionTarget:
    """Continuation reward with a confirmed-washout recovery override (float ∈ ℝ)."""

    horizon: int = 8
    penalty: float = 1.0
    trend_window: int = 70
    require_rising: bool = False
    recov_window: int = 20
    recov_dd: float = 0.12
    min_fwd_rally: float = 0.05

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.penalty < 0:
            raise ValueError(f"penalty must be >= 0, got {self.penalty}")
        if self.trend_window < 1:
            raise ValueError(f"trend_window must be >= 1, got {self.trend_window}")
        if self.recov_window < 1:
            raise ValueError(f"recov_window must be >= 1, got {self.recov_window}")
        if self.recov_dd <= 0:
            raise ValueError(f"recov_dd must be > 0, got {self.recov_dd}")
        if self.min_fwd_rally < 0:
            raise ValueError(f"min_fwd_rally must be >= 0, got {self.min_fwd_rally}")

    def _label(self, close: pd.Series) -> pd.Series:
        shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        observable = close.shift(-self.horizon).notna()
        window = pd.concat(shifts, axis=1)
        fwd_return = close.shift(-self.horizon) / close - 1.0
        fwd_min = window.min(axis=1).where(observable)
        fwd_downside = 1.0 - fwd_min / close
        base = fwd_return - self.penalty * fwd_downside

        sma = close.rolling(self.trend_window, min_periods=1).mean()
        trend_ok = close > sma
        if self.require_rising:
            trend_ok = trend_ok & (sma.diff() > 0)

        # Recovery override: a bar pulled deep below its recent high (washed out) whose
        # forward window actually rallies (confirmed) — a real bottom, not a knife.
        recent_high = close.rolling(self.recov_window, min_periods=1).max()
        washed_out = (close / recent_high - 1.0) <= -self.recov_dd
        if self.min_fwd_rally > 0:
            fwd_max = window.max(axis=1).where(observable)
            confirmed = (fwd_max / close - 1.0) >= self.min_fwd_rally
        else:
            confirmed = pd.Series(True, index=close.index)
        recov_ok = washed_out & confirmed

        keep = trend_ok | recov_ok
        # Kept: full base (NaN tail preserved). Else: clip positives to 0 so only the
        # "don't chase / this was a knife" negative signal survives.
        return base.where(keep, base.clip(upper=0.0))

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._label)
        return out
