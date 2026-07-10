"""Continuation-entry regression target (buy-side).

Teaches the entry regressor to want *good continuation entries* (breakout from a
base, pullback inside an uptrend, momentum continuation) instead of swing bottoms.
The zigzag-bottom target scores the falling approach to a low as high, so the model
learns "the harder it drops, the more I buy" (a falling knife) and ignores trending
breakouts (no pct-reversal pivot exists in a clean uptrend). This target fixes both.

For each row `t` over a forward window of `horizon` bars:
  fwd_return(t)   = close[t + horizon] / close[t] - 1
  fwd_downside(t) = 1 - min_{k=1..h}( close[t+k] / close[t] )     # deepest coming drop, >= 0
  base(t)         = fwd_return(t) - penalty * fwd_downside(t)     # a "clean rise" score

`base` already rewards a smooth advance and punishes a gain that only arrives after a
deep intra-window plunge. On top of that we *shape the label by trend context* so a
positive reward only survives when the bar is a continuation/breakout setup:

  sma(t)     = trailing mean of close over `trend_window` bars            # past-only, causal
  trend_ok(t)= close[t] > sma(t)   ( and rising sma if require_rising )
  target(t)  = base(t)                       if trend_ok(t)
             = min( base(t), 0 )             otherwise

So in an uptrend pullback or a breakout that holds above the base, the full
penalized return is rewarded; in a downtrend the positive part is clipped away
(only the negative "this was a knife" signal remains). The regressor therefore
learns to output a high entry score ONLY for continuation entries.

Note: ``trend_ok`` is a *label-shaping gate computed from past-only closes* — it is
NOT a runtime entry rule. Inference at bar `t` still uses the model's past-only
features; this gate only decides what the model is trained to want. The label is
intentionally future-derived (forward window) like every other regression target,
so feature columns are never touched.

Mirrors ForwardReturnPenalizedRegressionTarget's tail-NaN (last `horizon` rows per
symbol are NaN — the window is not yet observable), keeping the fail-loud
``require_no_nan`` guard intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ContinuationEntryRegressionTarget:
    """Drawdown-penalized forward return, trend-gated to continuation entries (float ∈ ℝ)."""

    horizon: int = 10
    penalty: float = 1.0
    trend_window: int = 50
    require_rising: bool = False

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.penalty < 0:
            raise ValueError(f"penalty must be >= 0, got {self.penalty}")
        if self.trend_window < 1:
            raise ValueError(f"trend_window must be >= 1, got {self.trend_window}")

    def _label(self, close: pd.Series) -> pd.Series:
        fwd_return = close.shift(-self.horizon) / close - 1.0
        shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        fwd_min = pd.concat(shifts, axis=1).min(axis=1)
        # NaN once the full window runs past the series end (mirrors forward-return).
        fwd_min = fwd_min.where(close.shift(-self.horizon).notna())
        fwd_downside = 1.0 - fwd_min / close
        base = fwd_return - self.penalty * fwd_downside

        sma = close.rolling(self.trend_window, min_periods=1).mean()
        trend_ok = close > sma
        if self.require_rising:
            trend_ok = trend_ok & (sma.diff() > 0)
        # In-trend: keep base (NaN tail preserved). Out-of-trend: clip positives to 0
        # so only the "this was a knife" negative signal survives.
        return base.where(trend_ok, base.clip(upper=0.0))

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._label)
        return out
