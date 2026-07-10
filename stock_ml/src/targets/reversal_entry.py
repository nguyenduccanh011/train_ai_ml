"""Reversal-entry regression target (buy-side, contra-phase).

The mirror image of ContinuationEntryRegressionTarget. continuation_entry rewards buying
in an UPTREND (close > trailing SMA) — which, trained on a trending market, produces a
coincident top-chaser: the entry score ends up correlated with recent gains and peaks at
price highs (see the score-phase audit; corr(score, dist_to_20d_high) ~ +0.36, forward-IC
~ 0). This target inverts the gate to teach a REVERSAL entry: score should be HIGH at a
confirmed bottom (a dip that then rebounds cleanly) and LOW at an extended/overheated top.

For each row `t` over a forward window of `horizon` bars:
  fwd_return(t)   = close[t + horizon] / close[t] - 1
  fwd_downside(t) = 1 - min_{k=1..h}( close[t+k] / close[t] )     # deepest coming drop, >= 0
  base(t)         = fwd_return(t) - penalty * fwd_downside(t)     # a "clean rebound" score

`base` rewards a smooth advance and punishes a "gain" that only arrives after a deep plunge
(so a falling knife that keeps dropping scores negative — this is what stops the model from
buying every dip). On top of that we gate by DIP context so a positive reward only survives
where the bar is pulled back below trend (a reversal setup), not extended at a high:

  sma(t)     = trailing mean of close over `dip_window` bars            # past-only, causal
  dip_ok(t)  = close[t] < sma(t)   ( and falling sma if require_falling )
  target(t)  = base(t)                       if dip_ok(t)
             = min( base(t), 0 )             otherwise

So a dip that rebounds cleanly gets the full positive reward (high score at bottoms); an
extended/overheated bar above trend has its positive part clipped away (only the "this is
risky, don't chase" negative signal remains → low score at tops). The regressor therefore
learns a high entry score ONLY for reversal/dip-rebound setups, and the model uses its
reversal-confirmation features (RSI turning up, momentum flip, candle/volume) to time WHEN
the dip becomes a buy.

`dip_ok` is a past-only label-shaping gate (NOT a runtime rule); inference still uses past-only
features. Mirrors ContinuationEntryRegressionTarget's tail-NaN (last `horizon` rows per symbol
NaN), keeping the fail-loud ``require_no_nan`` guard intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReversalEntryRegressionTarget:
    """Drawdown-penalized forward return, gated to DIP/reversal setups (float ∈ ℝ)."""

    horizon: int = 10
    penalty: float = 1.0
    dip_window: int = 50
    require_falling: bool = False
    min_fwd_rally: float = 0.0
    trend_window: int = 0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.penalty < 0:
            raise ValueError(f"penalty must be >= 0, got {self.penalty}")
        if self.dip_window < 1:
            raise ValueError(f"dip_window must be >= 1, got {self.dip_window}")
        if self.min_fwd_rally < 0:
            raise ValueError(f"min_fwd_rally must be >= 0, got {self.min_fwd_rally}")
        if self.trend_window < 0:
            raise ValueError(f"trend_window must be >= 0, got {self.trend_window}")
        if self.trend_window and self.trend_window <= self.dip_window:
            raise ValueError(
                f"trend_window ({self.trend_window}) must be > dip_window ({self.dip_window})"
            )

    def _label(self, close: pd.Series) -> pd.Series:
        shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        observable = close.shift(-self.horizon).notna()
        window = pd.concat(shifts, axis=1)
        fwd_return = close.shift(-self.horizon) / close - 1.0
        fwd_min = window.min(axis=1).where(observable)
        fwd_downside = 1.0 - fwd_min / close
        base = fwd_return - self.penalty * fwd_downside

        sma = close.rolling(self.dip_window, min_periods=1).mean()
        gate = close < sma
        if self.require_falling:
            gate = gate & (sma.diff() < 0)
        if self.trend_window:
            # UPTREND gate: only a pullback WITHIN a long-term uptrend (close above the long MA)
            # counts as a dip-buy. In a confirmed downtrend (close below long MA) the positive
            # reward is clipped → the model never learns to buy knives in a bear (the 2022 hole).
            long_sma = close.rolling(self.trend_window, min_periods=1).mean()
            gate = gate & (close > long_sma)
        if self.min_fwd_rally > 0:
            # CONFIRMATION: only reward a dip that actually rallies >= min_fwd_rally within the
            # window (a real bottom, not a knife). fwd_max measured over the same horizon.
            fwd_max = window.max(axis=1).where(observable)
            confirmed = (fwd_max / close - 1.0) >= self.min_fwd_rally
            gate = gate & confirmed
        # In a confirmed dip: keep base (NaN tail preserved). Else: clip positives to 0 so only
        # the "don't chase this" negative signal survives → low score at tops/knives.
        return base.where(gate, base.clip(upper=0.0))

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._label)
        return out
