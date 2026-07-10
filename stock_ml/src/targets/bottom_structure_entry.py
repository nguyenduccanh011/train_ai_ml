"""Bottom-structure entry regression target (realizability-aware, buy-side).

Motivation (champion 2417 X-ray, 2026-06-19): the existing reversal_entry head rewards horizon-END
forward RETURN gated to a dip. Across this strategy line, forward-return / forward-PEAK targets are
"masked": the head predicts the peak well (IC_fwdpeak +0.18) but it does NOT realize (the exit gives
the peak back; realized IC ~0). Feeding that head's signal harder (raw-vs-z, csrank) was A/B-refuted.
So this target keys NOT on the un-realizable end-of-horizon return but on REALIZABLE entry quality:

  reward a CORRECTION FOOT (close below trend AND a confirmed turn, not a falling knife) that is
  followed by an EARLY, CLEAN advance with LITTLE drawdown; and PENALIZE
    - "parked" entries: over park_window the price goes nowhere / down (capital idle), and
    - extended / top buys: positive reward clipped away where the bar is not a foot.

For each row t (all forward terms are past-only labels; inference uses past-only features):
  short window k=1..h:
    fwd_ret(t)  = close[t+h]/close[t] - 1                      # early advance (short h, realizable)
    fwd_dd(t)   = 1 - min_k close[t+k]/close[t]                # early drawdown (MAE proxy), >= 0
    base(t)     = fwd_ret(t) - penalty * fwd_dd(t)            # quick gain with little pain
  parked window j=1..park_window:
    pk_max(t)   = max_j close[t+j]/close[t] - 1
    pk_ret(t)   = close[t+park_window]/close[t] - 1
    parked(t)   = 1 if pk_max < park_floor AND pk_ret <= 0 else 0   # never advanced, ended flat/down
    base(t)    -= park_penalty * parked(t)
  structural FOOT gate:
    sma(t)      = trailing mean of close over dip_window                      # causal
    foot(t)     = close[t] < sma(t)  [ AND rsi14 turning up if require_turn ]  [ AND uptrend if trend_window ]
    target(t)   = base(t)            if foot(t)
                = min(base(t), 0)    otherwise   # only the "don't chase" negative survives off-foot

Forward span = max(horizon, park_window) (declared in registry.target_forward_span so the
walk-forward split gap covers it — fail-loud, no silent under-gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BottomStructureEntryRegressionTarget:
    """Realizable foot-entry quality: early clean advance, parked-penalized, foot-gated (float ∈ ℝ)."""

    horizon: int = 8
    penalty: float = 1.5
    dip_window: int = 50
    require_turn: bool = True
    rsi_window: int = 14
    park_window: int = 20
    park_floor: float = 0.04
    park_penalty: float = 0.5
    trend_window: int = 0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.penalty < 0:
            raise ValueError(f"penalty must be >= 0, got {self.penalty}")
        if self.dip_window < 1:
            raise ValueError(f"dip_window must be >= 1, got {self.dip_window}")
        if self.rsi_window < 2:
            raise ValueError(f"rsi_window must be >= 2, got {self.rsi_window}")
        if self.park_window < 1:
            raise ValueError(f"park_window must be >= 1, got {self.park_window}")
        if self.park_penalty < 0:
            raise ValueError(f"park_penalty must be >= 0, got {self.park_penalty}")
        if self.trend_window and self.trend_window <= self.dip_window:
            raise ValueError(
                f"trend_window ({self.trend_window}) must be > dip_window ({self.dip_window})"
            )

    def _rsi(self, close: pd.Series) -> pd.Series:
        d = close.diff()
        g = d.clip(lower=0).rolling(self.rsi_window, min_periods=self.rsi_window).mean()
        ln = (-d.clip(upper=0)).rolling(self.rsi_window, min_periods=self.rsi_window).mean()
        return 100.0 - 100.0 / (1.0 + g / (ln + 1e-9))

    def _label(self, close: pd.Series) -> pd.Series:
        # early-advance window (short horizon)
        observable = close.shift(-self.horizon).notna()
        win = pd.concat([close.shift(-k) for k in range(1, self.horizon + 1)], axis=1)
        fwd_ret = close.shift(-self.horizon) / close - 1.0
        fwd_min = win.min(axis=1).where(observable)
        fwd_dd = 1.0 - fwd_min / close
        base = fwd_ret - self.penalty * fwd_dd

        # parked penalty over the longer window
        pk_obs = close.shift(-self.park_window).notna()
        pk_win = pd.concat([close.shift(-k) for k in range(1, self.park_window + 1)], axis=1)
        pk_max = pk_win.max(axis=1).where(pk_obs) / close - 1.0
        pk_ret = (close.shift(-self.park_window) / close - 1.0).where(pk_obs)
        # NaN (not 0) where the parked window is not fully observable, so the tail-NaN matches the
        # declared forward span (park_window) and no row is labelled with a silently-zeroed penalty.
        parked = ((pk_max < self.park_floor) & (pk_ret <= 0.0)).astype(float).where(pk_obs)
        base = base - self.park_penalty * parked

        # structural foot gate (causal, past-only label shaping)
        sma = close.rolling(self.dip_window, min_periods=1).mean()
        gate = close < sma
        if self.require_turn:
            gate = gate & (self._rsi(close).diff(2) > 0)
        if self.trend_window:
            long_sma = close.rolling(self.trend_window, min_periods=1).mean()
            gate = gate & (close > long_sma)
        return base.where(gate, base.clip(upper=0.0))

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._label)
        return out
