"""Velocity-exit regression target (sell-side).

A turnover-aware generalisation of ``RiskExitRegressionTarget``. ``risk_exit`` holds
as long as *any* upside remains anywhere in the forward window, so on a slow grind up
it never sells — the position becomes a buy-and-hold (the live champion's ~158-bar
holds). This target keeps risk_exit's "don't sell into a real run" behaviour but stops
rewarding upside that only arrives *far* in the future: downside is measured over the
full ``horizon`` H, upside only over a short ``upside_horizon`` U <= H.

For each row `t`:
  fwd_downside(t)   = 1 - min_{k=1..H}( close[t+k] / close[t] )     # all coming risk, >= 0
  fwd_upside_near(t)= max_{k=1..U}( close[t+k] / close[t] ) - 1     # only NEAR upside, >= 0
  target(t)         = fwd_downside(t) - fwd_upside_near(t)

High value = sell. Behaviour vs risk_exit:
  - fast pop ahead (within U bars)  -> near-upside high -> target low  -> HOLD the move
  - slow grind (gain only past U)   -> near-upside ~0   -> target ~risk -> SELL (free capital)
  - imminent drop                   -> downside high    -> target high  -> SELL

So a position that only pays off slowly is exited early and the freed capital can chase a
faster setup, lifting trade count and pnl-per-hold velocity. With U == H this reduces
*exactly* to RiskExitRegressionTarget (continuity / backward-compat sanity).

Tail-NaN at the longest horizon H (last H rows per symbol unobservable), keeping the
fail-loud ``require_no_nan`` guard intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VelocityExitRegressionTarget:
    """Forward downside (horizon H) minus near-forward upside (horizon U<=H), as regression."""

    horizon: int = 20
    upside_horizon: int = 5
    # VOL-NORMALIZE (project deep-finding: "only VOL predicts MAGNITUDE"): divide the raw
    # downside-minus-near-upside label by the symbol's own trailing return-vol (std of daily
    # returns over vol_window, causal/backward-looking). The head then predicts ABNORMAL coming
    # downside relative to this stock's normal noise — so it HOLDS through ordinary high-vol
    # wiggles (a 5% drop on an 8%-vol name is normal -> low target) and SELLS only on a genuine
    # regime break (a 5% drop on a 2%-vol name is 2.5 sigma -> high target). Targets the
    # clip-winners wall (plain ratio over-sells high-vol runners). The normalizer uses only PAST
    # returns (no leak); the forward span is unchanged (still horizon H). False = plain ratio.
    vol_normalize: bool = False
    vol_window: int = 20

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.upside_horizon < 1:
            raise ValueError(f"upside_horizon must be >= 1, got {self.upside_horizon}")
        if self.upside_horizon > self.horizon:
            raise ValueError(
                f"upside_horizon ({self.upside_horizon}) must be <= horizon ({self.horizon})"
            )
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {self.vol_window}")

    def _label(self, close: pd.Series) -> pd.Series:
        down_shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        up_shifts = [close.shift(-k) for k in range(1, self.upside_horizon + 1)]
        # NaN once the full (longest) window runs past the series end.
        observable = close.shift(-self.horizon).notna()
        fwd_min = pd.concat(down_shifts, axis=1).min(axis=1).where(observable)
        fwd_max_near = pd.concat(up_shifts, axis=1).max(axis=1).where(observable)
        fwd_downside = 1.0 - fwd_min / close
        fwd_upside_near = fwd_max_near / close - 1.0
        label = fwd_downside - fwd_upside_near
        if self.vol_normalize:
            # Causal trailing return-vol of THIS symbol (past returns only -> no leak).
            vol = close.pct_change().rolling(self.vol_window, min_periods=2).std()
            label = label / (vol + 1e-4)
        return label

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._label)
        return out
