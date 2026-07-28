"""Cross-sectional forward-return-rank target.

For each row `t`:
  fwd_return(t) = close[t + horizon] / close[t] - 1
  target(t)     = cross-sectional rank of fwd_return within that DATE, centered to [-0.5, +0.5]

Rewards predicting which names OUTPERFORM THEIR PEERS (relative), not absolute return. Motivation
(core_features IC scan + [[cross-sectional-survives-zscoring]]): absolute forward return flips sign
in dead years (2024/2026) as the market beta dominates; the cross-sectional RANK of forward return
is regime-robust (relative ordering survives the flat/choppy regimes). Aligns the entry TARGET with
the cross-sectional RS FEATURES (which won). Purely price-derived, causal, no external data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForwardReturnCrossSectionalTarget:
    """Predict the within-day cross-sectional rank of forward return (regression, centered)."""

    horizon: int = 10

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns or "date" not in df.columns:
            raise ValueError("df must contain 'symbol' and 'date'")
        out = df.copy()
        out["fwd_return"] = out.groupby("symbol")[close_col].transform(
            lambda s: s.shift(-self.horizon) / s - 1.0
        )
        # cross-sectional rank within each date; NaN forward-returns (tail) stay NaN and are dropped.
        out["target"] = out.groupby("date")["fwd_return"].transform(
            lambda x: x.rank(pct=True) - 0.5
        )
        return out
