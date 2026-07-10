"""Down-leg depth regression target (sell-side, pivot-anchored).

Motivation: fixed-horizon downside targets (forward_drawdown h10/h40, reward_risk
h10) cut the forward window at a constant bar count, so a learned exit head trained
on them goes SILENT in the middle of a deep decline (the steep part already passed,
the next 10-40 bars look mild). The mechanical ``downleg`` force-gate beats them only
because it keeps selling through the WHOLE leg. This target gives the head a label
that "sees the whole leg": the bars belonging to a confirmed zigzag down-leg are lit
up with the *remaining* fractional drop to that leg's trough, and every other bar
(up-legs, ranges) is zero. The head can then learn to fire as a peak rolls over and
stay loud through the decline — a learned trailing stop, no force-gate.

For each bar ``t``:
  - Detect pivots with a percentage-reversal zigzag on close (same engine as the
    zigzag target).
  - For each confirmed PEAK followed by a confirmed BOTTOM, every bar t in
    [peak .. min(bottom, peak+max_span)] is inside that down-leg.
  - label(t) = max(0, 1 - min(close[t .. min(leg_end, t+max_span)]) / close[t])
    = the deepest further drop from t before the leg's trough (capped at max_span).
  - All bars NOT inside a down-leg get 0.

High label = price is in / entering a real down-leg → SELL. The dual-ML dispatch
sells when the exit head's z-score of this prediction exceeds the threshold, so the
sign matches forward_drawdown (high = sell).

Leakage: the trough/bottom is future-derived (a pivot confirms only after a reversal),
exactly like the zigzag and forward_drawdown labels — this writes ONLY the target
column, features stay past-only. The forward dependence is bounded to ``max_span``
bars; the last ``max_span`` rows per symbol are left NaN (window not yet observable,
mirrors forward_drawdown's tail-NaN), and the train/test gap is sized from
``max_span`` (see registry.target_forward_span / required_gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_ml.src.targets.zigzag import _zigzag_pivots


def _downleg_depth(
    close: np.ndarray, pct: float, max_span: int, min_leg_bars: int, peak_decay: float
) -> np.ndarray:
    """Per-bar down-leg sell label (0 off-leg).

    peak_decay == 0: remaining drop to the leg trough from each bar (loud through the
        whole leg — sells late/at-bottom).
    peak_decay > 0: the leg's FULL depth (from the peak) decayed exp(-(t-peak)/peak_decay)
        — high only at the leg top, fading fast → fires EARLY at the roll-over, quiet
        mid/late-leg (don't sell the bottom). Only confirmed real legs are lit (unlike
        forward_drawdown, which also lights in-uptrend pullback tops).
    """
    n = len(close)
    label = np.zeros(n, dtype=np.float32)
    if n == 0:
        return label
    bottoms, peaks = _zigzag_pivots(close, pct, min_leg_bars)
    piv = sorted([(i, "b") for i in bottoms] + [(i, "p") for i in peaks])
    for k, (idx, typ) in enumerate(piv):
        if typ != "p" or k + 1 >= len(piv):
            continue  # need a confirmed bottom after this peak to bound the leg
        b_idx = piv[k + 1][0]
        leg_end = min(b_idx, idx + max_span)
        if peak_decay > 0.0:
            if close[idx] <= 0:
                continue
            full_depth = 1.0 - close[idx : leg_end + 1].min() / close[idx]
            full_depth = full_depth if full_depth > 0.0 else 0.0
            for t in range(idx, leg_end + 1):
                val = full_depth * np.exp(-(t - idx) / peak_decay)
                if val > label[t]:
                    label[t] = val
        else:
            for t in range(idx, leg_end + 1):
                hi = min(leg_end, t + max_span)
                if close[t] <= 0:
                    continue
                trough = close[t : hi + 1].min()
                drop = 1.0 - trough / close[t]
                if drop > label[t]:
                    label[t] = drop if drop > 0.0 else 0.0
    # Tail-NaN: the last max_span bars cannot observe their full forward window.
    if max_span > 0 and n > 0:
        label[max(0, n - max_span):] = np.nan
    return label


@dataclass(frozen=True)
class DownlegDepthRegressionTarget:
    """Remaining drop to the next zigzag down-leg trough (regression, high = sell).

    Args:
        pct: reversal threshold defining a swing (e.g. 0.06 = 6% leg).
        max_span: cap on how far forward the leg/trough is measured (bars). Also
            bounds the forward-label dependence for the leakage gap; keep <= 40 so
            required_gap (2*span+5) stays within the standard 85-day gap.
        min_leg_bars: drop swings shorter than this many bars from the previous
            confirmed pivot (noise filter); 0 disables it.
    """

    pct: float = 0.06
    max_span: int = 40
    min_leg_bars: int = 0
    peak_decay: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.pct < 1.0):
            raise ValueError(f"pct must be in (0,1), got {self.pct}")
        if self.max_span < 1:
            raise ValueError(f"max_span must be >= 1, got {self.max_span}")
        if self.min_leg_bars < 0:
            raise ValueError(f"min_leg_bars must be >= 0, got {self.min_leg_bars}")
        if self.peak_decay < 0:
            raise ValueError(f"peak_decay must be >= 0, got {self.peak_decay}")

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        if close_col not in df.columns:
            raise ValueError(f"df must contain '{close_col}'")
        out = df.copy()

        def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
            close = g[close_col].to_numpy(dtype=np.float64)
            g = g.copy()
            g["target"] = _downleg_depth(
                close, self.pct, self.max_span, self.min_leg_bars, self.peak_decay
            )
            return g

        out = out.groupby("symbol", group_keys=False).apply(_per_symbol)
        return out
