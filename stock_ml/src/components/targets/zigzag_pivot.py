"""Zigzag-pivot oracle target.

Labels each bar by whether a confirmed swing low (or high) will form within
fwd_window bars forward. This is a hindsight oracle for training — it identifies
"true" wave-start entry zones defined by actual zigzag swing bottoms, rather
than arbitrary forward-return thresholds.

Positive class (+1): A confirmed zigzag swing bottom appears within [1, fwd_window]
    bars after bar t, AND the subsequent up-leg gains >= min_gain before the next
    confirmed swing top. This selects bars where a genuine wave start is imminent.

Negative class (-1, if n_classes == 3): A confirmed swing TOP appears within
    [1, fwd_window/2] bars after t, AND price subsequently drops >= min_loss.
    This marks "entering near the top" — the avoidance zone.

Neutral (0): Everything else.

Key advantage over forward-return oracle: instead of asking "does price rise X%
in N days", asks "is there a confirmed swing structure that validates this as a
wave start" — aligns training ground truth with actual technical structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _find_zigzag_pivots(
    close: np.ndarray,
    pct: float,
    min_leg: int,
) -> list[tuple[int, float, int]]:
    """Run causal zigzag and return confirmed pivots: (extreme_idx, extreme_price, type).

    type: +1 = peak (swing high), -1 = bottom (swing low).
    Confirmed at the bar where reversal >= pct occurs (not at the extreme bar).
    Returns pivots ordered by their extreme_idx (the actual swing price bar).
    """
    n = len(close)
    if n == 0:
        return []

    direction = 0
    ext_idx, ext_price = 0, close[0]
    last_confirmed_idx = 0
    pivots: list[tuple[int, float, int]] = []

    for i in range(1, n):
        price = close[i]
        if direction >= 0 and price > ext_price:
            ext_price, ext_idx, direction = price, i, 1
        elif direction <= 0 and price < ext_price:
            ext_price, ext_idx, direction = price, i, -1
        elif direction == 1 and price <= ext_price * (1.0 - pct):
            if ext_idx - last_confirmed_idx >= min_leg:
                pivots.append((ext_idx, ext_price, +1))
                last_confirmed_idx = ext_idx
            direction, ext_price, ext_idx = -1, price, i
        elif direction == -1 and price >= ext_price * (1.0 + pct):
            if ext_idx - last_confirmed_idx >= min_leg:
                pivots.append((ext_idx, ext_price, -1))
                last_confirmed_idx = ext_idx
            direction, ext_price, ext_idx = 1, price, i

    return pivots


class ZigzagPivotTarget:
    """Oracle entry/exit label from confirmed zigzag swing pivots.

    Config example::

        target:
          type: zigzag_pivot
          pct: 0.06          # zigzag reversal threshold (6%)
          min_leg: 3          # minimum bars per swing leg
          fwd_window: 15      # bars forward to look for swing bottom
          min_gain: 0.04      # up-leg after bottom must gain >= 4%
          min_loss: 0.04      # down-leg after top must drop >= 4%
          classes: 3
    """

    name = "zigzag_pivot"

    def __init__(
        self,
        pct: float = 0.06,
        min_leg: int = 3,
        fwd_window: int = 15,
        min_gain: float = 0.04,
        min_loss: float = 0.04,
        n_classes: int = 3,
    ) -> None:
        self.pct = pct
        self.min_leg = min_leg
        self.fwd_window = fwd_window
        self.min_gain = min_gain
        self.min_loss = min_loss
        self.n_classes = n_classes

    @property
    def num_classes(self) -> int:
        return self.n_classes

    @property
    def supports_exit_labels(self) -> bool:
        return False

    def generate_entry_labels(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].values.astype(float)
        n = len(close)
        targets = np.zeros(n, dtype=float)

        pivots = _find_zigzag_pivots(close, self.pct, self.min_leg)
        if not pivots:
            return pd.Series(targets, index=df.index)

        # Build sorted list of bottoms and tops with their extreme indices
        bottoms = [(idx, price) for idx, price, ptype in pivots if ptype == -1]
        tops = [(idx, price) for idx, price, ptype in pivots if ptype == +1]

        # Build gain map: for each bottom, what's the subsequent up-leg gain?
        # Find the next top after each bottom and compute the leg return.
        bottom_gain: dict[int, float] = {}
        top_idx_set = {idx for idx, _ in tops}
        all_sorted = sorted(pivots, key=lambda x: x[0])
        for k, (idx, price, ptype) in enumerate(all_sorted):
            if ptype == -1:
                # Find next pivot (should be a top)
                for idx2, price2, ptype2 in all_sorted[k + 1 :]:
                    if ptype2 == +1:
                        bottom_gain[idx] = (price2 - price) / price if price > 0 else 0.0
                        break

        top_loss: dict[int, float] = {}
        for k, (idx, price, ptype) in enumerate(all_sorted):
            if ptype == +1:
                for idx2, price2, ptype2 in all_sorted[k + 1 :]:
                    if ptype2 == -1:
                        top_loss[idx] = (price - price2) / price if price > 0 else 0.0
                        break

        # Label +1: a qualified bottom will form within [t+1, t+fwd_window]
        for bot_idx, _ in bottoms:
            gain = bottom_gain.get(bot_idx, 0.0)
            if gain < self.min_gain:
                continue
            # Mark bars from which this bottom is within the look-forward window
            start = max(0, bot_idx - self.fwd_window)
            end = bot_idx  # inclusive: bar AT the bottom is also a valid entry
            for t in range(start, end + 1):
                if targets[t] == 0:  # don't overwrite -1
                    targets[t] = 1.0

        # Label -1: a qualified top will form within [t+1, fwd_window//2] (sell zone)
        if self.n_classes == 3:
            short_fwd = max(1, self.fwd_window // 2)
            for top_idx_val, _ in tops:
                loss = top_loss.get(top_idx_val, 0.0)
                if loss < self.min_loss:
                    continue
                start = max(0, top_idx_val - short_fwd)
                end = top_idx_val
                for t in range(start, end + 1):
                    if targets[t] == 0:
                        targets[t] = -1.0

        # NaN out the tail (last fwd_window bars have no confirmed forward pivot)
        # Use the last confirmed pivot as the boundary
        if pivots:
            last_pivot_idx = pivots[-1][0]
            tail_start = last_pivot_idx + 1
            for t in range(tail_start, n):
                if targets[t] == 0:
                    targets[t] = np.nan
        # Mark the very last fwd_window bars as NaN regardless
        for t in range(max(0, n - self.fwd_window), n):
            targets[t] = np.nan

        return pd.Series(targets, index=df.index)

    def generate_exit_labels(self, *args, **kwargs):
        return None
