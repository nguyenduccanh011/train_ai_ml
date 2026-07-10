"""Zigzag pivot soft-label target.

Labels each bar by its proximity to a confirmed medium-term zigzag pivot, so an
ML regressor can learn the *shape* of swing bottoms (entry) or peaks (exit) and
predict them in real time before the pivot is confirmed.

Pivots are detected with a percentage-reversal zigzag on close:
  - Track the running extreme since the last confirmed pivot.
  - When price reverses by >= `pct` from that extreme, the extreme bar is
    confirmed as a pivot (a bottom if we were tracking a low, a peak if a high).

For each bar `t` the soft label is the proximity to the nearest *same-type*
pivot, decaying with bar distance:
  label(t) = max over same-type pivots p of  exp(-|t - t_p| / tau)
so label = 1 at the pivot bar and decays smoothly away. Output is a float in
[0, 1] → trained as regression.

Leakage note: the LABEL is intentionally future-derived (a pivot at t is only
known after price reverses). That is normal supervised learning. This module
writes ONLY the `target`/`exit_target` column and never touches feature columns,
so inference at bar t still uses past-only features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _zigzag_pivots(
    close: np.ndarray, pct: float, min_leg_bars: int = 0
) -> tuple[list[int], list[int]]:
    """Return (bottom_indices, peak_indices) of a percentage-reversal zigzag.

    A pivot is confirmed once price reverses by >= pct from the running extreme.
    The last in-progress leg is not confirmed (no future reversal yet), matching
    how a live zigzag behaves.

    ``min_leg_bars`` filters out swings shorter than that many bars from the
    previous confirmed pivot (noise legs); 0 disables the filter.
    """
    n = len(close)
    bottoms: list[int] = []
    peaks: list[int] = []
    if n == 0:
        return bottoms, peaks

    # direction: +1 = currently in an up-leg (tracking a high),
    #            -1 = currently in a down-leg (tracking a low),
    #             0 = undecided at start
    direction = 0
    ext_idx = 0
    ext_price = close[0]
    last_confirmed_idx = 0  # index of the last appended pivot (or series start)

    for i in range(1, n):
        price = close[i]
        if direction >= 0 and price > ext_price:
            # extend up-leg, new running high
            ext_price = price
            ext_idx = i
            direction = 1
        elif direction <= 0 and price < ext_price:
            # extend down-leg, new running low
            ext_price = price
            ext_idx = i
            direction = -1
        elif direction == 1 and price <= ext_price * (1.0 - pct):
            # reversal down from a high -> confirm peak at ext_idx
            if ext_idx - last_confirmed_idx >= min_leg_bars:
                peaks.append(ext_idx)
                last_confirmed_idx = ext_idx
            direction = -1
            ext_price = price
            ext_idx = i
        elif direction == -1 and price >= ext_price * (1.0 + pct):
            # reversal up from a low -> confirm bottom at ext_idx
            if ext_idx - last_confirmed_idx >= min_leg_bars:
                bottoms.append(ext_idx)
                last_confirmed_idx = ext_idx
            direction = 1
            ext_price = price
            ext_idx = i

    return bottoms, peaks


def _profitable_pivots(
    close: np.ndarray,
    bottoms: list[int],
    peaks: list[int],
    direction: str,
    min_fwd_leg: float,
) -> list[int]:
    """Keep only pivots whose FORWARD leg (to the next confirmed opposite pivot)
    moves at least `min_fwd_leg`.

    A bottom is "profitable" only if the up-leg to the next peak is >= min_fwd_leg
    (a real bounce worth buying, not a weak dead-cat or a knife that keeps falling);
    a peak is "profitable" only if the down-leg to the next bottom is >= min_fwd_leg
    (worth selling). The last pivot has no confirmed forward leg yet and is dropped.

    Leakage note: like the base label this is intentionally future-derived (the
    forward leg is only known once the next pivot confirms). Features stay past-only.
    """
    piv = sorted([(i, "b") for i in bottoms] + [(i, "p") for i in peaks])
    target_type = "b" if direction == "bottom" else "p"
    keep: list[int] = []
    for k, (idx, typ) in enumerate(piv):
        if typ != target_type or k + 1 >= len(piv):
            continue
        nxt_idx = piv[k + 1][0]
        if close[idx] <= 0:
            continue
        fwd_leg = abs(close[nxt_idx] / close[idx] - 1.0)
        if fwd_leg >= min_fwd_leg:
            keep.append(idx)
    return keep


def _proximity(n: int, pivot_idx: list[int], tau: float, one_sided: str | None = None) -> np.ndarray:
    """Soft label in [0,1]: exp(-distance_to_nearest_pivot / tau) per bar.

    ``one_sided`` restricts which side of each pivot is labeled:
      None   — symmetric (default; high before AND after the pivot).
      "post" — only bars AT/AFTER the pivot (learn "the turn just happened" =
               momentum reversal; buy/sell on confirmation, slightly late but causal-clean).
      "pre"  — only bars AT/BEFORE the pivot (learn "a pivot is approaching"; anticipatory
               but looks like a falling-knife / blow-off, harder to separate from continuation).
    """
    label = np.zeros(n, dtype=np.float32)
    if not pivot_idx:
        return label
    idx = np.arange(n, dtype=np.int64)
    if one_sided is None:
        piv = np.asarray(pivot_idx, dtype=np.int64)
        dist = np.min(np.abs(idx[:, None] - piv[None, :]), axis=1)
        return np.exp(-dist.astype(np.float32) / float(tau)).astype(np.float32)
    if one_sided not in ("pre", "post"):
        raise ValueError(f"one_sided must be None/'pre'/'post', got {one_sided!r}")
    for p in pivot_idx:
        d = idx - p  # >0 after pivot, <0 before
        mask = (d >= 0) if one_sided == "post" else (d <= 0)
        val = np.zeros(n, dtype=np.float32)
        val[mask] = np.exp(-np.abs(d[mask]).astype(np.float32) / float(tau))
        np.maximum(label, val, out=label)
    return label


@dataclass(frozen=True)
class ZigzagPivotTarget:
    """Soft-label proximity to medium-term zigzag pivots (regression target).

    Args:
        direction: "bottom" (entry) or "peak" (exit) — which pivot type to label.
        pct: reversal threshold defining a swing (e.g. 0.10 = 10%).
        tau: decay scale in bars; larger = wider labeled zone around each pivot.
        min_leg_bars: drop swings shorter than this many bars from the previous
            confirmed pivot (noise filter); 0 disables it.
        min_fwd_leg: keep only pivots whose forward leg (to the next opposite pivot)
            moves >= this fraction — i.e. "profitable" bottoms (big bounce ahead) or
            peaks (big drop ahead). 0 disables it (label ALL pivots; default).
        target_col: output column name (default "target"; use "exit_target" for
            the exit slot).
    """

    direction: str = "bottom"
    pct: float = 0.10
    tau: float = 5.0
    min_leg_bars: int = 0
    min_fwd_leg: float = 0.0
    one_sided: str | None = None
    target_col: str = "target"

    def __post_init__(self) -> None:
        if self.one_sided not in (None, "pre", "post"):
            raise ValueError(f"one_sided must be None/'pre'/'post', got {self.one_sided!r}")
        if self.direction not in ("bottom", "peak"):
            raise ValueError(f"direction must be 'bottom' or 'peak', got {self.direction!r}")
        if not (0.0 < self.pct < 1.0):
            raise ValueError(f"pct must be in (0,1), got {self.pct}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.min_leg_bars < 0:
            raise ValueError(f"min_leg_bars must be >= 0, got {self.min_leg_bars}")
        if self.min_fwd_leg < 0:
            raise ValueError(f"min_fwd_leg must be >= 0, got {self.min_fwd_leg}")

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        if close_col not in df.columns:
            raise ValueError(f"df must contain '{close_col}'")
        out = df.copy()

        def _labels_per_symbol(g: pd.DataFrame) -> pd.DataFrame:
            close = g[close_col].to_numpy(dtype=np.float64)
            bottoms, peaks = _zigzag_pivots(close, self.pct, self.min_leg_bars)
            if self.min_fwd_leg > 0:
                piv = _profitable_pivots(close, bottoms, peaks, self.direction, self.min_fwd_leg)
            else:
                piv = bottoms if self.direction == "bottom" else peaks
            g = g.copy()
            g[self.target_col] = _proximity(len(close), piv, self.tau, self.one_sided)
            return g

        out = out.groupby("symbol", group_keys=False).apply(_labels_per_symbol)
        return out
