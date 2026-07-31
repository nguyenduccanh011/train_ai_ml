"""Triple-barrier entry target (López de Prado, simplified, no meta-labeling).

For each bar ``t`` scan forward up to ``horizon`` bars:
  * upper barrier  close[t]*(1+pt)  hit first  -> label 1.0  (a clean profitable buy)
  * lower barrier  close[t]*(1-sl)  hit first  -> label 0.0  (stopped out)
  * neither within the window (vertical barrier) -> ``neutral_label`` (default 0.0)

Unlike a raw forward-return target this is ASYMMETRIC and path-aware: it rewards
bars where price reaches a profit target *before* a drawdown stop, which is exactly
"is this a good entry". Output is a float in [0,1], trained as regression (the dual-ML
recombine strategy z-scores it, so the binary scale is fine).

Tail convention (matches ForwardReturnRegressionTarget): the last ``horizon`` rows
per symbol have no fully-observable forward window -> NaN, so train folds never see
them and the fail-loud ``require_no_nan`` guard stays intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierTarget:
    """Path-aware profit-before-stop entry label (regression in [0,1])."""

    horizon: int = 20
    pt: float = 0.10  # profit-target barrier, fraction
    sl: float = 0.05  # stop barrier, fraction
    direction: str = "long"  # "long": label 1 if +pt before -sl (good BUY);
    #                          "short": label 1 if -pt before +sl (good SELL — drop first)
    neutral_label: float = 0.0  # label when neither barrier is touched in the window
    target_col: str = "target"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if not (self.pt > 0 and self.sl > 0):
            raise ValueError(f"pt and sl must be > 0, got pt={self.pt}, sl={self.sl}")
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long'/'short', got {self.direction!r}")

    def _label_close(self, close: np.ndarray) -> np.ndarray:
        n = len(close)
        out = np.full(n, np.nan, dtype=np.float32)
        if n == 0:
            return out
        h = self.horizon
        # "profit" barrier and "stop" barrier depend on direction. For long, profit is the
        # up-move (+pt) and stop is the down-move (-sl). For short, profit is the down-move
        # (-pt) and stop is the up-move (+sl). label 1 = profit barrier hit first.
        up_thr = self.pt if self.direction == "long" else self.sl
        dn_thr = self.sl if self.direction == "long" else self.pt
        prof_first = np.full(n, np.inf)  # first bar the profit barrier is touched
        stop_first = np.full(n, np.inf)  # first bar the stop barrier is touched
        for k in range(1, h + 1):
            if k >= n:
                break
            r = close[k:] / close[:-k] - 1.0  # ret over k bars for i in 0..n-k-1
            idx = np.arange(n - k)
            up = r >= up_thr
            dn = r <= -dn_thr
            prof = up if self.direction == "long" else dn
            stop = dn if self.direction == "long" else up
            new_p = prof & np.isinf(prof_first[idx])
            prof_first[idx[new_p]] = k
            new_s = stop & np.isinf(stop_first[idx])
            stop_first[idx[new_s]] = k
        # observable window: i has a full horizon iff i + h < n
        obs = np.arange(n) < (n - h)
        lab = np.where(
            prof_first < stop_first,
            1.0,
            np.where(np.isinf(prof_first) & np.isinf(stop_first), self.neutral_label, 0.0),
        )
        out[obs] = lab[obs].astype(np.float32)
        return out

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        if close_col not in df.columns:
            raise ValueError(f"df must contain '{close_col}'")
        out = df.copy()

        def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            g[self.target_col] = self._label_close(g[close_col].to_numpy(dtype=np.float64))
            return g

        return out.groupby("symbol", group_keys=False).apply(_per_symbol)
