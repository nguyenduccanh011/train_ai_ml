"""V19 position sizing helper — exact port from legacy backtest_v19_3 lines 2912-2947."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_v19_size(
    ind: dict[str, Any],
    i: int,
    *,
    trend: str,
    entry_score: int,
    vshape_entry: bool,
    regime_cfg: dict[str, Any],
) -> float:
    """Compute v19_3 position_size given trend/score/vshape flag and regime params.

    Returns float in [0.25, 1.0]. Bypass for vshape_entry yields 0.50 base.
    """
    close = ind["close"]
    opn = ind["open"]
    atr14 = ind["atr14"]
    ret_5d = ind["ret_5d"]
    ret5_hot = regime_cfg["ret5_hot"]
    size_mult = regime_cfg["size_mult"]

    atr_ratio = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

    if vshape_entry:
        position_size = 0.50
    elif trend == "strong" and entry_score >= 4:
        position_size = 0.95
    elif trend == "strong" and entry_score >= 3:
        position_size = 0.90
    elif trend == "moderate" and entry_score >= 3:
        position_size = 0.50
    elif trend == "weak":
        position_size = 0.30
    else:
        position_size = 0.50

    if atr_ratio > 0.055:
        position_size = min(position_size, 0.35)
    elif atr_ratio > 0.040:
        position_size = min(position_size, 0.50)

    if trend == "weak":
        position_size = min(position_size, 0.40)
    elif trend == "moderate":
        position_size = min(position_size, 0.70)

    if close[i] <= opn[i] and not vshape_entry:
        position_size *= 0.75

    if ret_5d[i] > ret5_hot:
        position_size = min(position_size, 0.40)

    position_size *= size_mult
    return max(0.25, min(position_size, 1.0))
