from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.components.base import FusionResult


def format_date(value: object) -> str:
    ts = pd.Timestamp(value)
    if ts.time() == pd.Timestamp(ts.date()).time():
        return ts.date().isoformat()
    return ts.isoformat()


def track_result(res: FusionResult, counters: Counter[str]) -> None:
    key = res.metadata.get("counter")
    if isinstance(key, str) and key:
        counters[key] += 1
    bulk = res.metadata.get("counters")
    if isinstance(bulk, dict):
        for bulk_key, value in bulk.items():
            if isinstance(bulk_key, str) and isinstance(value, int):
                counters[bulk_key] += value


def atr_stop(ind: dict[str, Any], i: int) -> float:
    atr14 = ind["atr14"]
    close = ind["close"]
    if not np.isnan(atr14[i]) and close[i] > 0:
        return max(0.025, min(1.8 * atr14[i] / close[i], 0.06))
    return 0.04
